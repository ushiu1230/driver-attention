"""Preview mirror ROI geometry or evaluate frozen saliency checkpoints (manual CLI)."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from utils.mirror_roi import (read_roi, sha256, gaze_label, roi_mask,
                              choose_threshold, binary_metrics, pr_metrics)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")


def preview(args):
    config = read_roi(args.roi, require_confirmed=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = sorted(Path(args.root).rglob("sim_gaze_df.csv"))
    if not records:
        raise ValueError("No recording CSV files found.")
    selected = np.unique(np.linspace(0, len(records)-1, min(args.records, len(records))).astype(int))
    index = []
    for record_index in selected:
        rec = records[record_index].parent
        frames = sorted((rec / "rgb").glob("*.jpg"))
        if not frames:
            raise ValueError(f"Missing RGB frames: {rec}")
        for frame_index in sorted(set([0, len(frames)//2, len(frames)-1])):
            path = frames[frame_index]
            with Image.open(path) as raw:
                original_size = raw.size
                image = raw.convert("RGB")
            image.thumbnail((1600, 800))
            w, h = image.size
            draw = ImageDraw.Draw(image)
            for polygon in config["polygons"]:
                vertices = [(x*w, y*h) for x, y in polygon]
                draw.line(vertices + vertices[:1], fill="red", width=3)
            # Candidate centers are anchors only; they are NOT masks or labels.
            ow, oh = original_size
            for x, y in [(38, 486), (ow-1-38, 486)]:
                px, py = x/ow*w, y/oh*h
                draw.ellipse((px-4, py-4, px+4, py+4), fill="cyan")
            target = out / f"{record_index}_{path.stem}.png"
            image.save(target)
            index.append({"source": str(path.resolve()), "original_size": original_size,
                          "preview": str(target), "candidate_centers_only": not bool(config["polygons"])})
    write_json(out / "preview_index.json", {"roi": config, "roi_sha256": sha256(args.roi),
                                            "frames": index})
    print(f"Saved {len(index)} previews to {out}. Review boundaries before confirmed=true.")


def split_records(records):
    records = sorted(set(records))
    if len(records) < 2:
        raise ValueError("Need at least two records for calibration/evaluation separation.")
    shuffled = np.random.default_rng(123).permutation(records)
    half = len(records)//2
    return {str(rec): "calibration" if i < half else "evaluation"
            for i, rec in enumerate(shuffled)}


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    from dataset.dataset import DG360LiteDataset, load_csv
    from infer import load_model

    roi = read_roi(args.roi)
    checkpoints = {}
    for item in args.checkpoints:
        name, sep, path = item.partition("=")
        if not sep or name in checkpoints or not name.replace("-", "").isalnum():
            raise ValueError("Use unique safe names: E0=path/best.pt E1=path/best.pt ...")
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        checkpoints[name] = Path(path)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Use a fresh output folder to preserve evidence: {out}")
    dataset = DG360LiteDataset(args.root, T=16, clip_step=8, frame_stride=1,
                              skip_first=30, img_size=(224, 1120), resize_mode="stretch",
                              augment=False, paper_protocol=True)
    record_names = [str(rec.relative_to(dataset.root)) for rec in dataset.data_paths]
    partitions = split_records(record_names)
    sizes = {}
    csv_hashes = {}
    for ri, rec in enumerate(dataset.data_paths):
        first = rec / "rgb" / f"{dataset.frame_ids[ri][0]:06d}.jpg"
        with Image.open(first) as image:
            sizes[ri] = image.size
        if roi.get("reference_image_size") and list(sizes[ri]) != roi["reference_image_size"]:
            raise ValueError(f"ROI geometry differs from reviewed source size: {rec}: {sizes[ri]}")
        csv_hashes[record_names[ri]] = sha256(rec / "sim_gaze_df.csv")
    rows, excluded = [], []
    for sample_index, (ri, start) in enumerate(dataset.samples):
        ids = dataset.frame_ids[ri][start:start+16]
        row = {"sample_index": sample_index, "record": record_names[ri],
               "sample_id": f"{record_names[ri]}:{ids[0]}-{ids[-1]}",
               "first_frame": ids[0], "last_frame": ids[-1], "input_frame_ids": ids,
               "partition": partitions[record_names[ri]]}
        label = gaze_label(load_csv(str(dataset.data_paths[ri] / "sim_gaze_df.csv")),
                           ids, sizes[ri], roi["polygons"])
        if label is None:
            excluded.append({**row, "reason": "not_all_16_gaze_points_valid"})
        else:
            rows.append({**row, "label": label})
    out.mkdir(parents=True)
    manifest = {"root": str(dataset.root.resolve()), "seed": 123,
                "roi": roi, "roi_sha256": sha256(args.roi), "csv_sha256": csv_hashes,
                "implementation_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
                    for p in [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "utils/mirror_roi.py"]},
                "record_partitions": partitions, "samples": rows, "excluded": excluded,
                "protocol": {"T": 16, "clip_step": 8, "stride": 1, "skip_first": 30,
                             "resize": "stretch", "img_size": [224, 1120],
                             "label_window": "input frames only; all 16 gaze points valid"},
                "limitation": "Exploratory validation reused for checkpoint selection; not independent test or unseen-driver evidence."}
    write_json(out / "manifest.json", manifest)
    for part in ("calibration", "evaluation"):
        labels = [r["label"] for r in rows if r["partition"] == part]
        if set(labels) != {0, 1}:
            write_json(out / "status.json", {"status": "not_feasible", "reason":
                       f"{part} lacks both classes", "labels": {"0": labels.count(0), "1": labels.count(1)}})
            print(f"Not feasible: {part} lacks both classes. ROI unchanged; no inference started.")
            return
    mask_np = roi_mask(roi["polygons"], 224, 1120)
    if not mask_np.any() or mask_np.all():
        raise ValueError("ROI raster must cover some, but not all, output pixels.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly choose --device cpu.")
    selected_indices = [r["sample_index"] for r in rows]
    summaries = {}
    for name, path in checkpoints.items():
        digest = sha256(path)
        model, checkpoint, experiment, saved, output_type = load_model(path, device)
        required = {"clip_step": 8, "frame_stride": 1, "skip_first": 30,
                    "img_size": [224, 1120], "resize_mode": "stretch", "paper_protocol": True}
        if experiment != "tsm_attn_pool" or output_type != "sigmoid" or any(
                saved.get(key) != value for key, value in required.items()):
            raise ValueError(f"{name}: checkpoint does not match the fixed saliency protocol.")
        model.eval()
        amp = device.type == "cuda" and bool(saved.get("use_amp", False))
        roi_tensor = torch.from_numpy(mask_np).to(device)
        loader = DataLoader(Subset(dataset, selected_indices), batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
        scores = []
        with torch.inference_mode():
            for batch in tqdm(loader, desc=name):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    prediction = model(batch["rgb"].to(device))
                prediction = prediction.float()[0, 0]
                if not torch.isfinite(prediction).all() or (prediction < 0).any() or prediction.sum() <= 0:
                    raise ValueError("Non-finite, negative or zero-mass saliency output.")
                scores.append(float((prediction[roi_tensor].sum()/prediction.sum()).cpu()))
        if sha256(path) != digest:
            raise RuntimeError("Checkpoint changed during evaluation; rerun with a fixed checkpoint copy.")
        calibration = np.array([r["partition"] == "calibration" for r in rows])
        labels, scores = np.array([r["label"] for r in rows]), np.array(scores)
        threshold = choose_threshold(labels[calibration], scores[calibration])
        report = ~calibration
        metrics = {**binary_metrics(labels[report], scores[report], threshold),
                   **pr_metrics(labels[report], scores[report])}
        baselines = {}
        for baseline, constant in (("always_outside", 0.), ("always_mirror", 1.)):
            constant_scores = np.full(int(report.sum()), constant)
            baselines[baseline] = {**binary_metrics(labels[report], constant_scores, .5),
                                   **pr_metrics(labels[report], constant_scores)}
        output_rows = []
        for row, score in zip(rows, scores):
            pred = int(score >= threshold)
            outcome = ("TP" if pred else "FN") if row["label"] else ("FP" if pred else "TN")
            output_rows.append({**row, "score": float(score), "prediction": pred, "outcome": outcome})
        with (out / f"{name}_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
        summaries[name] = {"checkpoint": str(path.resolve()), "checkpoint_sha256": digest,
                           "checkpoint_args": saved, "threshold": threshold,
                           "calibration_count": int(calibration.sum()), "evaluation_count": int(report.sum()),
                           "metrics": metrics, "baselines": baselines}
        examples = []
        for outcome in ("TP", "FP", "FN", "TN"):
            candidates = [r for r in output_rows if r["partition"] == "evaluation" and r["outcome"] == outcome]
            # Predeclared order: first two by dataset order, not best-looking maps.
            for row in candidates[:2]:
                save_example(out, name, row, dataset, model, device, amp, roi)
                examples.append(row)
        write_json(out / f"{name}_examples.json", examples)
        write_json(out / "metrics.json", summaries)
        del model, checkpoint
    lines = ["# Mirror ROI detection (exploratory)", "", manifest["limitation"], "",
             "AP is average precision with grouped score ties; PR-AUC uses trapezoidal integration.", "",
             "| Model | Threshold | Precision | Recall | F1 | AP | Positive prevalence |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        m = summary["metrics"]
        lines.append(f"| {name} | {summary['threshold']:.6f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['ap']:.4f} | {m['prevalence']:.4f} |")
        for baseline, bm in summary["baselines"].items():
            lines.append(f"| {name}/{baseline} | 0.5 | {bm['precision']:.4f} | {bm['recall']:.4f} | {bm['f1']:.4f} | {bm['ap']:.4f} | {bm['prevalence']:.4f} |")
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(out / "status.json", {"status": "complete", "models": list(summaries)})


def save_example(out, name, row, dataset, model, device, amp, roi):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    batch = dataset[row["sample_index"]]
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        prediction = model(batch["rgb"].unsqueeze(0).to(device))[0, 0].float().cpu().numpy()
    rgb = batch["rgb"][-1].numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb*np.array([.229, .224, .225])+np.array([.485, .456, .406]), 0, 1)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for ax, image, title in zip(axes, [rgb, batch["sal"][0].numpy(), prediction],
                               ["Final RGB frame", "Dataset saliency (not the binary gaze label)", "Predicted saliency"]):
        ax.imshow(image, cmap="magma" if image.ndim == 2 else None)
        for polygon in roi["polygons"]:
            points = np.asarray(polygon + polygon[:1]) * np.array([1120, 224])
            ax.plot(points[:, 0], points[:, 1], color="cyan")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{name} {row['outcome']} | label={row['label']} score={row['score']:.5f} | {row['sample_id']}")
    fig.tight_layout()
    fig.savefig(out / f"{name}_{row['outcome']}_{row['sample_index']}.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--root", required=True)
        p.add_argument("--roi", default="configs/mirror_roi.json")
        p.add_argument("--out", required=True)
        if command == "preview":
            p.add_argument("--records", type=int, default=6)
        else:
            p.add_argument("--checkpoints", nargs="+", required=True)
            p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
            p.add_argument("--num-workers", type=int, choices=[0, 1, 2], default=2)
    args = parser.parse_args()
    (preview if args.command == "preview" else evaluate)(args)


if __name__ == "__main__":
    main()
