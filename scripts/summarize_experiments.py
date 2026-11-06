"""Summarize trained E0/E1/E2 checkpoints; no training or evaluation is started."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.mirror_roi import sha256


COMMON = dict(experiment="tsm_attn_pool", batch_size=1, accumulation_steps=5,
              lr=1e-6, weight_decay=.01, num_workers=2, clip_step=8, frame_stride=1,
              skip_first=30, img_size=[224, 1120], resize_mode="stretch", seed=123,
              dropout=0., augment=False, weighted_samples=False, paper_protocol=True,
              loss_profile="drivergaze360", w_kld=1., w_cc=1., w_mse=0., w_nss=0.,
              num_epochs=10, early_stopping_patience=10, use_amp=True)


def summarize(paths):
    import torch
    rows, init_paths, data_paths, init_hashes = [], set(), set(), set()
    for name, path in paths.items():
        if not path.is_file():
            rows.append({"experiment": name, "status": "incomplete", "checkpoint": str(path)})
            continue
        before = sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        args = checkpoint["args"]
        expected = {**COMMON, "freeze_backbone_stages": 12 if name == "E1" else 0,
                    "backbone_lr_scale": .1 if name == "E2" else 1., "freeze_backbone": False}
        mismatches = {key: {"expected": value, "actual": args.get(key)}
                      for key, value in expected.items() if args.get(key) != value}
        if checkpoint.get("output") != "sigmoid":
            mismatches["output"] = checkpoint.get("output")
        if args.get("eval_only", False) or int(checkpoint.get("optimizer_updates", 0)) <= 0:
            mismatches["training_progress"] = "Expected trained checkpoint with positive optimizer updates."
        if not torch.isfinite(torch.tensor(checkpoint["val_loss"])):
            raise ValueError(f"Non-finite loss in {path}")
        if before != sha256(path):
            raise RuntimeError(f"Checkpoint changed while reading {path}; retry after training.")
        init_paths.add(str(Path(args["init_weights"]).resolve()))
        data_paths.add((str(Path(args["train_path"]).resolve()), str(Path(args["val_path"]).resolve())))
        row = {"experiment": name, "status": "valid" if not mismatches else "protocol_mismatch",
               "checkpoint": str(path.resolve()), "sha256": before, "mismatches": mismatches,
               "epoch": int(checkpoint["epoch"])+1, "val_loss": checkpoint["val_loss"],
               "metrics": checkpoint["val_metrics"], "optimizer_updates": checkpoint.get("optimizer_updates"),
               "args": args}
        receipt_path = path.parent / "run_manifest.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            row["launch_receipt"] = receipt
            init_hashes.add(receipt["initialization_sha256"])
            if receipt.get("finished"):
                row["foreground_wall_seconds"] = (datetime.fromisoformat(receipt["finished"]) -
                                                   datetime.fromisoformat(receipt["started"])).total_seconds()
            if receipt.get("initialization_unchanged") is False:
                raise ValueError(f"Initialization changed during {name}")
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_files = list(path.parent.glob("events.out.tfevents.*"))
        observed = []
        for event_file in event_files:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            observed.extend(event.wall_time for tag in accumulator.Tags().get("scalars", [])
                            for event in accumulator.Scalars(tag))
        row["observed_log_span_seconds"] = max(observed)-min(observed) if observed else None
        row["time_note"] = "Observed scalar log span, not exact GPU runtime; may include later epochs/restarts."
        rows.append(row)
        del checkpoint
    if len(init_paths) > 1 or len(data_paths) > 1 or len(init_hashes) > 1:
        raise ValueError("Runs have different initialization paths or dataset roots.")
    baseline = next((r for r in rows if r["experiment"] == "E0" and r["status"] == "valid"), None)
    for row in rows:
        if baseline and row["status"] == "valid":
            row["delta_vs_E0"] = row["val_loss"] - baseline["val_loss"]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e0", default="checkpoints/E0-full-finetune/E0-full-finetune/best.pt")
    p.add_argument("--e1", default="checkpoints/E1-partial-freeze/best.pt")
    p.add_argument("--e2", default="checkpoints/E2-discriminative-lr/best.pt")
    p.add_argument("--out", default="results/unified_protocol")
    args = p.parse_args()
    rows = summarize({name: Path(getattr(args, name.lower())) for name in ("E0", "E1", "E2")})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"runs": rows, "provenance_limit": "Same recorded paths do not prove historical image identity or initialization hash. Active E0 predates the run receipt tool.",
               "limitation": "Validation used for model selection, with shared drivers; no independent test/unseen-driver claim."}
    (out / "E0_E1_E2_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Trained E0–E2", "", payload["limitation"], "", payload["provenance_limit"], "",
             "| Run | Status | Best epoch | Loss | Delta vs E0 | KLD | CC | SIM | NSS | Updates | Log span (h) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["status"] != "valid":
            lines.append(f"| {row['experiment']} | {row['status']} | — | — | — | — | — | — | — | — | — |")
            continue
        m = row["metrics"]
        span = row["observed_log_span_seconds"]
        lines.append(f"| {row['experiment']} | valid | {row['epoch']} | {row['val_loss']:.6f} | {row.get('delta_vs_E0', float('nan')):.6f} | {m['KLD']:.4f} | {m['CC']:.4f} | {m['SIM']:.4f} | {m['NSS']:.4f} | {row['optimizer_updates']} | {span/3600 if span is not None else 'unknown'} |")
    (out / "E0_E1_E2_results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
