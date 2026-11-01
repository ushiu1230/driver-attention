"""Inference for checkpoints written by train.py (baseline / temporal variants).

  python infer.py --image dataset/.../test/C017/012/002/rgb/000040.jpg
  python infer.py --checkpoint checkpoints/tsm/best.pt --image .../rgb/000040.jpg

Temporal checkpoints need T consecutive frames: the preceding frames are read
from the same recording. Inference saves both the sigmoid saliency map and a
sum-normalized probability map for distribution-based metrics.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2

from dataset.dataset import resize_with_pad
from models import DriverGazeNet
from models.model_factory import EXPERIMENTS

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/baseline/best.pt")
    p.add_argument("--image", required=True, help="RGB frame (the TARGET frame for temporal models)")
    p.add_argument("--out-dir", default="results/Inference")
    return p.parse_args()


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    experiment = ckpt.get("experiment", "baseline")
    saved = ckpt.get("args", {})
    output_type = ckpt.get("output", "sigmoid")
    if output_type == "sigmoid":
        from models import SingleFrameSaliencyNet
        cfg = EXPERIMENTS[experiment]
        if cfg["input_frames"] == 1 and not cfg["use_tsm"]:
            model = SingleFrameSaliencyNet(
                backbone_name=saved.get("backbone", "mobilenet_v3_large"),
                pretrained=False,
                decoder_width=saved.get("decoder_width", 128),
            )
        else:
            model = DriverGazeNet(
                backbone_name=saved.get("backbone", "mobilenet_v3_large"),
                num_segments=cfg["input_frames"],
                temporal_mode=cfg["temporal_mode"],
                use_tsm=cfg["use_tsm"],
                pretrained=False,
                decoder_width=saved.get("decoder_width", 128),
                apply_sigmoid=True,
            )
    else:
        if output_type != "logits":
            raise ValueError(f"Unsupported checkpoint output type: {output_type}")
        cfg = EXPERIMENTS[experiment]
        model = DriverGazeNet(
            backbone_name=saved.get("backbone", "mobilenet_v3_large"),
            num_segments=cfg["input_frames"],
            temporal_mode=cfg["temporal_mode"],
            use_tsm=cfg["use_tsm"],
            pretrained=False,
            decoder_width=saved.get("decoder_width", 128),
            apply_sigmoid=False,
        )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt, experiment, saved, output_type


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, experiment, saved, output_type = load_model(
        Path(args.checkpoint), device
    )
    cfg = EXPERIMENTS[experiment]
    T = cfg["input_frames"]
    saved_size = saved.get("img_size")
    if saved_size is None:
        saved_size = (saved.get("img_h", 224), saved.get("img_w", 1120))
    size = tuple(int(dimension) for dimension in saved_size)
    if len(size) != 2 or any(dimension < 1 for dimension in size):
        raise ValueError(f"Invalid image size in checkpoint: {saved_size}")
    stretch = saved.get("resize_mode", "pad") == "stretch"

    def fit(img):
        if stretch:
            return v2.functional.resize(img, size=list(size), antialias=True)
        return resize_with_pad(img, size)

    image_path = Path(args.image)
    number = int(image_path.stem)
    recording_dir = image_path.parent.parent
    metadata_path = recording_dir / "sim_gaze_df.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Recording metadata not found: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8-sig") as metadata_file:
        frame_ids = [int(row["Frame"]) for row in csv.DictReader(metadata_file)]
    if number not in frame_ids:
        raise ValueError(f"Target frame {number} is not listed in {metadata_path}")
    target_index = frame_ids.index(number)
    if target_index < T - 1:
        raise ValueError(
            f"Target frame {number} has only {target_index} preceding frames; "
            f"checkpoint requires {T - 1}."
        )
    clip_ids = frame_ids[target_index - T + 1:target_index + 1]
    frame_paths = [
        image_path.with_name(f"{frame_id:06d}{image_path.suffix}")
        for frame_id in clip_ids
    ]
    missing = [str(p) for p in frame_paths if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing frames for T={T}: {missing[:3]} ...")

    preprocess = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize(mean=MEAN, std=STD)])
    raw_last = read_image(str(frame_paths[-1]), mode=ImageReadMode.RGB)
    frames = [preprocess(fit(read_image(str(p), mode=ImageReadMode.RGB))) for p in frame_paths]
    clip = torch.stack(frames, 0).unsqueeze(0).to(device)  # [1, T, 3, H, W]

    valid = (fit(torch.ones(1, *raw_last.shape[-2:], dtype=torch.uint8)) > 0).flatten().to(device)

    with torch.inference_mode():
        output = model(clip)[0, 0].flatten().float()
    if output_type == "sigmoid":
        saliency = output.clamp(0, 1).masked_fill(~valid, 0.0)
    else:       # logits -> masked spatial softmax
        saliency = output.sigmoid().masked_fill(~valid, 0.0)
        density = torch.softmax(
            output.masked_fill(~valid, float("-inf")), dim=0
        )
    if output_type == "sigmoid":
        density = saliency
    density = density / density.sum().clamp_min(1e-8)
    saliency = saliency.view(*size).cpu().numpy()
    prob = density.view(*size).cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.checkpoint).stem}_{image_path.stem}"
    np.save(out_dir / f"{stem}_saliency.npy", saliency.astype(np.float32))
    np.save(out_dir / f"{stem}_prob.npy", prob.astype(np.float32))

    vis = saliency / (saliency.max() + 1e-8)  # per-image scaling for display only
    heat = (plt.get_cmap("jet")(vis)[..., :3] * 255).astype(np.uint8)
    heat[~valid.view(*size).cpu().numpy()] = 0
    Image.fromarray((vis * 255).astype(np.uint8)).save(out_dir / f"{stem}_gray.png")
    rgb = (frames[-1] * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1))
    rgb = (rgb.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    overlay = (0.6 * rgb + 0.4 * heat).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")

    py, px = np.unravel_index(prob.argmax(), prob.shape)
    print(f"Checkpoint: {args.checkpoint} | experiment={experiment} | epoch={ckpt.get('epoch')} "
          f"| output={output_type}")
    metrics = ckpt.get("val_metrics")
    if metrics:
        print("Validation:", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                       if isinstance(v, float)))
    elif ckpt.get("val_loss") is not None:
        print(f"Validation loss: {ckpt['val_loss']:.4f}")
    print(f"Probability map {prob.shape}, sums to {prob.sum():.4f}; peak at (x={px}, y={py})")
    print(f"Saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()