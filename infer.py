"""Inference for checkpoints written by train.py (baseline / temporal variants).

  python infer.py --image dataset/.../test/C017/012/002/rgb/000040.jpg
  python infer.py --checkpoint checkpoints/tsm/best.pt --image .../rgb/000040.jpg

Temporal checkpoints need T consecutive frames: the preceding frames are read
from the same recording when ``--image`` is used. Alternatively, ``--clip``
accepts a video file or a directory of RGB frames and uses its final T frames.
Inference saves a saliency map and its normalized probability density.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2

from dataset.dataset import make_fixation_mask, resize_image
from models import DriverGazeNet
from models.model_factory import EXPERIMENTS
from utils.metrics import SaliencyMetrics

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/baseline/best.pt")
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        help="Target RGB frame; earlier frames are loaded from its recording",
    )
    input_group.add_argument(
        "--clip",
        help="RGB video file or directory of RGB frames; uses its final T frames",
    )
    p.add_argument(
        "--ground-truth",
        help="Optional matching ground-truth saliency image/video; auto-detected "
        "for DriverGaze360 recordings",
    )
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
                dropout=saved.get("dropout", 0.1),
                apply_sigmoid=output_type == "sigmoid",
            )
        else:
            model = DriverGazeNet(
                backbone_name=saved.get("backbone", "mobilenet_v3_large"),
                num_segments=cfg["input_frames"],
                temporal_mode=cfg["temporal_mode"],
                use_tsm=cfg["use_tsm"],
                pretrained=False,
                decoder_width=saved.get("decoder_width", 128),
                dropout=saved.get("dropout", 0.1),
                apply_sigmoid=True,
            )
    else:
        if output_type not in ("logits", "spatial_softmax"):
            raise ValueError(f"Unsupported checkpoint output type: {output_type}")
        cfg = EXPERIMENTS[experiment]
        model = DriverGazeNet(
            backbone_name=saved.get("backbone", "mobilenet_v3_large"),
            num_segments=cfg["input_frames"],
            temporal_mode=cfg["temporal_mode"],
            use_tsm=cfg["use_tsm"],
            pretrained=False,
            decoder_width=saved.get("decoder_width", 128),
            dropout=saved.get("dropout", 0.1),
            apply_sigmoid=False,
        )
    model.load_checkpoint_weights(ckpt["model"])
    return model.to(device).eval(), ckpt, experiment, saved, output_type


def read_rgb_clip(source, num_frames):
    if source.is_dir():
        image_paths = [
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        image_paths.sort(
            key=lambda path: (
                0 if path.stem.isdecimal() else 1,
                int(path.stem) if path.stem.isdecimal() else path.name.casefold(),
            )
        )
        if len(image_paths) < num_frames:
            raise ValueError(
                f"Clip directory has {len(image_paths)} RGB frames; "
                f"checkpoint requires {num_frames}."
            )
        image_paths = image_paths[-num_frames:]
        return [
            read_image(str(path), mode=ImageReadMode.RGB) for path in image_paths
        ], image_paths[-1].stem

    if not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open RGB video: {source}")

    frames = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
            if len(frames) > num_frames:
                frames.pop(0)
    finally:
        capture.release()

    if len(frames) < num_frames:
        raise ValueError(
            f"Video contains {len(frames)} readable frames; "
            f"checkpoint requires {num_frames}."
        )
    return frames, source.stem


def read_video_last_frame(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open ground-truth video: {path}")

    last_frame = None
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()

    if last_frame is None:
        raise ValueError(f"Ground-truth video contains no readable frames: {path}")
    return torch.from_numpy(last_frame.copy()).permute(2, 0, 1)


def find_ground_truth(args, input_path, input_stem):
    if args.ground_truth:
        path = Path(args.ground_truth)
        if not path.is_file():
            raise FileNotFoundError(f"Ground-truth file not found: {path}")
        return path

    if args.image:
        image_path = Path(args.image)
        candidates = [
            image_path.parent.parent / "saliency" / f"{image_path.stem}{suffix}"
            for suffix in (image_path.suffix, ".jpg", ".jpeg", ".png")
        ]
        return next((path for path in candidates if path.is_file()), None)

    if input_path.is_dir():
        if input_path.name.casefold() == "rgb":
            candidates = [
                input_path.parent / "saliency" / f"{input_stem}{suffix}"
                for suffix in (".jpg", ".jpeg", ".png")
            ]
            return next((path for path in candidates if path.is_file()), None)
        return None

    if input_path.stem.casefold() in {"rgb", "video"}:
        candidate = input_path.with_name(f"saliency{input_path.suffix}")
        return candidate if candidate.is_file() else None
    return None


def load_ground_truth(path):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        if not path.is_file():
            raise FileNotFoundError(f"Ground-truth image not found: {path}")
        return read_image(str(path), mode=ImageReadMode.RGB)
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth video not found: {path}")
    return read_video_last_frame(path)


def load_input_fixation(args, input_path, input_stem, raw_image_size, size, resize_mode):
    if args.image:
        image_path = Path(args.image)
        metadata_path = image_path.parent.parent / "sim_gaze_df.csv"
        target_frame_id = int(image_path.stem)
    elif input_path.is_dir() and input_path.name.casefold() == "rgb":
        metadata_path = input_path.parent / "sim_gaze_df.csv"
        try:
            target_frame_id = int(input_stem)
        except ValueError as error:
            raise ValueError(
                "DriverGaze360 RGB frame filenames must use numeric frame IDs"
            ) from error
    elif not input_path.is_dir() and input_path.stem.casefold() == "rgb":
        metadata_path = input_path.parent / "sim_gaze_df.csv"
        target_frame_id = None
    else:
        return None

    if not metadata_path.is_file():
        return None
    metadata = pd.read_csv(metadata_path)
    if "Frame" not in metadata.columns:
        raise ValueError(f"CSV has no 'Frame' column: {metadata_path}")
    frame_ids = pd.to_numeric(metadata["Frame"], errors="coerce")
    if frame_ids.isna().any() or (frame_ids % 1 != 0).any():
        raise ValueError(f"Invalid frame ID in {metadata_path}")
    if target_frame_id is None:
        if metadata.empty:
            raise ValueError(f"No frame annotations found in {metadata_path}")
        target_frame_id = int(frame_ids.iloc[-1])

    target_gaze = metadata.loc[frame_ids == target_frame_id]
    if target_gaze.empty:
        raise ValueError(
            f"Target frame {target_frame_id} is not listed in {metadata_path}"
        )
    return make_fixation_mask(
        target_gaze,
        src_img_size=raw_image_size,
        target_size=size,
        resize_mode=resize_mode,
    )


def save_comparison(out_dir, stem, rgb, prediction, target, metrics):
    rgb_image = np.asarray(rgb, dtype=np.uint8)
    pred_map = prediction.squeeze().cpu().numpy()
    target_map = target.squeeze().cpu().numpy()
    pred_display = pred_map / (pred_map.max() + 1e-8)
    target_display = target_map / (target_map.max() + 1e-8)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB input")
    axes[1].imshow(target_display, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred_display, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")

    metrics_text = " | ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
    figure.suptitle(metrics_text)
    figure.tight_layout()
    figure.savefig(out_dir / f"{stem}_comparison.png", dpi=150)
    plt.close(figure)


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
    resize_mode = saved.get("resize_mode", "pad")

    def fit(img):
        return resize_image(img, size, resize_mode)

    preprocess = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize(mean=MEAN, std=STD)])
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Target image not found: {image_path}")
        try:
            number = int(image_path.stem)
        except ValueError as error:
            raise ValueError(
                f"Target image filename must have a numeric frame ID: {image_path}"
            ) from error
        recording_dir = image_path.parent.parent
        metadata_path = recording_dir / "sim_gaze_df.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Recording metadata not found: {metadata_path}")
        with metadata_path.open(newline="", encoding="utf-8-sig") as metadata_file:
            raw_frame_ids = [row["Frame"] for row in csv.DictReader(metadata_file)]
        try:
            frame_ids = [int(float(frame_id)) for frame_id in raw_frame_ids]
            if any(
                float(raw_id) != frame_id
                for raw_id, frame_id in zip(raw_frame_ids, frame_ids)
            ):
                raise ValueError("Frame IDs must be integers")
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid frame ID in {metadata_path}") from error
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
        missing = [str(path) for path in frame_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing frames for T={T}: {missing[:3]}"
            )
        raw_frames = [
            read_image(str(path), mode=ImageReadMode.RGB) for path in frame_paths
        ]
        input_stem = image_path.stem
        input_path = image_path
    else:
        input_path = Path(args.clip)
        raw_frames, input_stem = read_rgb_clip(input_path, T)

    raw_last = raw_frames[-1]
    frames = [preprocess(fit(frame)) for frame in raw_frames]
    clip = torch.stack(frames, 0).unsqueeze(0).to(device)  # [1, T, 3, H, W]

    valid = (fit(torch.ones(1, *raw_last.shape[-2:], dtype=torch.uint8)) > 0).flatten().to(device)

    with torch.inference_mode():
        output = model(clip)[0, 0].flatten().float()
    if output_type == "sigmoid":
        saliency = output.clamp(0, 1).masked_fill(~valid, 0.0)
    elif output_type == "spatial_softmax":
        saliency = torch.softmax(
            output.masked_fill(~valid, float("-inf")), dim=0
        ).masked_fill(~valid, 0.0)
    else:       # logits -> masked spatial softmax
        saliency = output.sigmoid().masked_fill(~valid, 0.0)
        density = torch.softmax(
            output.masked_fill(~valid, float("-inf")), dim=0
        )
    if output_type in ("sigmoid", "spatial_softmax"):
        density = saliency
    density = density / density.sum().clamp_min(1e-8)
    saliency = saliency.view(*size).cpu().numpy()
    prob = density.view(*size).cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.checkpoint).stem}_{input_stem}"
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

    print(f"Checkpoint: {args.checkpoint} | experiment={experiment} | epoch={ckpt.get('epoch')} "
          f"| output={output_type}")
    metrics = ckpt.get("val_metrics")
    if metrics:
        print("Validation:", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                       if isinstance(v, float)))
    elif ckpt.get("val_loss") is not None:
        print(f"Validation loss: {ckpt['val_loss']:.4f}")
    print(f"Probability map {prob.shape}, sums to {prob.sum():.4f}")

    ground_truth_path = find_ground_truth(args, input_path, input_stem)
    if ground_truth_path is None:
        print(
            "Ground truth not found automatically. Pass --ground-truth "
            "<saliency-image-or-video> to compare."
        )
    else:
        ground_truth = load_ground_truth(ground_truth_path)
        if tuple(ground_truth.shape[-2:]) != tuple(raw_last.shape[-2:]):
            raise ValueError(
                "Ground-truth and RGB frame dimensions differ: "
                f"GT={tuple(ground_truth.shape[-2:])}, "
                f"RGB={tuple(raw_last.shape[-2:])}"
            )
        ground_truth = v2.ToDtype(torch.float32, scale=True)(fit(ground_truth))
        ground_truth = ground_truth.mean(dim=0, keepdim=True)
        ground_truth = ground_truth.masked_fill(~valid.view(1, *size).cpu(), 0)
        prediction_tensor = torch.from_numpy(saliency).unsqueeze(0).unsqueeze(0)
        target_tensor = ground_truth.unsqueeze(0)
        valid_mask = valid.view(1, 1, *size).cpu()
        if target_tensor.sum() <= 0:
            raise ValueError(f"Ground-truth saliency map is empty: {ground_truth_path}")
        fixation = load_input_fixation(
            args,
            input_path,
            input_stem,
            raw_last.shape[-2:],
            size,
            resize_mode,
        )
        metric_values = SaliencyMetrics()(
            prediction_tensor,
            target_tensor,
            valid_mask,
            fixation=fixation.unsqueeze(0) if fixation is not None else None,
        )
        metrics = {
            name: values.item() for name, values in metric_values.items()
        }
        np.save(
            out_dir / f"{stem}_ground_truth.npy",
            ground_truth.squeeze(0).numpy().astype(np.float32),
        )
        save_comparison(
            out_dir,
            stem,
            rgb,
            prediction_tensor[0],
            target_tensor[0],
            metrics,
        )
        print(
            f"Compared with ground truth: {ground_truth_path} | "
            + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        )
        if fixation is None:
            print("NSS unavailable: matching gaze annotations were not found.")
    print(f"Saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()