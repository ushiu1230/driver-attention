import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from ..dataset.dataset import resize_with_pad


DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "drivergaze360_subset" / "dataset" / "train"
PREVIEW_SIZE = (224, 1120)  # (H, W), matching DG360LiteDataset's fit-and-pad transform.
RECORDING_COUNT = 3
SAMPLE_COUNT = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview sampled RGB/saliency pairs across recordings."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--recordings",
        type=int,
        default=RECORDING_COUNT,
        help="Random recordings to include in the contact sheet.",
    )
    parser.add_argument(
        "--samples-per-recording",
        type=int,
        default=SAMPLE_COUNT,
        help="Random frames to sample from each recording, spread over its duration.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        help="Save the contact sheet to this path instead of opening a window.",
    )
    return parser.parse_args()


def image_files(folder):
    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }


def find_recordings(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    recordings = []
    for csv_path in root.rglob("sim_gaze_df.csv"):
        recording = csv_path.parent
        if (recording / "rgb").is_dir() and (recording / "saliency").is_dir():
            recordings.append(recording)
    recordings.sort()
    if not recordings:
        raise FileNotFoundError(
            f"No recordings containing rgb/, saliency/, and sim_gaze_df.csv found under {root}"
        )
    return recordings


def sample_frame_ids(rgb_files, saliency_files, count, rng, recording):
    rgb_ids = set(rgb_files)
    saliency_ids = set(saliency_files)
    if rgb_ids != saliency_ids:
        missing_saliency = sorted(rgb_ids - saliency_ids)
        missing_rgb = sorted(saliency_ids - rgb_ids)
        raise ValueError(
            f"RGB/saliency filenames differ in {recording}: "
            f"missing saliency={missing_saliency[:5]}, missing RGB={missing_rgb[:5]}"
        )
    frame_ids = sorted(rgb_ids, key=lambda frame_id: int(frame_id))
    if not frame_ids:
        raise FileNotFoundError(f"No JPG frames found in {recording / 'rgb'}")

    # Pick randomly within evenly spaced temporal bins to cover scene changes.
    bins = np.array_split(np.asarray(frame_ids, dtype=object), min(count, len(frame_ids)))
    return [str(rng.choice(frame_bin.tolist())) for frame_bin in bins]


def load_pair(rgb_path, saliency_path):
    with Image.open(rgb_path) as image:
        rgb = image.convert("RGB")
    with Image.open(saliency_path) as image:
        saliency = image.convert("L")

    if rgb.size != saliency.size:
        raise ValueError(
            "RGB/saliency source dimensions differ; refusing to hide a possible "
            f"alignment issue:\n  RGB: {rgb_path} ({rgb.size})\n"
            f"  Saliency: {saliency_path} ({saliency.size})"
        )

    source_range = (int(np.asarray(saliency).min()), int(np.asarray(saliency).max()))
    rgb = np.asarray(resize_with_pad(rgb, PREVIEW_SIZE))
    saliency = np.asarray(resize_with_pad(saliency, PREVIEW_SIZE))
    saliency_visual = np.empty_like(saliency)
    cv2.normalize(
        saliency,
        saliency_visual,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    heatmap = cv2.cvtColor(
        cv2.applyColorMap(saliency_visual, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    overlay = cv2.addWeighted(rgb, 0.6, heatmap, 0.4, 0)
    return rgb, saliency_visual, overlay, source_range


def main():
    args = parse_args()
    if args.recordings < 1:
        raise ValueError("--recordings must be at least 1")
    if args.samples_per_recording < 1:
        raise ValueError("--samples-per-recording must be at least 1")

    recordings = find_recordings(args.root)
    rng = random.Random(args.seed)
    recordings = rng.sample(recordings, k=min(args.recordings, len(recordings)))
    previews = []

    for recording in recordings:
        rgb_files = image_files(recording / "rgb")
        saliency_files = image_files(recording / "saliency")
        frame_ids = sample_frame_ids(
            rgb_files, saliency_files, args.samples_per_recording, rng, recording
        )
        print(f"{recording.relative_to(args.root)}: sampled frames {', '.join(frame_ids)}")
        for frame_id in frame_ids:
            rgb, saliency, overlay, saliency_range = load_pair(
                rgb_files[frame_id], saliency_files[frame_id]
            )
            previews.append((
                recording.relative_to(args.root),
                frame_id,
                rgb,
                saliency,
                overlay,
                saliency_range,
            ))

    fig, axes = plt.subplots(
        len(previews), 3, figsize=(15, max(4, 3.2 * len(previews))), squeeze=False
    )
    for row, (
        recording,
        frame_id,
        rgb,
        saliency,
        overlay,
        saliency_range,
    ) in enumerate(previews):
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{recording} | frame {frame_id} | RGB")
        axes[row, 1].imshow(saliency, cmap="hot", vmin=0, vmax=255)
        axes[row, 1].set_title(f"Saliency (source range {saliency_range})")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        for axis in axes[row]:
            axis.axis("off")

    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=120, bbox_inches="tight")
        print(f"Saved contact sheet to {args.output}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
