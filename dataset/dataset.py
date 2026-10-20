import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.transforms import v2


@lru_cache(maxsize=128)
def load_csv(csv_path: str) -> pd.DataFrame:
    """Cache metadata by CSV path (rather than by recording directory)."""
    return pd.read_csv(csv_path)


class DG360LiteDataset(Dataset):
    """Load 16-frame RGB clips and the last frame's saliency target.

    Expected recording layout:
      root/train/C001/001/001-1/rgb/000001.jpg
      root/train/C001/001/001-1/saliency/000001.jpg
      root/train/C001/001/001-1/sim_gaze_df.csv

    Pass root=.../train for training data and root=.../val for validation data.
    """

    def __init__(
        self,
        root,
        T=16,
        frame_stride=1,
        overlap=8,
        img_size=(224, 1120),
        skip_first=30,
        check=True,
    ):
        self.root = Path(root)
        self.T = int(T)
        self.stride = int(frame_stride)
        self.overlap = int(overlap)  # step between clip start indices
        self.img_size = tuple(img_size)  # (H, W)
        self.skip_first = int(skip_first)

        if self.T < 1 or self.stride < 1 or self.overlap < 1:
            raise ValueError("T, frame_stride, and overlap must be positive integers")
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.data_paths = self._make_paths()
        if not self.data_paths:
            raise RuntimeError(
                f"No recordings found under {self.root}. Expected folders containing "
                "rgb/, saliency/, and sim_gaze_df.csv."
            )

        self.data_len = self._check_files(check)
        self.samples = self._make_samples()
        if not self.samples:
            raise RuntimeError(
                "No valid clips were created. Check recording lengths, T, skip_first, and stride."
            )

        self.rgb_tf = v2.Compose([
            v2.Resize(size=self.img_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.gt_tf = v2.Compose([
            v2.Resize(size=self.img_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def _make_paths(self):
        # Discover recording directories robustly, regardless of nesting depth.
        paths = []
        for csv_path in self.root.rglob("sim_gaze_df.csv"):
            rec_dir = csv_path.parent
            if (rec_dir / "rgb").is_dir() and (rec_dir / "saliency").is_dir():
                paths.append(rec_dir)
        return sorted(set(paths))

    @staticmethod
    def _jpg_stems(folder):
        return sorted(
            p.stem for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
        )

    def _check_files(self, check):
        lengths = []
        for rec_dir in self.data_paths:
            csv_path = rec_dir / "sim_gaze_df.csv"
            meta = load_csv(str(csv_path))
            if "Frame" not in meta.columns:
                raise ValueError(f"CSV has no 'Frame' column: {csv_path}")

            rgb_stems = self._jpg_stems(rec_dir / "rgb")
            sal_stems = self._jpg_stems(rec_dir / "saliency")
            n_csv = len(meta)

            if check:
                expected_stems = [f"{int(i):06d}" for i in meta["Frame"].tolist()]
                if rgb_stems != sal_stems:
                    raise ValueError(f"RGB/saliency filenames differ in {rec_dir}")
                if rgb_stems != expected_stems:
                    raise ValueError(
                        f"Frame filenames do not align with CSV Frame column in {rec_dir}. "
                        f"RGB={len(rgb_stems)}, CSV={n_csv}. Check first/last indices."
                    )
            else:
                if len(rgb_stems) != len(sal_stems):
                    raise ValueError(f"RGB/saliency frame counts differ in {rec_dir}")
                if len(rgb_stems) != n_csv:
                    raise ValueError(
                        f"RGB/CSV frame counts differ in {rec_dir}: "
                        f"RGB={len(rgb_stems)}, CSV={n_csv}"
                    )

            lengths.append(n_csv)
        return lengths

    def _make_samples(self):
        samples = []
        for recording_idx, n_frames in enumerate(self.data_len):
            # Largest valid start index, inclusive.
            last_start = n_frames - 1 - (self.T - 1) * self.stride
            for start in range(self.skip_first, last_start + 1, self.overlap):
                samples.append((recording_idx, start))
        return samples

    def __len__(self):
        return len(self.samples)

    def _read(self, recording_idx, frame_idx, kind):
        # frame_idx is zero-based; extracted filenames start at 000001.jpg.
        image_path = (
            self.data_paths[recording_idx]
            / kind
            / f"{frame_idx + 1:06d}.jpg"
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        mode = ImageReadMode.RGB if kind == "rgb" else ImageReadMode.GRAY
        image = decode_jpeg(read_file(str(image_path)), mode=mode)
        return image

    def __getitem__(self, index):
        recording_idx, start = self.samples[index]
        frame_indices = [
            start + i * self.stride for i in range(self.T)
        ]

        rgb_frames = []
        for frame_idx in frame_indices:
            frame = self._read(recording_idx, frame_idx, "rgb")
            rgb_frames.append(self.rgb_tf(frame))
        rgb_clip = torch.stack(rgb_frames, dim=0)  # [T, 3, H, W]

        # Baseline target: saliency map for the final frame in this clip.
        target_idx = frame_indices[-1]
        saliency = self._read(recording_idx, target_idx, "saliency")
        saliency = self.gt_tf(saliency)  # [1, H, W], float in [0, 1]

        return {"rgb": rgb_clip, "sal": saliency}
