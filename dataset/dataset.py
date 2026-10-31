from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.transforms import v2


def resize_with_pad(image, size):
    """Resize without distortion and center-pad to (height, width)."""
    target_height, target_width = size
    height, width = v2.functional.get_size(image)
    scale = min(target_height / height, target_width / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))

    image = v2.functional.resize(
        image,
        size=[resized_height, resized_width],
        antialias=True,
    )
    vertical_padding = target_height - resized_height
    horizontal_padding = target_width - resized_width
    padding = [
        horizontal_padding // 2,
        vertical_padding // 2,
        horizontal_padding - horizontal_padding // 2,
        vertical_padding - vertical_padding // 2,
    ]
    return v2.functional.pad(image, padding, fill=0)


@lru_cache(maxsize=128)
def load_csv(csv_path: str) -> pd.DataFrame:
    """Cache metadata by CSV path (rather than by recording directory)."""
    return pd.read_csv(csv_path)


class DG360LiteDataset(Dataset):
    """Load RGB clips for current-frame saliency prediction.

    Each sample uses the ``T`` RGB frames ending at frame ``t`` as input and
    the dataset-provided saliency map for frame ``t`` as its target. Both are
    resized with preserved aspect ratio to fit ``img_size``, then center-padded
    with zeros to that fixed canvas size.

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
        clip_step=8,
        img_size=(224, 1120),
        skip_first=30,
    ):
        self.root = Path(root)
        self.T = int(T)
        self.stride = int(frame_stride)
        self.clip_step = int(clip_step)
        self.img_size = tuple(img_size)  # (H, W)
        self.skip_first = int(skip_first)

        if self.T < 1 or self.stride < 1 or self.clip_step < 1:
            raise ValueError("T, frame_stride, and clip_step must be positive integers")
        if len(self.img_size) != 2 or any(size < 1 for size in self.img_size):
            raise ValueError("img_size must contain positive (height, width) values")
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.data_paths = self._make_paths()
        if not self.data_paths:
            raise RuntimeError(
                f"No recordings found under {self.root}. Expected folders containing "
                "rgb/, saliency/, and sim_gaze_df.csv."
            )

        self.frame_ids = self._check_files()
        self.data_len = [len(frame_ids) for frame_ids in self.frame_ids]
        self.samples = self._make_samples()
        if not self.samples:
            raise RuntimeError(
                "No valid clips were created. Check recording lengths, T, skip_first, and stride."
            )

        self.rgb_tf = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.gt_tf = v2.Compose([
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

    def _check_files(self):
        frame_ids_by_recording = []
        for rec_dir in self.data_paths:
            csv_path = rec_dir / "sim_gaze_df.csv"
            meta = load_csv(str(csv_path))
            if "Frame" not in meta.columns:
                raise ValueError(f"CSV has no 'Frame' column: {csv_path}")

            rgb_stems = self._jpg_stems(rec_dir / "rgb")
            sal_stems = self._jpg_stems(rec_dir / "saliency")
            n_csv = len(meta)
            try:
                raw_frame_ids = meta["Frame"].tolist()
                frame_ids = [int(frame_id) for frame_id in raw_frame_ids]
                if any(
                    float(raw_id) != frame_id
                    for raw_id, frame_id in zip(raw_frame_ids, frame_ids)
                ):
                    raise ValueError("Frame IDs must be integers")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid frame ID in {csv_path}") from error

            if len(set(frame_ids)) != n_csv:
                raise ValueError(f"Duplicate frame IDs in {csv_path}")
            if any(current <= previous for previous, current in zip(frame_ids, frame_ids[1:])):
                raise ValueError(f"Frame IDs are not strictly increasing in {csv_path}")

            expected_stems = [f"{frame_id:06d}" for frame_id in frame_ids]
            if rgb_stems != sal_stems:
                raise ValueError(f"RGB/saliency filenames differ in {rec_dir}")
            if rgb_stems != expected_stems:
                raise ValueError(
                    f"Frame filenames do not align with CSV Frame IDs in {rec_dir}. "
                    f"RGB={len(rgb_stems)}, saliency={len(sal_stems)}, CSV={n_csv}. "
                    "Check that each CSV Frame ID has matching RGB and saliency files."
                )

            frame_ids_by_recording.append(frame_ids)
        return frame_ids_by_recording

    def _make_samples(self):
        samples = []
        for recording_idx, n_frames in enumerate(self.data_len):
            # Largest valid start index, inclusive.
            last_start = n_frames - 1 - (self.T - 1) * self.stride
            for start in range(self.skip_first, last_start + 1, self.clip_step):
                samples.append((recording_idx, start))
        return samples

    def __len__(self):
        return len(self.samples)

    def _read(self, recording_idx, frame_id, kind):
        image_path = (
            self.data_paths[recording_idx]
            / kind
            / f"{frame_id:06d}.jpg"
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        mode = ImageReadMode.RGB if kind == "rgb" else ImageReadMode.GRAY
        try:
            image = decode_jpeg(read_file(str(image_path)), mode=mode)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Failed to read or decode {kind} image: {image_path}"
            ) from error
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected one decoded image tensor from {image_path}")
        return image

    def __getitem__(self, index):
        recording_idx, start = self.samples[index]
        frame_positions = [
            start + i * self.stride for i in range(self.T)
        ]
        frame_ids = [
            self.frame_ids[recording_idx][position] for position in frame_positions
        ]

        rgb_frames = []
        target_rgb_size = None
        for frame_id in frame_ids:
            frame = self._read(recording_idx, frame_id, "rgb")
            target_rgb_size = frame.shape[-2:]
            frame = resize_with_pad(frame, self.img_size)
            rgb_frames.append(self.rgb_tf(frame))
        rgb_clip = torch.stack(rgb_frames, dim=0)  # [T, 3, H, W]

        # Baseline target: saliency map for the final frame in this clip.
        target_id = frame_ids[-1]
        saliency = self._read(recording_idx, target_id, "saliency")
        if target_rgb_size is None:
            raise RuntimeError("A clip must contain at least one RGB frame")
        if tuple(saliency.shape[-2:]) != tuple(target_rgb_size):
            raise ValueError(
                "RGB/saliency source dimensions differ for frame "
                f"{target_id:06d} in {self.data_paths[recording_idx]}: "
                f"RGB={tuple(target_rgb_size)}, "
                f"saliency={tuple(saliency.shape[-2:])}"
            )
        saliency = resize_with_pad(saliency, self.img_size)
        saliency = self.gt_tf(saliency)  # [1, H, W], float in [0, 1]

        return {"rgb": rgb_clip, "sal": saliency}
