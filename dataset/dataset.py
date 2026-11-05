import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
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


def resize_image(image, size, mode="stretch"):
    """Resize to (height, width), either directly or with aspect-preserving padding."""
    if mode == "stretch":
        return v2.functional.resize(image, size=list(size), antialias=True)
    if mode == "pad":
        return resize_with_pad(image, size)
    raise ValueError(f"Unsupported resize mode: {mode}")


def make_fixation_mask(gaze, src_img_size, target_size, resize_mode="stretch"):
    """Map gaze coordinates into the resized image coordinate system."""
    if not {"gaze_x", "gaze_y"}.issubset(gaze.columns):
        raise ValueError("Gaze metadata must contain gaze_x and gaze_y columns")
    if resize_mode not in ("stretch", "pad"):
        raise ValueError("resize_mode must be 'stretch' or 'pad'")

    source_height, source_width = src_img_size
    target_height, target_width = target_size
    if min(source_height, source_width, target_height, target_width) < 1:
        raise ValueError("Source and target image dimensions must be positive")

    if resize_mode == "stretch":
        resized_height, resized_width = target_height, target_width
        pad_top = pad_left = 0
    else:
        scale = min(
            target_height / source_height,
            target_width / source_width,
        )
        resized_height = max(1, round(source_height * scale))
        resized_width = max(1, round(source_width * scale))
        pad_top = (target_height - resized_height) // 2
        pad_left = (target_width - resized_width) // 2

    x = pd.to_numeric(gaze["gaze_x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(gaze["gaze_y"], errors="coerce").to_numpy(dtype=np.float64)
    in_bounds = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x < source_width)
        & (y >= 0)
        & (y < source_height)
    )
    x, y = x[in_bounds], y[in_bounds]

    x = np.rint(x * resized_width / source_width).astype(np.int64)
    y = np.rint(y * resized_height / source_height).astype(np.int64)
    x = np.clip(x, 0, resized_width - 1) + pad_left
    y = np.clip(y, 0, resized_height - 1) + pad_top

    fixation_mask = torch.zeros(
        (1, target_height, target_width), dtype=torch.float32
    )
    if x.size:
        fixation_mask[0, y, x] = 1.0
    return fixation_mask


@lru_cache(maxsize=128)
def load_csv(csv_path: str) -> pd.DataFrame:
    """Cache metadata by CSV path (rather than by recording directory)."""
    return pd.read_csv(csv_path)


class DG360LiteDataset(Dataset):
    """Load RGB clips for current-frame saliency prediction.

    Each sample uses the ``T`` RGB frames ending at frame ``t`` as input and
    the dataset-provided saliency map for frame ``t`` as its target. Both are
    resized according to ``resize_mode`` to the fixed ``img_size``.
    ``target_offset`` lets a short input (such as the single-frame baseline)
    sample the same target frames as a longer temporal clip.

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
        target_offset=0,
        augment=False,
        resize_mode="stretch",
    ):
        self.root = Path(root)
        self.T = int(T)
        self.stride = int(frame_stride)
        self.clip_step = int(clip_step)
        self.img_size = tuple(img_size)  # (H, W)
        self.skip_first = int(skip_first)
        self.target_offset = int(target_offset)
        self.augment = bool(augment)
        self.resize_mode = resize_mode

        if self.T < 1 or self.stride < 1 or self.clip_step < 1:
            raise ValueError("T, frame_stride, and clip_step must be positive integers")
        if self.target_offset < 0:
            raise ValueError("target_offset must be non-negative")
        if len(self.img_size) != 2 or any(size < 1 for size in self.img_size):
            raise ValueError("img_size must contain positive (height, width) values")
        if self.resize_mode not in ("stretch", "pad"):
            raise ValueError("resize_mode must be 'stretch' or 'pad'")
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
        self.color_jitter = v2.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.02,
        )

    def _make_paths(self):
        """Find iteration-level recordings under driver/session directories."""
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root}")

        paths = []
        driver_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        for driver_dir in driver_dirs:
            session_dirs = sorted(path for path in driver_dir.iterdir() if path.is_dir())
            for session_dir in session_dirs:
                recording_dirs = sorted(
                    path for path in session_dir.iterdir() if path.is_dir()
                )
                paths.extend(recording_dirs)
        return paths

    @staticmethod
    def _jpg_stems(folder):
        stems = [
            p.stem for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
        ]
        return sorted(
            stems,
            key=lambda stem: (0, int(stem)) if stem.isdecimal() else (1, stem),
        )

    def _check_files(self):
        """Validate each recording and return its ordered CSV frame IDs."""
        frame_ids_by_recording = []
        required_columns = {"Frame", "gaze_x", "gaze_y"}

        for rec_dir in self.data_paths:
            rgb_dir = rec_dir / "rgb"
            saliency_dir = rec_dir / "saliency"
            csv_path = rec_dir / "sim_gaze_df.csv"

            missing = [
                str(path)
                for path in (rgb_dir, saliency_dir)
                if not path.is_dir()
            ]
            if not csv_path.is_file():
                missing.append(str(csv_path))
            if missing:
                raise FileNotFoundError(
                    f"Incomplete recording {rec_dir}; missing: {', '.join(missing)}"
                )

            metadata = load_csv(str(csv_path))
            missing_columns = required_columns.difference(metadata.columns)
            if missing_columns:
                raise ValueError(
                    f"CSV {csv_path} is missing columns: "
                    f"{', '.join(sorted(missing_columns))}"
                )
            if metadata.empty:
                raise ValueError(f"CSV contains no frame rows: {csv_path}")

            raw_frame_ids = pd.to_numeric(metadata["Frame"], errors="coerce")
            if raw_frame_ids.isna().any() or not np.isfinite(raw_frame_ids).all():
                raise ValueError(f"CSV contains invalid Frame IDs: {csv_path}")
            if (raw_frame_ids % 1 != 0).any() or (raw_frame_ids < 1).any():
                raise ValueError(
                    f"Frame IDs must be positive integers in {csv_path}"
                )

            frame_ids = raw_frame_ids.astype(np.int64).tolist()
            if any(
                current <= previous
                for previous, current in zip(frame_ids, frame_ids[1:])
            ):
                raise ValueError(
                    f"Frame IDs must be unique and strictly increasing in {csv_path}"
                )

            expected_stems = [f"{frame_id:06d}" for frame_id in frame_ids]
            rgb_stems = self._jpg_stems(rgb_dir)
            saliency_stems = self._jpg_stems(saliency_dir)
            if rgb_stems != expected_stems:
                missing_stems = sorted(set(expected_stems) - set(rgb_stems))
                extra_stems = sorted(set(rgb_stems) - set(expected_stems))
                raise ValueError(
                    f"RGB filenames do not match CSV Frame IDs in {rec_dir}: "
                    f"RGB={len(rgb_stems)}, CSV={len(expected_stems)}, "
                    f"missing={missing_stems[:5]}, extra={extra_stems[:5]}"
                )
            if saliency_stems != expected_stems:
                missing_stems = sorted(set(expected_stems) - set(saliency_stems))
                extra_stems = sorted(set(saliency_stems) - set(expected_stems))
                raise ValueError(
                    f"Saliency filenames do not match CSV Frame IDs in {rec_dir}: "
                    f"saliency={len(saliency_stems)}, CSV={len(expected_stems)}, "
                    f"missing={missing_stems[:5]}, extra={extra_stems[:5]}"
                )

            frame_ids_by_recording.append(frame_ids)

        return frame_ids_by_recording

    def _make_samples(self):
        samples = []
        for recording_idx, n_frames in enumerate(self.data_len):
            # Largest valid start index, inclusive.
            last_start = (
                n_frames - 1 - self.target_offset - (self.T - 1) * self.stride
            )
            for start in range(self.skip_first, last_start + 1, self.clip_step):
                samples.append((recording_idx, start))
        return samples

    def _weight_cache_paths(self, cache_dir=None):
        """Return cache paths keyed by dataset and sample-generation settings."""
        cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else self.root / ".sample_weight_cache"
        )
        records = []
        for rec_dir, frame_ids in zip(self.data_paths, self.frame_ids):
            csv_path = rec_dir / "sim_gaze_df.csv"
            csv_stat = csv_path.stat()
            saliency_stat = (rec_dir / "saliency").stat()
            records.append(
                {
                    "record": str(rec_dir.resolve()),
                    "csv_size": csv_stat.st_size,
                    "csv_mtime_ns": csv_stat.st_mtime_ns,
                    "saliency_dir_mtime_ns": saliency_stat.st_mtime_ns,
                    "frames": len(frame_ids),
                }
            )
        signature = {
            "version": 1,
            "records": records,
            "T": self.T,
            "stride": self.stride,
            "clip_step": self.clip_step,
            "skip_first": self.skip_first,
            "target_offset": self.target_offset,
            "img_size": self.img_size,
            "resize_mode": self.resize_mode,
            "sample_count": len(self.samples),
        }
        key = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        return (
            cache_dir / f"average_saliency_{key}.pt",
            cache_dir / f"sample_kld_weights_{key}.pt",
        )

    @staticmethod
    def _load_tensor_cache(path, expected_shape=None):
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, ValueError):
            return None
        if not isinstance(value, torch.Tensor):
            return None
        if expected_shape is not None and tuple(value.shape) != tuple(expected_shape):
            return None
        if not torch.isfinite(value).all():
            return None
        return value.float()

    @staticmethod
    def _save_tensor_cache(value, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(value.cpu(), temporary_path)
        temporary_path.replace(path)

    def _target_saliency_distribution(self, sample_idx):
        recording_idx, start = self.samples[sample_idx]
        target_position = (
            start + self.target_offset + (self.T - 1) * self.stride
        )
        target_id = self.frame_ids[recording_idx][target_position]
        saliency = self._read(recording_idx, target_id, "saliency")
        valid = resize_image(
            torch.ones_like(saliency), self.img_size, self.resize_mode
        ) > 0
        saliency = self.gt_tf(
            resize_image(saliency, self.img_size, self.resize_mode)
        ).masked_fill(~valid, 0)
        total = saliency.sum()
        if not torch.isfinite(total) or total <= 0:
            rec_dir = self.data_paths[recording_idx]
            raise ValueError(
                "Cannot calculate sample weights from an empty or invalid "
                f"saliency target: {rec_dir}, frame {target_id:06d}"
            )
        return saliency / total

    def get_avg_saliency(self, cache_dir=None, force_recompute=False):
        """Compute the mean probability saliency map over training samples.

        The returned tensor has shape [1, H, W] and sums to one. Use this on
        the training dataset only; validation targets must not enter the cache.
        """
        average_path, _ = self._weight_cache_paths(cache_dir)
        if not force_recompute:
            cached = self._load_tensor_cache(
                average_path, expected_shape=(1, *self.img_size)
            )
            if cached is not None and cached.sum() > 0:
                return cached / cached.sum()

        running_sum = torch.zeros((1, *self.img_size), dtype=torch.float32)
        for sample_idx in range(len(self.samples)):
            running_sum += self._target_saliency_distribution(sample_idx)
        average = running_sum / len(self.samples)
        average = average / average.sum().clamp_min(torch.finfo(average.dtype).tiny)
        self._save_tensor_cache(average, average_path)
        return average

    def get_sample_weights(
        self, cache_dir=None, force_recompute=False, eps=1e-8
    ):
        """Return per-sample KL(target || average_saliency) weights on CPU.

        Weights follow ``self.samples`` order and are raw KLD values. The
        trainer is responsible for percentile clipping and mean normalization.
        """
        _, weights_path = self._weight_cache_paths(cache_dir)
        if not force_recompute:
            cached = self._load_tensor_cache(
                weights_path, expected_shape=(len(self.samples),)
            )
            if cached is not None and (cached >= 0).all():
                return cached

        average = self.get_avg_saliency(
            cache_dir=cache_dir, force_recompute=force_recompute
        ).flatten()
        log_average = average.clamp_min(eps).log()
        weights = torch.empty(len(self.samples), dtype=torch.float32)
        for sample_idx in range(len(self.samples)):
            target = self._target_saliency_distribution(sample_idx).flatten()
            weights[sample_idx] = (
                target * (target.clamp_min(eps).log() - log_average)
            ).sum().clamp_min(0)

        self._save_tensor_cache(weights, weights_path)
        return weights

    def __len__(self):
        return len(self.samples)

    def _read(self, recording_idx, frame_id, kind):
        image_dir = self.data_paths[recording_idx] / kind
        image_path = image_dir / f"{frame_id:06d}.jpg"
        if not image_path.is_file():
            image_path = image_dir / f"{frame_id:06d}.jpeg"
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
            start + self.target_offset + i * self.stride
            for i in range(self.T)
        ]
        frame_ids = [
            self.frame_ids[recording_idx][position] for position in frame_positions
        ]

        rgb_frames = []
        target_rgb_size = None
        for frame_id in frame_ids:
            frame = self._read(recording_idx, frame_id, "rgb")
            target_rgb_size = frame.shape[-2:]
            frame = resize_image(frame, self.img_size, self.resize_mode)
            rgb_frames.append(frame)
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
        # mask vùng ảnh thật, tạo từ kích thước gốc
        valid = resize_image(
            torch.ones_like(saliency), self.img_size, self.resize_mode
        )
        mask = (valid > 0).float()

        saliency = resize_image(saliency, self.img_size, self.resize_mode)
        saliency = self.gt_tf(saliency)

        csv_path = self.data_paths[recording_idx] / "sim_gaze_df.csv"
        metadata = load_csv(str(csv_path))
        target_gaze = metadata.loc[metadata["Frame"] == target_id]
        if len(target_gaze) != 1:
            raise ValueError(
                f"Expected one gaze annotation for target frame {target_id} "
                f"in {csv_path}, found {len(target_gaze)}"
            )
        fixation = make_fixation_mask(
            target_gaze,
            src_img_size=target_rgb_size,
            target_size=self.img_size,
            resize_mode=self.resize_mode,
        )

        if self.augment:
            if torch.rand(()) < 0.5:
                rgb_clip = rgb_clip.flip(-1)
                saliency = saliency.flip(-1)
                mask = mask.flip(-1)
                fixation = fixation.flip(-1)
            rgb_clip = self.color_jitter(rgb_clip)
            rgb_clip = rgb_clip.masked_fill(mask.unsqueeze(0) == 0, 0)

        rgb_clip = torch.stack(
            [self.rgb_tf(frame) for frame in rgb_clip],
            dim=0,
        )

        return {
            "rgb": rgb_clip,
            "sal": saliency,
            "mask": mask,
            "fixation": fixation,
            "sample_idx": torch.tensor(index, dtype=torch.long),
        }
