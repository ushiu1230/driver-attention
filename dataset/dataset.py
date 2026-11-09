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
    if tuple(image.shape[-2:]) == tuple(size) and mode in ("stretch", "pad"):
        return image
    if mode == "stretch":
        return v2.functional.resize(image, size=list(size), antialias=True)
    if mode == "pad":
        return resize_with_pad(image, size)
    raise ValueError(f"Unsupported resize mode: {mode}")


def make_fixation_mask(gaze, src_img_size, target_size, resize_mode="stretch", paper_protocol=False):
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
    if paper_protocol:
        # Source implementation rounds first, then rejects out-of-canvas points.
        x = np.rint(x * resized_width / source_width) + pad_left
        y = np.rint(y * resized_height / source_height) + pad_top
        valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < target_width) & (y >= 0) & (y < target_height)
        mask = torch.zeros((1, target_height, target_width), dtype=torch.float32)
        mask[0, y[valid].astype(np.int64), x[valid].astype(np.int64)] = 1
        return mask
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


def paper_fixation_frame_ids(frame_ids, start, T, stride=1, target_offset=0):
    """Return the source protocol's input and following-clip gaze frame IDs."""
    positions = [
        start + target_offset + i * stride for i in range(2 * T)
        if start + target_offset + i * stride < len(frame_ids)
    ]
    return [frame_ids[position] for position in positions]


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
        paper_protocol=False,
        gaze_source_size=None,
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
        self.paper_protocol = bool(paper_protocol)
        self.gaze_source_size = tuple(gaze_source_size) if gaze_source_size is not None else None
        if self.gaze_source_size is not None and (len(self.gaze_source_size) != 2 or min(self.gaze_source_size) < 1):
            raise ValueError("gaze_source_size must contain positive (H, W)")
        if self.T < 1 or self.stride < 1 or self.clip_step < 1:
            raise ValueError("T, frame_stride, and clip_step must be positive integers")
        if self.paper_protocol and self.target_offset % self.stride:
            raise ValueError("paper_protocol requires target_offset divisible by frame_stride")
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

        self.preprocessing = [self._load_preprocessing(rec) for rec in self.data_paths]

        self.frame_ids = self._check_files()
        if self.paper_protocol:
            for rec_dir, ids in zip(self.data_paths, self.frame_ids):
                if any(b != a + 1 for a, b in zip(ids, ids[1:])):
                    raise ValueError(
                        "The local paper indexes frames as contiguous positions; "
                        f"non-contiguous frame IDs found in {rec_dir}."
                    )
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
            if self.paper_protocol:
                # Match the local paper source: range(30, N - T*stride, step).
                upper = n_frames - self.T * self.stride - self.target_offset
                for start in range(self.skip_first, upper, self.clip_step):
                    samples.append((recording_idx, start))
                continue
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
            "version": 3,
            "paper_protocol": self.paper_protocol,
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
        if not torch.isfinite(saliency).all() or (saliency < 0).any():
            rec_dir = self.data_paths[recording_idx]
            raise ValueError(
                "Cannot calculate sample weights from an invalid "
                f"saliency target: {rec_dir}, frame {target_id:06d}"
            )
        if total <= 0:
            return None
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
            if cached is not None and (cached >= 0).all() and cached.sum() > 0:
                return cached / cached.sum()

        running_sum = torch.zeros((1, *self.img_size), dtype=torch.float32)
        valid_samples = 0
        for sample_idx in range(len(self.samples)):
            target = self._target_saliency_distribution(sample_idx)
            if target is None:
                continue
            running_sum += target
            valid_samples += 1
        if valid_samples == 0:
            raise RuntimeError("No non-empty saliency targets available for the training prior")
        average = running_sum / valid_samples
        average = average / average.sum().clamp_min(torch.finfo(average.dtype).tiny)
        self._save_tensor_cache(average, average_path)
        print(
            f"Saliency prior: retained {valid_samples}/{len(self.samples)} targets; "
            f"empty saliency={len(self.samples) - valid_samples}", flush=True,
        )
        return average

    def get_sample_weights(
        self, cache_dir=None, force_recompute=False, eps=1e-8
    ):
        """Return per-sample KL(target || average_saliency) weights on CPU.

        Weights follow ``self.samples`` order and are raw KLD values. Empty
        targets have zero weight and False in ``valid_saliency_samples``.
        The trainer clips and normalizes weights over valid samples only.
        """
        _, weights_path = self._weight_cache_paths(cache_dir)
        validity_path = weights_path.with_name("validity_" + weights_path.name)
        if not force_recompute:
            cached = self._load_tensor_cache(
                weights_path, expected_shape=(len(self.samples),)
            )
            validity = self._load_tensor_cache(
                validity_path, expected_shape=(len(self.samples),)
            )
            if (
                cached is not None and (cached >= 0).all()
                and validity is not None and ((validity == 0) | (validity == 1)).all()
                and validity.any() and (cached[~validity.bool()] == 0).all()
            ):
                self.valid_saliency_samples = validity.bool()
                return cached

        average = self.get_avg_saliency(
            cache_dir=cache_dir, force_recompute=force_recompute
        ).flatten()
        log_average = average.clamp_min(eps).log()
        weights = torch.zeros(len(self.samples), dtype=torch.float32)
        validity = torch.zeros(len(self.samples), dtype=torch.bool)
        for sample_idx in range(len(self.samples)):
            target = self._target_saliency_distribution(sample_idx)
            if target is None:
                continue
            validity[sample_idx] = True
            target = target.flatten()
            weights[sample_idx] = (
                target * (target.clamp_min(eps).log() - log_average)
            ).sum().clamp_min(0)

        if not validity.any():
            raise RuntimeError("No non-empty saliency targets available for sample weighting")
        self._save_tensor_cache(weights, weights_path)
        self._save_tensor_cache(validity, validity_path)
        self.valid_saliency_samples = validity
        return weights

    def __len__(self):
        return len(self.samples)

    def _load_preprocessing(self, recording):
        path = recording / "preprocessing.json"
        if not path.exists():
            if (recording / "rgb" / ".prepared").exists():
                raise ValueError(f"Missing geometry metadata in {recording}; rerun prepare_dataset.sh")
            return None  # Legacy, full-resolution data remains supported.
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            for key in ("image_size", "gaze_source_size"):
                size = info[key]
                if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
                    raise ValueError(f"Invalid {key}")
            if info["version"] != 1 or info["resize_mode"] != "stretch" or info["csv_coordinates"] != "unchanged":
                raise ValueError("Unsupported preprocessing geometry")
            if self.resize_mode != "stretch":
                raise ValueError("Pre-resized stretch data requires resize_mode=stretch")
            if self.gaze_source_size is not None and self.gaze_source_size != tuple(info["gaze_source_size"]):
                raise ValueError("gaze_source_size conflicts with the recorded CSV canvas")
            return info
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid geometry metadata {path}: {error}") from error

    def _gaze_canvas(self, recording_idx, decoded_size):
        info = self.preprocessing[recording_idx]
        if info is not None:
            if tuple(decoded_size) != tuple(info["image_size"]):
                raise ValueError(f"JPEG dimensions disagree with preprocessing.json in {self.data_paths[recording_idx]}")
            return tuple(info["gaze_source_size"])
        if self.gaze_source_size is not None:
            return self.gaze_source_size
        if self.paper_protocol:
            return (720, 6400)
        # A resized image alone cannot identify the original CSV coordinate canvas.
        if tuple(decoded_size) == self.img_size:
            raise ValueError("Ambiguous gaze coordinates: set gaze_source_size or rerun prepare_dataset.sh to record geometry")
        return decoded_size

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
        info = self.preprocessing[recording_idx]
        if info is not None and tuple(image.shape[-2:]) != tuple(info["image_size"]):
            raise ValueError(f"JPEG dimensions disagree with preprocessing.json: {image_path}")
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
        gaze_canvas = self._gaze_canvas(recording_idx, target_rgb_size)
        # mask vùng ảnh thật, tạo từ kích thước gốc
        valid = resize_image(
            torch.ones_like(saliency), self.img_size, self.resize_mode
        )
        mask = (valid > 0).float()

        saliency = resize_image(saliency, self.img_size, self.resize_mode)
        saliency = self.gt_tf(saliency)

        csv_path = self.data_paths[recording_idx] / "sim_gaze_df.csv"
        metadata = load_csv(str(csv_path))
        if self.paper_protocol:
            fixation_ids = set(paper_fixation_frame_ids(
                self.frame_ids[recording_idx], start,
                self.T + self.target_offset // self.stride, self.stride,
            ))
            fixation_gaze = metadata.loc[metadata["Frame"].isin(fixation_ids)]
        else:
            fixation_gaze = metadata.loc[metadata["Frame"] == target_id]
        if fixation_gaze.empty:
            # The source trainer filters empty fixation maps before loss.
            fixation = torch.zeros((1, *self.img_size), dtype=torch.float32)
        else:
            fixation = make_fixation_mask(
                fixation_gaze,
                src_img_size=gaze_canvas,
                target_size=self.img_size,
                resize_mode=self.resize_mode,
                paper_protocol=self.paper_protocol,
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
