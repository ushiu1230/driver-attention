
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.transforms import v2


@lru_cache(maxsize=None)
def load_csv(iter_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(iter_dir, "sim_gaze_df.csv"))


class DG360LiteDataset(Dataset):
    def __init__(
        self,
        root,
        T=16,
        frame_stride=1,
        overlap=8,  # Thực chất là bước nhảy giữa 2 điểm bắt đầu clip
        img_size=(224, 1120),  # (H, W) mặc định của code gốc
        skip_first=30,  # 1s x 30 FPS, giống code gốc
        src_size=(6400, 720),  # (W, H) của ảnh panorama gốc, dùng cho gaze_x/y
        check=True,
    ):
        self.root = root
        self.T = T
        self.stride = frame_stride
        self.overlap = overlap
        self.img_size = tuple(img_size)
        self.skip_first = skip_first
        self.src_size = src_size

        self.data_paths = self._make_paths()
        self.data_len = self._check_files(check)
        self.samples = self._make_samples()

        self.rgb_tf = v2.Compose([
            v2.Resize(size=self.img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.gt_tf = v2.Compose([
            v2.Resize(size=self.img_size),
            v2.ToDtype(torch.float32, scale=True),
        ])

    # Xây dựng danh sách mẫu
    def _make_paths(self):
        paths = []
        for user in sorted(os.listdir(self.root)):
            up = os.path.join(self.root, user)
            if not os.path.isdir(up):
                continue

            for session in sorted(os.listdir(up)):
                sp = os.path.join(up, session)
                if not os.path.isdir(sp):
                    continue

                for it in sorted(os.listdir(sp)):
                    ip = os.path.join(sp, it)
                    if os.path.isdir(ip):
                        paths.append(ip)

        return sorted(paths)

    def _check_files(self, check):
        lens = []
        for p in self.data_paths:
            meta = load_csv(p)

            if check:
                rgb = [
                    Path(f).stem
                    for f in sorted(os.listdir(os.path.join(p, "rgb")))
                ]
                sal = [
                    Path(f).stem
                    for f in sorted(os.listdir(os.path.join(p, "saliency")))
                ]
                idx = [f"{int(i):06d}" for i in sorted(meta.Frame)]


                assert rgb == sal, f"rgb/saliency khác tên file: {p}"
                assert rgb == idx, f"tên file khác cột Frame trong CSV: {p}"

            lens.append(len(meta))

        return lens

    def _make_samples(self):
        samples = []
        for d, n in enumerate(self.data_len):
            max_start = n - self.T * self.stride
            for s in range(self.skip_first, max_start, self.overlap):
                samples.append((d, s))

        return samples

    def __len__(self):
        return len(self.samples)

    # # Đọc dữ liệu
    # def _read(self, d, frame_idx, kind, mode):
    #     # Dòng 0 của CSV -> file 000001.jpg
    #     path = os.path.join(
    #         self.data_paths[d],
    #         kind,
    #         f"{frame_idx + 1:06d}.jpg",
    #     )
    #     # Phần tiếp theo không hiển thị đầy đủ trong ảnh.
