"""
This is a implement of the
Driver Attention Prediction using lightweight convolutional neural network (CNN)
with tsm module, trained on DriverGaze360 dataset.

"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models

# define backbone (function name, weight, )


class TemporalShift(nn.Module):
    """TSM (Lin et al., ICCV 2019).
    Input shape: (B*T, C, H, W).
    """

    def __init__(self, n_segment, fold_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.shape
        t = self.n_segment

        # Reshape the batch of frames into video clips
        x = x.view(nt // t, t, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)

        # Shift the first channel fold from the next frame
        out[:, :-1, :fold] = x[:, 1:, :fold]

        # Shift the second channel fold from the previous frame
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]

        # Keep the remaining channels unchanged
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(nt, c, h, w)


def insert_tsm(module, n_segment, fold_div=8):
    """Insert TSM into residual branches of a backbone."""

    targets = [
        m for m in module.modules()
        if getattr(m, "use_res_connect", False)
    ]

    for m in targets:
        # MobileNetV3 uses 'block'; MobileNetV2 uses 'conv'
        name = "block" if hasattr(m, "block") else "conv"

        setattr(
            m,
            name,
            nn.Sequential(
                TemporalShift(n_segment, fold_div),
                getattr(m, name),
            ),
        )

    return len(targets)


class TemporalPool(nn.Module):
    """Pool (B*T, C, H, W) into (B, C, H, W)."""

    def __init__(self, mode, channels):
        super().__init__()
        self.mode = mode

        if mode == "attn":
            self.score = nn.Linear(channels, 1)
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)

        elif mode not in ("mean", "last"):
            raise ValueError(f"Unsupported temporal mode: {mode}")

    def forward(self, x, B, T):
        # x: [B*T, C, H, W]
        x = x.reshape(B, T, *x.shape[1:])

        if self.mode == "last":
            return x[:, -1]

        if self.mode == "mean":
            return x.mean(dim=1)

        if self.mode == "attn":
            # Global spatial average for each frame
            # [B, T, C, H, W] -> [B, T, C]
            pooled = x.mean(dim=(-2, -1))

            # Score each frame: [B, T, 1] -> [B, T]
            scores = self.score(pooled).squeeze(-1)

            # Normalize weights over time
            weights = torch.softmax(scores, dim=1)

            # Weighted sum of feature maps
            return (
                x * weights[:, :, None, None, None]
            ).sum(dim=1)
