"""
This is a implement of the
Driver Attention Prediction using lightweight convolutional neural network (CNN)
with tsm module, trained on DriverGaze360 dataset.

"""

import torch
import torch.nn as nn
from torchvision import models
from models.saliency_decoder import SaliencyDecoder

# define backbone presets used by the driver-attention model builder
# each entry: (constructor, weights enum, weights variant, feature indices, channels)
BACKBONES = {
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        "MobileNet_V3_Large_Weights",
        "IMAGENET1K_V2",
        (3, 6, 12, 16),
        (24, 40, 112, 960),
    ),
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        "MobileNet_V3_Small_Weights",
        "IMAGENET1K_V1",
        (1, 3, 8, 12),
        (16, 24, 48, 576),
    ),
    "mobilenet_v2": (
        models.mobilenet_v2,
        "MobileNet_V2_Weights",
        "IMAGENET1K_V1",
        (3, 6, 13, 18),
        (24, 32, 96, 1280),
    ),
}


def get_backbone(name):
    """Return constructor, weights, stage indices, and channels for a preset."""
    try:
        return BACKBONES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported backbone: {name}") from exc


class TemporalShift(nn.Module):
    """TSM (Lin et al., ICCV 2019).
    Input shape: (B*T, C, H, W).
    """

    def __init__(self, n_segment, fold_div=8):
        super().__init__()
        if n_segment < 1:
            raise ValueError("n_segment must be positive")
        if fold_div < 1:
            raise ValueError("fold_div must be positive")
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"TemporalShift expects [B*T, C, H, W], got {tuple(x.shape)}"
            )
        nt, c, h, w = x.shape
        t = self.n_segment
        if nt % t != 0:
            raise ValueError(
                f"Leading dimension {nt} must be divisible by n_segment={t}"
            )

        # Reshape the batch of frames into video clips
        x = x.reshape(nt // t, t, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)

        # Shift the first channel fold from the next frame
        out[:, :-1, :fold] = x[:, 1:, :fold]

        # Shift the second channel fold from the previous frame
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]

        # Keep the remaining channels unchanged
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.reshape(nt, c, h, w)


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
        if channels < 1:
            raise ValueError("channels must be positive")
        self.mode = mode

        if mode == "attn":
            self.score = nn.Linear(channels, 1)
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)

        elif mode not in ("mean", "last"):
            raise ValueError(f"Unsupported temporal mode: {mode}")

    def forward(self, x, B, T):
        if x.ndim != 4:
            raise ValueError(
                f"TemporalPool expects [B*T, C, H, W], got {tuple(x.shape)}"
            )
        if B < 1 or T < 1 or x.shape[0] != B * T:
            raise ValueError(
                f"Expected leading dimension B*T={B * T}, got {x.shape[0]}"
            )
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


class DriverGazeNet(nn.Module):
    """Predict the final frame's saliency map from an RGB clip.

    Input: [B, T, 3, H, W]. Output: saliency logits [B, 1, H, W].
    """

    def __init__(
        self,
        backbone_name="mobilenet_v3_large",
        num_segments=16,
        temporal_mode="last",
        use_tsm=True,
        pretrained=True,
        decoder_width=128,
        fold_div=8,
    ):
        super().__init__()
        if num_segments < 1:
            raise ValueError("num_segments must be positive")

        (
            constructor,
            weights_enum_name,
            weights_variant,
            self.stage_indices,
            stage_channels,
        ) = get_backbone(backbone_name)

        weights_enum = getattr(models, weights_enum_name)
        weights = getattr(weights_enum, weights_variant) if pretrained else None
        self.backbone = constructor(weights=weights)
        self.num_segments = num_segments
        self.use_tsm = use_tsm
        self.temporal_mode = temporal_mode

        if use_tsm:
            insert_tsm(self.backbone.features, num_segments, fold_div)

        self.temporal_pools = nn.ModuleList(
            TemporalPool(temporal_mode, channels) for channels in stage_channels
        )
        self.decoder = SaliencyDecoder(
            in_channels=stage_channels,
            width=decoder_width,
        )

    def forward(self, rgb):
        if rgb.ndim != 5:
            raise ValueError(
                f"Expected RGB clips shaped [B, T, 3, H, W], got {tuple(rgb.shape)}"
            )
        batch_size, num_frames, channels, height, width = rgb.shape
        if channels != 3:
            raise ValueError(f"Expected 3 RGB channels, got {channels}")
        if num_frames < 1 or height < 1 or width < 1:
            raise ValueError("Input clips must have positive frame and image dimensions")
        if self.use_tsm and num_frames != self.num_segments:
            raise ValueError(
                f"Expected T={self.num_segments} frames for TSM, got T={num_frames}"
            )

        if not self.use_tsm and self.temporal_mode == "last":
            rgb = rgb[:, -1:]
            num_frames = 1

        x = rgb.reshape(batch_size * num_frames, channels, height, width)
        features = []
        stage_positions = {
            index: position for position, index in enumerate(self.stage_indices)
        }

        for index, layer in enumerate(self.backbone.features):
            x = layer(x)
            if index in stage_positions:
                features.append(x)

        if len(features) != len(self.stage_indices):
            raise RuntimeError(
                "Backbone did not produce all configured feature stages; "
                f"found {len(features)} of {len(self.stage_indices)}"
            )

        pooled_features = [
            temporal_pool(feature, batch_size, num_frames)
            for temporal_pool, feature in zip(self.temporal_pools, features)
        ]
        return self.decoder(pooled_features, output_size=(height, width))
