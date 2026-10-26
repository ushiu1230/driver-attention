import torch.nn as nn
import torch.nn.functional as F


class SaliencyDecoder(nn.Module):
    def __init__(self, in_channels=(24, 40, 112, 960), width=128):
        super().__init__()
        if not in_channels or any(channels < 1 for channels in in_channels):
            raise ValueError("in_channels must contain positive channel counts")
        if width < 1:
            raise ValueError("width must be positive")

        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, width, kernel_size=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
            for channels in in_channels
        ])

        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, padding=1,
                          groups=width, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, kernel_size=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
            for _ in range(len(in_channels) - 1)
        ])

        # Produces saliency logits; apply sigmoid for visualization/inference.
        self.head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, features, output_size):
        if len(features) != len(self.lateral):
            raise ValueError(
                f"Expected {len(self.lateral)} feature maps, got {len(features)}"
            )
        if len(output_size) != 2 or any(size < 1 for size in output_size):
            raise ValueError("output_size must contain positive (height, width)")
        # features are ordered from highest spatial resolution to lowest.
        x = self.lateral[-1](features[-1])

        for i in range(len(features) - 2, -1, -1):
            x = F.interpolate(
                x,
                size=features[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = self.refine[i](x + self.lateral[i](features[i]))

        x = F.interpolate(
            x, size=output_size, mode="bilinear", align_corners=False
        )
        return self.head(x)  # [B, 1, H, W] logits