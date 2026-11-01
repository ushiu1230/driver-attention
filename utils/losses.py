import torch
import torch.nn as nn


class SaliencyLoss(nn.Module):
    """Masked regression loss for continuous saliency-intensity maps."""

    def __init__(
        self, lambda_mse: float = 1.0, lambda_cc: float = 0.0, eps: float = 1e-8
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_cc = lambda_cc
        self.eps = eps

    def forward(self, prediction, target, mask=None):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError(
                "prediction and target must have matching [B, C, H, W] shapes"
            )
        if mask is not None and mask.shape != target.shape:
            raise ValueError("mask must have the same shape as prediction and target")

        pred = prediction.float().flatten(start_dim=1)
        gt = target.float().flatten(start_dim=1)
        valid = (
            torch.ones_like(pred, dtype=torch.bool)
            if mask is None
            else mask.flatten(start_dim=1).bool()
        )
        valid_count = valid.sum(dim=1)
        if (valid_count == 0).any():
            raise ValueError("Each sample must contain valid image pixels")

        difference = (pred - gt).masked_fill(~valid, 0)
        per_sample_mse = difference.square().sum(dim=1) / valid_count
        loss = self.lambda_mse * per_sample_mse

        if self.lambda_cc:
            count = valid_count.unsqueeze(1)
            pred_centered = (
                pred - pred.masked_fill(~valid, 0).sum(dim=1, keepdim=True) / count
            ).masked_fill(~valid, 0)
            gt_centered = (
                gt - gt.masked_fill(~valid, 0).sum(dim=1, keepdim=True) / count
            ).masked_fill(~valid, 0)
            numerator = (pred_centered * gt_centered).sum(dim=1)
            denominator = (
                pred_centered.square().sum(dim=1)
                * gt_centered.square().sum(dim=1)
            ).sqrt().clamp_min(torch.finfo(pred.dtype).tiny)
            cc = numerator / denominator
            loss = loss + self.lambda_cc * (1.0 - cc)

        return loss.mean()
