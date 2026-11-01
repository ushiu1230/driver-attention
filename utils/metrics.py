import torch
import torch.nn as nn


class SaliencyMetrics(nn.Module):
    """Compute per-sample saliency metrics over valid image pixels."""

    def forward(self, prediction, target, mask=None):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError(
                "prediction and target must have matching [B, C, H, W] shapes"
            )
        if prediction.shape[1] != 1:
            raise ValueError("Saliency metrics expect a single-channel map")
        if mask is not None and mask.shape != target.shape:
            raise ValueError("mask must have the same shape as prediction and target")

        pred_intensity = prediction.float().flatten(start_dim=1).clamp(0.0, 1.0)
        gt_intensity = target.float().flatten(start_dim=1).clamp(0.0, 1.0)
        valid = (
            torch.ones_like(pred_intensity, dtype=torch.bool)
            if mask is None
            else mask.flatten(start_dim=1).bool()
        )
        valid_count = valid.sum(dim=1)
        if (valid_count == 0).any():
            raise ValueError("Each sample must contain valid image pixels")

        pred_intensity = pred_intensity.masked_fill(~valid, 0)
        gt_intensity = gt_intensity.masked_fill(~valid, 0)
        pred = pred_intensity / pred_intensity.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        gt = gt_intensity / gt_intensity.sum(dim=1, keepdim=True).clamp_min(1e-8)

        kld = (
            gt
            * (
                gt.clamp_min(1e-8).log()
                - pred.clamp_min(1e-8).log()
            )
        ).sum(dim=1)

        count = valid_count.unsqueeze(1)
        pred_centered = (
            pred_intensity
            - pred_intensity.sum(dim=1, keepdim=True) / count
        ).masked_fill(~valid, 0)
        gt_centered = (
            gt_intensity
            - gt_intensity.sum(dim=1, keepdim=True) / count
        ).masked_fill(~valid, 0)
        cc = (pred_centered * gt_centered).sum(dim=1) / (
            pred_centered.square().sum(dim=1)
            * gt_centered.square().sum(dim=1)
        ).sqrt().clamp_min(torch.finfo(pred.dtype).tiny)
        similarity = torch.minimum(pred, gt).sum(dim=1)

        height, width = target.shape[-2:]
        pred_peak = pred_intensity.masked_fill(
            ~valid, float("-inf")
        ).argmax(dim=1)
        gt_peak = gt_intensity.masked_fill(~valid, float("-inf")).argmax(dim=1)
        pred_y, pred_x = pred_peak // width, pred_peak % width
        gt_y, gt_x = gt_peak // width, gt_peak % width
        peak_error = torch.sqrt(
            (pred_x - gt_x).float().square()
            + (pred_y - gt_y).float().square()
        )
        return {
            "KLD": kld,
            "CC": cc,
            "SIM": similarity,
            "peak_px": peak_error,
        }