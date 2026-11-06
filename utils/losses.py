import torch
import torch.nn as nn


class SaliencyLoss(nn.Module):
    """Masked saliency objective with optional KLD, CC, MSE, and NSS terms."""

    def __init__(
        self,
        lambda_kld: float = 1.0,
        lambda_cc: float = 1.0,
        lambda_mse: float = 0.0,
        lambda_nss: float = 0.0,
        eps: float = 1e-8,
        lambda_sal: float = 1.0,
        profile: str = "project",
    ):
        super().__init__()
        self.lambda_sal = lambda_sal
        self.lambda_kld = lambda_kld
        self.lambda_cc = lambda_cc
        self.lambda_mse = lambda_mse
        self.lambda_nss = lambda_nss
        self.eps = eps
        if profile not in ("project", "drivergaze360"):
            raise ValueError("profile must be 'project' or 'drivergaze360'")
        self.profile = profile

    def forward(
        self, prediction, target, mask=None, fixation=None, sample_weights=None
    ):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError(
                "prediction and target must have matching [B, C, H, W] shapes"
            )
        if mask is not None and mask.shape != target.shape:
            raise ValueError("mask must have the same shape as prediction and target")
        if self.lambda_nss and (
            fixation is None or fixation.shape != target.shape
        ):
            raise ValueError(
                "NSS loss requires a fixation map with the same shape as target"
            )

        if sample_weights is not None:
            if sample_weights.ndim != 1 or sample_weights.shape[0] != prediction.shape[0]:
                raise ValueError("sample_weights must have shape [batch_size]")
            if not torch.isfinite(sample_weights).all() or (sample_weights < 0).any():
                raise ValueError("sample_weights must be finite and non-negative")

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

        loss = torch.zeros_like(valid_count, dtype=pred.dtype)

        if self.lambda_kld:
            pred_valid = pred.masked_fill(~valid, 0)
            gt_valid = gt.masked_fill(~valid, 0)
            pred_sum = pred_valid.sum(dim=1, keepdim=True)
            gt_sum = gt_valid.sum(dim=1, keepdim=True)
            if (pred_sum <= 0).any() or (gt_sum <= 0).any():
                raise ValueError(
                    "KLD requires positive prediction and target mass per sample"
                )

            if self.profile == "drivergaze360":
                # Match the local source's sum+eps normalization and clamp.
                pred_prob = (pred_valid / (pred_sum + self.eps)).clamp_min(self.eps)
                gt_prob = (gt_valid / (gt_sum + self.eps)).clamp_min(self.eps)
            else:
                pred_prob = pred_valid / pred_sum
                gt_prob = gt_valid / gt_sum
            per_sample_kld = (
                gt_prob
                * (
                    gt_prob.clamp_min(self.eps).log()
                    - pred_prob.clamp_min(self.eps).log()
                )
            ).sum(dim=1)
            loss = loss + self.lambda_kld * per_sample_kld

        if self.lambda_cc:
            count = valid_count.unsqueeze(1)
            if self.profile == "drivergaze360":
                pred_sum = pred.masked_fill(~valid, 0).sum(dim=1, keepdim=True)
                gt_sum = gt.masked_fill(~valid, 0).sum(dim=1, keepdim=True)
                pred_dist = pred.masked_fill(~valid, 0) / (pred_sum + self.eps)
                gt_dist = gt.masked_fill(~valid, 0) / (gt_sum + self.eps)
                pred_centered = (
                    pred_dist - pred_dist.sum(dim=1, keepdim=True) / count
                ).masked_fill(~valid, 0)
                gt_centered = (
                    gt_dist - gt_dist.sum(dim=1, keepdim=True) / count
                ).masked_fill(~valid, 0)
                # Match source unbiased standard deviations and denominator,
                # while clamping before sqrt to keep constant maps differentiable.
                pred_var = pred_centered.square().sum(dim=1, keepdim=True) / (count - 1).clamp_min(1)
                gt_var = gt_centered.square().sum(dim=1, keepdim=True) / (count - 1).clamp_min(1)
                pred_std = pred_var.clamp_min(self.eps ** 2).sqrt() + self.eps
                gt_std = gt_var.clamp_min(self.eps ** 2).sqrt() + self.eps
                pred_norm = pred_centered / pred_std
                gt_norm = gt_centered / gt_std
                numerator = (pred_norm * gt_norm).sum(dim=1)
                denominator = (
                    pred_norm.square().sum(dim=1)
                    * gt_norm.square().sum(dim=1)
                    + self.eps
                ).sqrt()
                cc = numerator / denominator
                loss = loss + self.lambda_cc * (1.0 - cc)
            else:
                pred_centered = (
                    pred - pred.masked_fill(~valid, 0).sum(dim=1, keepdim=True) / count
                ).masked_fill(~valid, 0)
                gt_centered = (
                    gt - gt.masked_fill(~valid, 0).sum(dim=1, keepdim=True) / count
                ).masked_fill(~valid, 0)
                numerator = (pred_centered * gt_centered).sum(dim=1)
                variance_product = (
                    pred_centered.square().sum(dim=1)
                    * gt_centered.square().sum(dim=1)
                )
                # Correlation is undefined for a constant map. Assign zero CC
                # and zero CC gradient there; never differentiate sqrt at zero.
                has_variance = variance_product > torch.finfo(pred.dtype).tiny
                denominator = torch.where(
                    has_variance, variance_product, torch.ones_like(variance_product)
                ).sqrt()
                cc = torch.where(
                    has_variance, numerator / denominator, torch.zeros_like(numerator)
                )
                loss = loss - self.lambda_cc * cc

        if self.lambda_mse:
            difference = (pred - gt).masked_fill(~valid, 0)
            per_sample_mse = difference.square().sum(dim=1) / valid_count
            loss = loss + self.lambda_mse * per_sample_mse

        if self.lambda_nss:
            fixation_points = fixation.flatten(start_dim=1).bool() & valid
            fixation_count = fixation_points.sum(dim=1)
            if (fixation_count == 0).any():
                raise ValueError(
                    "NSS loss requires at least one valid fixation point per sample"
                )
            if (valid_count < 2).any():
                raise ValueError("NSS loss requires at least two valid image pixels")

            count = valid_count.unsqueeze(1)
            pred_centered = (
                pred - pred.masked_fill(~valid, 0).sum(dim=1, keepdim=True) / count
            ).masked_fill(~valid, 0)
            pred_std = (
                pred_centered.square().sum(dim=1, keepdim=True)
                / (count - 1)
            ).clamp_min(self.eps ** 2).sqrt()
            pred_normalized = pred_centered / pred_std
            nss = (
                pred_normalized * fixation_points.to(pred.dtype)
            ).sum(dim=1) / fixation_count
            loss = loss - self.lambda_nss * nss

        if sample_weights is not None:
            loss = loss * sample_weights.to(device=loss.device, dtype=loss.dtype)
        return self.lambda_sal * loss.mean()
