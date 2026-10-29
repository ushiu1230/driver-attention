import torch
import torch.nn as nn


class SaliencyLoss(nn.Module):
    def __init__(self, lambda_cc: float = 0.0, eps: float = 1e-8):
        super().__init__()
        self.lambda_cc = lambda_cc
        self.eps = eps

    def forward(self, prediction, target):
        # prediction, target: [B, 1, H, W]
        pred = prediction.flatten(start_dim=1).clamp_min(self.eps)
        gt = target.flatten(start_dim=1).clamp_min(0.0)

        # Chuẩn hóa từng ảnh thành phân phối xác suất
        pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(self.eps)
        gt = gt / gt.sum(dim=1, keepdim=True).clamp_min(self.eps)

        kld = (
            gt * (
                gt.clamp_min(self.eps).log()
                - pred.clamp_min(self.eps).log()
            )
        ).sum(dim=1).mean()

        if self.lambda_cc == 0:
            return kld

        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        gt_centered = gt - gt.mean(dim=1, keepdim=True)

        numerator = (pred_centered * gt_centered).sum(dim=1)
        denominator = (
            pred_centered.square().sum(dim=1)
            * gt_centered.square().sum(dim=1)
        ).sqrt().clamp_min(self.eps)

        cc = numerator / denominator
        cc_loss = (1.0 - cc).mean()

        return kld + self.lambda_cc * cc_loss