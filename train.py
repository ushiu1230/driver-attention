import torch
from torch.utils.data import DataLoader

from dataset.dataset import DG360LiteDataset
from models import SingleFrameSaliencyNet


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            rgb = batch["rgb"].to(device)  # [B, T, 3, H, W]
            target = batch["sal"].to(device)  # [B, 1, H, W]

            # Bỏ mẫu có target rỗng để tránh chuẩn hóa phân phối không hợp lệ.
            valid = target.flatten(start_dim=1).sum(dim=1) > 1e-8
            if not valid.any():
                continue

            rgb = rgb[valid]
            target = target[valid]

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = model(rgb)
            loss = criterion(prediction, target)

            if training:
                loss.backward()
                optimizer.step()

            count = target.size(0)
            total_loss += loss.detach().item() * count
            total_samples += count

    if total_samples == 0:
        raise RuntimeError("Epoch không có mẫu saliency hợp lệ.")

    return total_loss / total_samples


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = DG360LiteDataset(
    root="data/processed/train",
    T=16,
    frame_stride=1,
    clip_step=8,
    img_size=(224, 1120),
)
val_ds = DG360LiteDataset(
    root="data/processed/val",
    T=16,
    frame_stride=1,
    clip_step=8,
    img_size=(224, 1120),
)

train_loader = DataLoader(
    train_ds, batch_size=2, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_ds, batch_size=2, shuffle=False, num_workers=2
)

model = SingleFrameSaliencyNet(pretrained=True).to(device)
criterion = SaliencyLoss(lambda_cc=0.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

best_val_loss = float("inf")

for epoch in range(20):
    train_loss = run_epoch(
        model, train_loader, criterion, device, optimizer=optimizer
    )
    val_loss = run_epoch(
        model, val_loader, criterion, device, optimizer=None
    )

    print(
        f"Epoch {epoch + 1:02d}: "
        f"train KLD={train_loss:.4f}, val KLD={val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            "best_single_frame.pt",
        )