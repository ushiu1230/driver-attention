"""Smoke-check one source-protocol batch from each split before long runs."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dataset.dataset import DG360LiteDataset
from models.model_factory import build_model
from utils.losses import SaliencyLoss


def inspect_split(root, model, criterion, device, T=16, max_samples=128):
    dataset = DG360LiteDataset(
        root=root, T=T, frame_stride=1, clip_step=8, img_size=(224, 1120),
        skip_first=30, target_offset=0, augment=False, resize_mode="stretch",
        paper_protocol=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for sample_index, batch in enumerate(loader):
        if sample_index >= max_samples:
            break
        target = batch["sal"].to(device).float()
        mask = batch["mask"].to(device) > 0
        fixation = batch["fixation"].to(device) > 0
        target = target.masked_fill(~mask, 0)
        target_mass = target.flatten(1).sum(1, keepdim=True)
        if target_mass.item() <= 0 or not fixation.any():
            continue
        target = target / target_mass.view(-1, 1, 1, 1)
        rgb = batch["rgb"].to(device)
        model.zero_grad(set_to_none=True)
        output = model(rgb)
        prediction = output.float().masked_fill(~mask, 0)
        loss = criterion(prediction, target, mask, fixation=fixation)
        loss.backward()
        if not torch.isfinite(loss) or not all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError(f"Non-finite forward/backward for {root}")
        recording_index, start = dataset.samples[sample_index]
        target_position = start + T - 1
        result = {
            "split": str(root), "record_count": len(dataset.data_paths),
            "candidate_sample_count": len(dataset), "sample_index": sample_index,
            "record": str(dataset.data_paths[recording_index]),
            "start_position": start,
            "input_frame_ids": [
                dataset.frame_ids[recording_index][start + i] for i in range(T)
            ],
            "target_frame_id": dataset.frame_ids[recording_index][target_position],
            "fixation_pixels": int(fixation.sum().item()),
            "valid_pixels": int(mask.sum().item()), "loss": float(loss.detach()),
        }
        return result
    raise RuntimeError(f"No valid target/fixation in first {max_samples} samples of {root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = SimpleNamespace(
        experiment="tsm_attn_pool", backbone="mobilenet_v3_large",
        no_pretrained=True, decoder_width=128, dropout=0.0,
    )
    model = build_model(model_args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_checkpoint_weights(checkpoint["model"])
    model.train()
    criterion = SaliencyLoss(lambda_kld=1, lambda_cc=1, lambda_mse=0,
                             lambda_nss=0, profile="drivergaze360")
    results = [
        inspect_split(args.train_path, model, criterion, device),
        inspect_split(args.val_path, model, criterion, device),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"device": str(device), "results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
