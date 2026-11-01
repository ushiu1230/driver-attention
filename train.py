from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import DG360LiteDataset
from models.model_factory import EXPERIMENTS, build_model
from utils.losses import SaliencyLoss
from utils.metrics import SaliencyMetrics
from utils.tflogger import TFLogger


class Trainer:
    """Train DriverGazeNet with the DriverGaze360 dataset."""

    def __init__(self, run_id, args):
        if args.experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {args.experiment}")
        if args.batch_size < 1 or args.num_epochs < 1:
            raise ValueError("batch size and number of epochs must be positive")
        if args.num_workers < 0:
            raise ValueError("num-workers must be non-negative")
        if args.w_mse < 0 or args.w_cc < 0:
            raise ValueError("loss weights must be non-negative")

        self.run_id = run_id
        self.args = args
        self.experiment = args.experiment
        self.experiment_config = EXPERIMENTS[self.experiment]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = args.use_amp and self.device.type == "cuda"
        self.epochs_run = 0
        self.best_val_loss = float("inf")

        self.logger = TFLogger(run_id, args.save_dir, args=args)
        if args.use_amp and not self.use_amp:
            print("AMP requested but CUDA is unavailable; using full precision.")
        if self.use_amp:
            print(f"Using AMP on {self.device}")

        self.train_ds = self._make_dataset(args.train_path)
        self.val_ds = self._make_dataset(args.val_path)
        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        self.model = build_model(args).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=args.lr_patience
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion = SaliencyLoss(
            lambda_mse=args.w_mse, lambda_cc=args.w_cc
        )
        self.metrics = SaliencyMetrics()

        if args.resume:
            if not args.ckpt:
                raise ValueError("--resume requires --ckpt")
            self.load_checkpoint(args.ckpt)

        print(
            f"Train: {len(self.train_ds)} samples | validation: {len(self.val_ds)} "
            f"| experiment={self.experiment} | device={self.device}"
        )

    def _make_dataset(self, root):
        return DG360LiteDataset(
            root=root,
            T=self.experiment_config["input_frames"],
            frame_stride=self.args.frame_stride,
            clip_step=self.args.clip_step,
            img_size=tuple(self.args.img_size),
            skip_first=self.args.skip_first,
            target_offset=self.experiment_config["target_offset"],
        )

    def _predict_saliency(self, rgb, mask):
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
        ):
            prediction = self.model(rgb)

        valid = mask > 0
        if not valid.flatten(1).any(dim=1).all():
            raise ValueError("A batch item has no valid image pixels")
        return prediction.float().masked_fill(~valid, 0.0)

    def _valid_samples(self, batch):
        rgb = batch["rgb"].to(self.device, non_blocking=True)
        target = batch["sal"].to(self.device, non_blocking=True).float()
        mask = batch["mask"].to(self.device, non_blocking=True) > 0
        target = target.masked_fill(~mask, 0)
        valid = target.flatten(1).sum(dim=1) > 0
        return rgb[valid], target[valid], mask[valid]

    def _run_epoch(self, epoch, training):
        self.model.train(training)
        loader = self.train_dl if training else self.val_dl
        phase = "train" if training else "val"
        loss_total = 0.0
        metric_totals = {"KLD": 0.0, "CC": 0.0, "SIM": 0.0, "peak_px": 0.0}
        sample_count = 0
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch + 1} | {phase}",
            disable=not self.args.progress,
        )

        for step, batch in enumerate(progress):
            rgb, target, mask = self._valid_samples(batch)
            if rgb.shape[0] == 0:
                continue

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                prediction = self._predict_saliency(rgb, mask)
                loss = self.criterion(prediction, target, mask)
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            batch_count = rgb.shape[0]
            loss_total += loss.detach().item() * batch_count
            batch_metrics = self.metrics(prediction.detach(), target, mask)
            for name, values in batch_metrics.items():
                metric_totals[name] += values.sum().item()
            sample_count += batch_count

            if step % self.args.log_interval == 0:
                self.logger.log(
                    {"loss": loss.detach()}, phase, step=epoch * len(loader) + step
                )
                if step == 0:
                    self.logger.log_images(
                        rgb, prediction, target, phase=f"{phase}/sal",
                        step=epoch,
                    )

        if sample_count == 0:
            raise RuntimeError(f"No non-empty saliency maps were found in {phase} data")

        mean_loss = loss_total / sample_count
        metrics = {
            name: value / sample_count for name, value in metric_totals.items()
        }
        return mean_loss, metrics

    def save_checkpoint(self, epoch, val_loss, val_metrics, filename):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "experiment": self.experiment,
            "output": "sigmoid",
            "decoder_norm": "bn",
            "args": vars(self.args),
        }
        out_path = Path(self.logger.save_dir) / filename
        torch.save(checkpoint, out_path)
        print(f"Saved checkpoint: {out_path}")

    def load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.epochs_run = int(checkpoint["epoch"]) + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    def train(self):
        try:
            for epoch in range(self.epochs_run, self.args.num_epochs):
                train_loss, _ = self._run_epoch(epoch, training=True)
                val_loss, val_metrics = self._run_epoch(epoch, training=False)
                self.scheduler.step(val_loss)
                improved = val_loss < self.best_val_loss
                if improved:
                    self.best_val_loss = val_loss

                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1:02d}: train loss={train_loss:.6f} | "
                    f"val loss={val_loss:.6f} KLD={val_metrics['KLD']:.4f} "
                    f"CC={val_metrics['CC']:.4f} "
                    f"SIM={val_metrics['SIM']:.4f} "
                    f"peak={val_metrics['peak_px']:.2f} px | lr={current_lr:.2e}"
                )
                self.logger.log_epoch(
                    train_loss, val_loss, val_metrics, epoch + 1
                )
                self.save_checkpoint(
                    epoch, val_loss, val_metrics, "last.pt"
                )
                if improved:
                    self.save_checkpoint(
                        epoch, val_loss, val_metrics, "best.pt"
                    )
                self.logger.flush()
        finally:
            self.logger.close()


__all__ = ["Trainer"]
