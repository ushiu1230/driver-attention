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
        if args.accumulation_steps < 1:
            raise ValueError("accumulation steps must be positive")
        if args.early_stopping_patience < 1:
            raise ValueError("early-stopping patience must be positive")
        if args.num_workers < 0:
            raise ValueError("num-workers must be non-negative")
        if (
            args.w_sal <= 0
            or args.w_kld < 0
            or args.w_cc < 0
            or args.w_mse < 0
            or args.w_nss < 0
        ):
            raise ValueError("loss weights must be non-negative")

        self.run_id = run_id
        self.args = args
        self.experiment = args.experiment
        self.experiment_config = EXPERIMENTS[self.experiment]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = args.use_amp and self.device.type == "cuda"
        self.epochs_run = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        self.logger = TFLogger(run_id, args.save_dir, args=args)
        if args.use_amp and not self.use_amp:
            print("AMP requested but CUDA is unavailable; using full precision.")
        if self.use_amp:
            print(f"Using AMP on {self.device}")

        self.train_ds = self._make_dataset(args.train_path)
        self.val_ds = self._make_dataset(args.val_path, augment=False)
        self.train_sample_weights = None
        if args.weighted_samples:
            raw_weights = self.train_ds.get_sample_weights(
                cache_dir=Path(args.save_dir) / "dataset_cache"
            )
            if not torch.isfinite(raw_weights).all() or (raw_weights < 0).any():
                raise ValueError("Computed KLD sample weights are invalid")
            lower = torch.quantile(raw_weights, 0.20)
            upper = torch.quantile(raw_weights, 0.80)
            clipped_weights = raw_weights.clamp(min=lower, max=upper)
            mean_weight = clipped_weights.mean()
            if mean_weight <= torch.finfo(clipped_weights.dtype).eps:
                print(
                    "KLD sample weights have no usable variation; "
                    "falling back to uniform training weights."
                )
                clipped_weights.fill_(1.0)
            else:
                clipped_weights /= mean_weight
            self.train_sample_weights = clipped_weights.to(self.device)
            print(
                "KLD sample weighting enabled: "
                f"{len(clipped_weights)} weights, clipped to "
                f"[{lower.item():.6g}, {upper.item():.6g}] and mean-normalized."
            )

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
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        if args.init_weights:
            self.initialize_weights(args.init_weights)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion = SaliencyLoss(
            lambda_kld=args.w_kld,
            lambda_cc=args.w_cc,
            lambda_mse=args.w_mse,
            lambda_nss=args.w_nss,
            lambda_sal=args.w_sal,
        )
        self.metrics = SaliencyMetrics()

        if args.resume:
            if not args.ckpt:
                raise ValueError("--resume requires --ckpt")
            self.load_checkpoint(args.ckpt)

        print(
            f"Train: {len(self.train_ds)} samples | validation: {len(self.val_ds)} "
            f"| experiment={self.experiment} | device={self.device} "
            f"| batch={args.batch_size} x accumulation={args.accumulation_steps} "
            f"(effective batch={args.batch_size * args.accumulation_steps})"
        )

    def initialize_weights(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model"), dict
        ):
            raise ValueError(
                "Initialization checkpoint must contain a model state dict."
            )
        if checkpoint.get("experiment") != self.experiment:
            raise ValueError(
                "Initialization checkpoint experiment "
                f"{checkpoint.get('experiment')!r} does not match the current "
                f"experiment {self.experiment!r}."
            )
        if checkpoint.get("output") != "spatial_softmax":
            raise ValueError(
                "Initialization checkpoint must use spatial-softmax output."
            )
        self.model.load_checkpoint_weights(checkpoint["model"])
        print(
            f"Initialized model weights from {path}; optimizer and training "
            "progress start fresh."
        )

    def _make_dataset(self, root, augment=False):
        return DG360LiteDataset(
            root=root,
            T=self.experiment_config["input_frames"],
            frame_stride=self.args.frame_stride,
            clip_step=self.args.clip_step,
            img_size=tuple(self.args.img_size),
            skip_first=self.args.skip_first,
            target_offset=self.experiment_config["target_offset"],
            augment=augment,
            resize_mode=self.args.resize_mode,
        )

    def _predict_saliency(self, rgb, mask):
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
        ):
            prediction = self.model(rgb)

        valid = mask > 0
        if not valid.flatten(1).any(dim=1).all():
            raise ValueError("A batch item has no valid image pixels")
        logits = prediction.float().flatten(start_dim=2)
        logits = logits.masked_fill(~valid.flatten(start_dim=2), float("-inf"))
        probabilities = torch.softmax(logits, dim=-1).view_as(prediction)
        return probabilities.masked_fill(~valid, 0.0)

    def _valid_samples(self, batch):
        rgb = batch["rgb"].to(self.device, non_blocking=True)
        target = batch["sal"].to(self.device, non_blocking=True).float()
        mask = batch["mask"].to(self.device, non_blocking=True) > 0
        fixation = batch["fixation"].to(self.device, non_blocking=True) > 0
        sample_idx = batch["sample_idx"].to(self.device, non_blocking=True)
        target = target.masked_fill(~mask, 0)
        target_mass = target.flatten(1).sum(dim=1, keepdim=True)
        valid = target_mass.squeeze(1) > 0
        target = target / target_mass.clamp_min(torch.finfo(target.dtype).tiny).view(
            -1, 1, 1, 1
        )
        missing_fixation_count = 0
        if self.args.w_nss:
            has_fixation = fixation.flatten(1).any(dim=1)
            missing_fixation_count = int((valid & ~has_fixation).sum().item())
            valid &= has_fixation
        return (
            rgb[valid],
            target[valid],
            mask[valid],
            fixation[valid],
            sample_idx[valid],
            missing_fixation_count,
        )

    def _optimizer_step(self, accumulated_micro_batches):
        self.scaler.unscale_(self.optimizer)
        if accumulated_micro_batches < self.args.accumulation_steps:
            correction = self.args.accumulation_steps / accumulated_micro_batches
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def _run_epoch(self, epoch, training):
        self.model.train(training)
        loader = self.train_dl if training else self.val_dl
        phase = "train" if training else "val"
        loss_total = 0.0
        metric_totals = {
            "KLD": 0.0,
            "CC": 0.0,
            "SIM": 0.0,
            "NSS": 0.0,
        }
        metric_counts = {name: 0 for name in metric_totals}
        sample_count = 0
        missing_fixation_count = 0
        accumulated_micro_batches = 0
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch + 1} | {phase}",
            disable=not self.args.progress,
        )

        for step, batch in enumerate(progress):
            (
                rgb,
                target,
                mask,
                fixation,
                sample_idx,
                missing_fixations,
            ) = self._valid_samples(batch)
            missing_fixation_count += missing_fixations
            if rgb.shape[0] == 0:
                continue

            if training and accumulated_micro_batches == 0:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                prediction = self._predict_saliency(rgb, mask)
                sample_weights = (
                    self.train_sample_weights[sample_idx]
                    if training and self.train_sample_weights is not None
                    else None
                )
                loss = self.criterion(
                    prediction,
                    target,
                    mask,
                    fixation=fixation,
                    sample_weights=sample_weights,
                )
                if training:
                    self.scaler.scale(
                        loss / self.args.accumulation_steps
                    ).backward()
                    accumulated_micro_batches += 1
                    if accumulated_micro_batches == self.args.accumulation_steps:
                        self._optimizer_step(accumulated_micro_batches)
                        accumulated_micro_batches = 0

            batch_count = rgb.shape[0]
            loss_total += loss.detach().item() * batch_count
            batch_metrics = self.metrics(
                prediction.detach(), target, mask, fixation=fixation
            )
            for name, values in batch_metrics.items():
                finite = torch.isfinite(values)
                metric_totals[name] += values[finite].sum().item()
                metric_counts[name] += int(finite.sum().item())
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

        if training and accumulated_micro_batches:
            self._optimizer_step(accumulated_micro_batches)

        if sample_count == 0:
            if missing_fixation_count:
                raise RuntimeError(
                    f"No samples with valid target-frame fixation were found in "
                    f"{phase} data; excluded {missing_fixation_count} samples."
                )
            raise RuntimeError(
                f"No non-empty saliency maps were found in {phase} data"
            )
        if missing_fixation_count:
            print(
                f"{phase}: skipped {missing_fixation_count} samples without "
                "a valid target-frame fixation for NSS"
            )

        mean_loss = loss_total / sample_count
        metrics = {
            name: (
                metric_totals[name] / metric_counts[name]
                if metric_counts[name]
                else float("nan")
            )
            for name in metric_totals
        }
        if not metric_counts["NSS"]:
            print(f"{phase}: NSS unavailable; no samples contain valid fixations")
        return mean_loss, metrics

    def save_checkpoint(self, epoch, val_loss, val_metrics, filename):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "experiment": self.experiment,
            "output": "spatial_softmax",
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
        saved_args = checkpoint.get("args")
        if not isinstance(saved_args, dict):
            raise ValueError(
                "Checkpoint does not contain training arguments, so its loss "
                "configuration cannot be verified. Start a new run instead."
            )
        if any(name not in saved_args for name in ("w_cc", "w_mse")):
            raise ValueError(
                "Checkpoint is missing loss weights, so its objective cannot "
                "be verified. Start a new run instead."
            )
        saved_loss_weights = {
            "w_sal": saved_args.get("w_sal", 1.0),
            "w_kld": saved_args.get("w_kld", 0.0),
            "w_cc": saved_args["w_cc"],
            "w_mse": saved_args["w_mse"],
            "w_nss": saved_args.get("w_nss", 0.0),
        }
        changed_weights = [
            f"{name}: checkpoint={saved_weight}, current={getattr(self.args, name)}"
            for name, saved_weight in saved_loss_weights.items()
            if float(saved_weight) != float(getattr(self.args, name))
        ]
        if changed_weights:
            raise ValueError(
                "Cannot resume with different loss weights ("
                + ", ".join(changed_weights)
                + "). Use a new run name to train with the new objective."
            )
        training_settings = (
            "clip_step",
            "frame_stride",
            "accumulation_steps",
            "weighted_samples",
            "img_size",
            "resize_mode",
            "lr",
            "weight_decay",
            "dropout",
            "augment",
        )
        legacy_setting_defaults = {
            "accumulation_steps": 1,
            "weighted_samples": False,
            "resize_mode": "pad",
            "dropout": 0.0,
            "augment": False,
        }
        changed_settings = [
            f"{name}: checkpoint="
            f"{saved_args.get(name, legacy_setting_defaults.get(name, '<missing>'))}, "
            f"current={getattr(self.args, name)}"
            for name in training_settings
            if saved_args.get(name, legacy_setting_defaults.get(name, "<missing>"))
            != getattr(self.args, name)
        ]
        if changed_settings:
            raise ValueError(
                "Cannot resume with different or unverifiable training settings ("
                + ", ".join(changed_settings)
                + "). Use the original settings or start a new run."
            )
        saved_early_stopping_patience = saved_args.get(
            "early_stopping_patience"
        )
        if saved_early_stopping_patience is None:
            raise ValueError(
                "Checkpoint is missing early-stopping patience, so its "
                "training configuration cannot be verified."
            )
        if self.args.early_stopping_patience < saved_early_stopping_patience:
            raise ValueError(
                "Cannot resume with a lower early-stopping patience than the "
                f"checkpoint ({self.args.early_stopping_patience} < "
                f"{saved_early_stopping_patience})."
            )
        if self.args.early_stopping_patience > saved_early_stopping_patience:
            print(
                "Extending early-stopping patience from "
                f"{saved_early_stopping_patience} to "
                f"{self.args.early_stopping_patience} epochs."
            )
        if checkpoint.get("output", "sigmoid") != "spatial_softmax":
            raise ValueError(
                "Cannot resume a checkpoint trained with a different output "
                "activation. Start a new run."
            )

        self.model.load_checkpoint_weights(checkpoint["model"])
        optimizer_state = checkpoint["optimizer"]
        temporal_pe_key = "temporal_pe.temporal_pe"
        if temporal_pe_key not in checkpoint["model"]:
            optimizer_state = optimizer_state.copy()
            optimizer_state["param_groups"] = [
                {**group, "params": list(group["params"])}
                for group in optimizer_state["param_groups"]
            ]
            current_groups = self.optimizer.state_dict()["param_groups"]
            if len(optimizer_state["param_groups"]) != len(current_groups):
                raise ValueError(
                    "Cannot resume this pre-PE checkpoint because its optimizer "
                    "parameter groups do not match the current model."
                )
            group_deltas = [
                len(current["params"]) - len(saved["params"])
                for saved, current in zip(
                    optimizer_state["param_groups"], current_groups
                )
            ]
            if group_deltas.count(1) != 1 or any(
                delta not in (0, 1) for delta in group_deltas
            ):
                raise ValueError(
                    "Cannot resume this pre-PE checkpoint because its optimizer "
                    "parameter layout does not match the current model."
                )
            pe_group_index = group_deltas.index(1)
            if (
                self.optimizer.param_groups[pe_group_index]["params"][
                    sum(1 for _ in self.model.backbone.parameters())
                ]
                is not self.model.temporal_pe.temporal_pe
            ):
                raise ValueError(
                    "Cannot resume this pre-PE checkpoint because the temporal "
                    "PE parameter ordering is incompatible."
                )
            saved_ids = [
                parameter_id
                for group in optimizer_state["param_groups"]
                for parameter_id in group["params"]
            ]
            optimizer_state["param_groups"][pe_group_index]["params"].insert(
                sum(1 for _ in self.model.backbone.parameters()),
                max(saved_ids, default=-1) + 1,
            )
            print(
                "Checkpoint predates temporal PE; resuming with zero PE and "
                "preserving the saved optimizer state for existing parameters."
            )
        checkpoint_lrs = [
            float(group["lr"])
            for group in optimizer_state["param_groups"]
            if "lr" in group
        ]
        if len(checkpoint_lrs) != len(optimizer_state["param_groups"]) or any(
            saved_lr != float(self.args.lr) for saved_lr in checkpoint_lrs
        ):
            raise ValueError(
                "Cannot resume a checkpoint whose optimizer learning rate is "
                "not the configured fixed learning rate. Start a new run."
            )
        self.optimizer.load_state_dict(optimizer_state)
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.epochs_run = int(checkpoint["epoch"]) + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )

    def train(self):
        try:
            for epoch in range(self.epochs_run, self.args.num_epochs):
                train_loss, _ = self._run_epoch(epoch, training=True)
                val_loss, val_metrics = self._run_epoch(epoch, training=False)
                improved = val_loss < self.best_val_loss
                if improved:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1:02d}: train loss={train_loss:.6f} | "
                    f"val loss={val_loss:.6f} KLD={val_metrics['KLD']:.4f} "
                    f"CC={val_metrics['CC']:.4f} "
                    f"SIM={val_metrics['SIM']:.4f} "
                    f"NSS={val_metrics['NSS']:.4f} "
                    f"lr={current_lr:.2e}"
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
                if (
                    self.epochs_without_improvement
                    >= self.args.early_stopping_patience
                ):
                    print(
                        "Early stopping: validation loss did not improve for "
                        f"{self.epochs_without_improvement} epochs."
                    )
                    break
        finally:
            self.logger.close()


__all__ = ["Trainer"]
