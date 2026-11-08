import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.model_factory import EXPERIMENTS
from train import Trainer


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--hparams", default=str(Path(__file__).parent / "configs" / "hparams.json"))
    config_args, _ = config_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(
        description="Train DriverGazeNet on the DriverGaze360 saliency dataset.",
        parents=[config_parser],
    )
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    parser.add_argument("--name", default=None, help="Name used for this run's outputs")
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument(
        "--train-path",
        help="Root of the training recordings (e.g. downloaded dataset/train)",
    )
    parser.add_argument(
        "--val-path",
        help="Root of the validation recordings (e.g. downloaded dataset/val)",
    )
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--w-sal", type=float, default=1.0)
    parser.add_argument("--w-kld", type=float, default=1.0)
    parser.add_argument("--w-cc", type=float, default=1.0)
    parser.add_argument("--w-mse", type=float, default=0.0)
    parser.add_argument("--w-nss", type=float, default=0.0)
    parser.add_argument(
        "--loss-profile", choices=("project", "drivergaze360"), default="project",
        help="Use the project objective or the saliency equations in the local paper source",
    )
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1,
                        help="Batches prepared per worker when num-workers > 0")
    parser.add_argument(
        "--clip-step", "--overlap", dest="clip_step", type=int, default=16,
        help="Step between sampled clips; smaller values overlap more",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--skip-first", type=int, default=30)
    parser.add_argument(
        "--paper-protocol", action=argparse.BooleanOptionalAction,
        help="Use source end-exclusive clip sampling and clip-plus-future fixation filtering",
    )
    parser.add_argument("--img-size", type=int, nargs=2, default=(224, 1120))
    parser.add_argument("--gaze-source-size", type=int, nargs=2, default=None,
                        help="CSV gaze coordinate canvas (H W); paper default is 720 6400")
    parser.add_argument(
        "--resize-mode",
        choices=("stretch", "pad"),
        default="stretch",
        help="Directly resize to target dimensions or preserve aspect ratio and pad",
    )
    parser.add_argument(
        "--backbone",
        choices=("mobilenet_v3_large", "mobilenet_v3_small", "mobilenet_v2"),
        default="mobilenet_v3_large",
    )
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable horizontal flip and mild color jitter on training clips",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze backbone weights and BatchNorm statistics; train pooling and decoder",
    )
    parser.add_argument(
        "--freeze-backbone-stages", type=int, default=0,
        help="Freeze this many early backbone feature blocks and their BatchNorm statistics",
    )
    parser.add_argument(
        "--backbone-lr-scale", type=float, default=1.0,
        help="Backbone LR multiplier; 0.1 gives discriminative LR relative to the head",
    )
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--weighted-samples",
        action="store_true",
        help="Weight training samples by clipped KL divergence from mean train saliency",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate init-weights on validation and exit")
    parser.add_argument("--ckpt", default="")
    parser.add_argument(
        "--init-weights",
        default="",
        help="Initialize model weights only from a compatible checkpoint",
    )
    parser.add_argument("--no-logs", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    try:
        config = json.loads(Path(config_args.hparams).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("hparams must be a JSON object")
        actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "hparams")}
        for key, value in config.items():
            if key not in actions:
                raise ValueError(f"Unknown hparams key: {key}")
            action = actions[key]
            if action.nargs == 0:
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be boolean")
                continue
            values = value if isinstance(value, list) else [value]
            if isinstance(action.nargs, int) and len(values) != action.nargs:
                raise ValueError(f"{key} requires {action.nargs} values")
            if action.nargs is None and isinstance(value, list):
                raise ValueError(f"{key} requires one value")
            for item in values:
                expected = action.type or str
                if isinstance(item, bool) or not isinstance(item, expected if expected != float else (int, float)):
                    raise ValueError(f"Invalid type for {key}")
                if action.choices is not None and item not in action.choices:
                    raise ValueError(f"Invalid choice for {key}: {item}")
        parser.set_defaults(**config)
    except (OSError, ValueError) as error:
        parser.error(f"Cannot load hparams: {error}")
    args = parser.parse_args(argv)
    if not args.train_path or not args.val_path:
        parser.error("train-path and val-path must be set in hparams or CLI")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    if args.freeze_backbone_stages < 0 or args.backbone_lr_scale <= 0:
        raise ValueError("freeze stages must be non-negative and backbone LR scale positive")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be positive")
    if args.resume and args.init_weights:
        raise ValueError("--resume and --init-weights cannot be used together")
    if args.clip_step < 1 or args.frame_stride < 1:
        raise ValueError("--clip-step and --frame-stride must be positive")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be positive")
    if (
        args.w_sal <= 0
        or args.w_kld < 0
        or args.w_cc < 0
        or args.w_mse < 0
        or args.w_nss < 0
        or args.w_kld + args.w_cc + args.w_mse + args.w_nss == 0
    ):
        raise ValueError(
            "Loss weights must be valid, w-sal must be positive, and at least "
            "one objective weight must be positive"
        )
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError(
            "Learning rate must be positive; other optimizer values non-negative"
        )
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_id = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(run_id, args)
    if args.eval_only:
        if not args.init_weights:
            raise ValueError("--eval-only requires --init-weights")
        val_loss, metrics = trainer._run_epoch(0, training=False)
        result = {
            "run": run_id,
            "checkpoint": args.init_weights,
            "val_loss": val_loss,
            "metrics": metrics,
            "sample_count": len(trainer.val_ds),
            "paper_protocol": args.paper_protocol,
            "loss_profile": args.loss_profile,
        }
        output = Path(args.save_dir) / run_id / "evaluation.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        trainer.logger.close()
        return
    trainer.train()


if __name__ == "__main__":
    main()
