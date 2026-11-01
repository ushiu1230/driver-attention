import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.model_factory import EXPERIMENTS
from train import Trainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train DriverGazeNet on the DriverGaze360 saliency dataset."
    )
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    parser.add_argument("--name", default=None, help="Name used for this run's outputs")
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument(
        "--train-path",
        default="./dataset/drivergaze360_subset/dataset/train",
        help="Training split containing recording folders",
    )
    parser.add_argument(
        "--val-path",
        default="./dataset/drivergaze360_subset/dataset/val",
        help="Validation split containing recording folders",
    )
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--w-mse", type=float, default=1.0)
    parser.add_argument("--w-cc", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--clip-step", "--overlap", dest="clip_step", type=int, default=8,
        help="Step between sampled clips; smaller values overlap more",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--skip-first", type=int, default=30)
    parser.add_argument("--img-size", type=int, nargs=2, default=(224, 1120))
    parser.add_argument(
        "--backbone",
        choices=("mobilenet_v3_large", "mobilenet_v3_small", "mobilenet_v2"),
        default="mobilenet_v3_large",
    )
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--no-logs", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.clip_step < 1 or args.frame_stride < 1:
        raise ValueError("--clip-step and --frame-stride must be positive")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be positive")
    if args.w_mse < 0 or args.w_cc < 0 or args.w_mse + args.w_cc == 0:
        raise ValueError("At least one of --w-mse or --w-cc must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError(
            "Learning rate must be positive; other optimizer values non-negative"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_id = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(run_id, args)
    trainer.train()


if __name__ == "__main__":
    main()
