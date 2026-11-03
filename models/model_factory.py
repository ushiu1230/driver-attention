"""Model and temporal experiment factory for DriverGazeNet."""

from typing import TypedDict

from torch import nn

from .drivergaze_net import DriverGazeNet


class ExperimentConfig(TypedDict):
    input_frames: int
    target_offset: int
    use_tsm: bool
    temporal_mode: str
    checkpoint_name: str


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline": {
        "input_frames": 1,
        "target_offset": 15,
        "use_tsm": False,
        "temporal_mode": "last",
        "checkpoint_name": "best_single_frame",
    },
    "tsm": {
        "input_frames": 16,
        "target_offset": 0,
        "use_tsm": True,
        "temporal_mode": "last",
        "checkpoint_name": "best_tsm",
    },
    "temporal_mean_pool": {
        "input_frames": 16,
        "target_offset": 0,
        "use_tsm": False,
        "temporal_mode": "mean",
        "checkpoint_name": "best_temporal_mean_pool",
    },
    "temporal_attn_pool": {
        "input_frames": 16,
        "target_offset": 0,
        "use_tsm": False,
        "temporal_mode": "attn",
        "checkpoint_name": "best_temporal_attn_pool",
    },
    "tsm_attn_pool": {
        "input_frames": 16,
        "target_offset": 0,
        "use_tsm": True,
        "temporal_mode": "attn",
        "checkpoint_name": "best_tsm_attn_pool",
    },
}


def build_model(args) -> nn.Module:
    """Build the configured DriverGazeNet experiment from CLI arguments."""
    try:
        config = EXPERIMENTS[args.experiment]
    except KeyError as error:
        raise ValueError(f"Unknown experiment: {args.experiment}") from error

    model = DriverGazeNet(
        backbone_name=args.backbone,
        num_segments=config["input_frames"],
        temporal_mode=config["temporal_mode"],
        use_tsm=config["use_tsm"],
        pretrained=not args.no_pretrained,
        decoder_width=args.decoder_width,
        dropout=args.dropout,
        apply_sigmoid=False,
    )
    print(f"Using DriverGazeNet experiment: {args.experiment}")
    print_model_info(model)
    return model


def print_model_info(model: nn.Module) -> None:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    model_size_mb = total_parameters * 4 / (1024 ** 2)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Model size (FP32): {model_size_mb:.2f} MB")


__all__ = ["EXPERIMENTS", "ExperimentConfig", "build_model", "print_model_info"]
