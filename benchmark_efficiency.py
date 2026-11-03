"""Measure model size and single-input inference efficiency for a checkpoint."""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

from infer import load_model
from models.model_factory import EXPERIMENTS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--precision", choices=("fp32", "fp16"), default="fp32",
        help="Use CUDA autocast for fp16; FLOPs are counted in the model's FP32 graph",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.img_size and any(dimension < 1 for dimension in args.img_size):
        parser.error("--img-size dimensions must be positive")
    return args


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    return torch.device(requested)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device, precision):
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def main(argv=None):
    args = parse_args(argv)
    device = select_device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("--precision fp16 requires a CUDA device.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, checkpoint, experiment, saved, _ = load_model(
        checkpoint_path, torch.device("cpu")
    )
    model = model.to(device).eval()
    experiment_config = EXPERIMENTS[experiment]
    saved_size = saved.get("img_size", (224, 1120))
    height, width = args.img_size or saved_size
    if height < 1 or width < 1:
        raise ValueError(f"Invalid input size: {(height, width)}")

    input_shape = (
        args.batch_size,
        experiment_config["input_frames"],
        3,
        height,
        width,
    )
    torch.manual_seed(0)
    inputs = torch.randn(input_shape, device=device)
    del checkpoint

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    flop_analysis = FlopCountAnalysis(model, (inputs,))
    total_flops = flop_analysis.total()
    flops_by_operator = {
        str(operator): int(count)
        for operator, count in sorted(flop_analysis.by_operator().items())
    }
    unsupported_operators = {
        str(operator): int(count)
        for operator, count in sorted(flop_analysis.unsupported_ops().items())
    }

    with torch.inference_mode():
        for _ in range(args.warmup):
            with autocast_context(device, args.precision):
                model(inputs)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        durations_ms = []
        for _ in range(args.iterations):
            synchronize(device)
            started = time.perf_counter()
            with autocast_context(device, args.precision):
                model(inputs)
            synchronize(device)
            durations_ms.append((time.perf_counter() - started) * 1000)

    result = {
        "checkpoint": str(checkpoint_path),
        "experiment": experiment,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or platform.machine()
        ),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_shape": list(input_shape),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "torch_cpu_threads": torch.get_num_threads(),
        "parameters_total": total_parameters,
        "parameters_trainable": trainable_parameters,
        "fvcore_flops_per_batch": int(total_flops),
        "fvcore_flops_per_sample": int(total_flops / args.batch_size),
        "fvcore_flops_complete": not bool(unsupported_operators),
        "fvcore_flops_by_operator": flops_by_operator,
        "fvcore_unsupported_operators": unsupported_operators,
        "latency_ms_per_batch_mean": statistics.mean(durations_ms),
        "latency_ms_per_batch_median": statistics.median(durations_ms),
        "latency_ms_per_batch_p95": float(
            torch.tensor(durations_ms).quantile(0.95).item()
        ),
        "peak_cuda_memory_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else None
        ),
        "peak_cuda_memory_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            if device.type == "cuda"
            else None
        ),
    }

    output = json.dumps(result, indent=2)
    print(output)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"Saved benchmark JSON: {output_path}")


if __name__ == "__main__":
    main()
