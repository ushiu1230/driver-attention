"""Measure shuffled, end-to-end CPU video loading without training a model."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset
from dataset.dataset import DG360LiteDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--backends", nargs="+", choices=("opencv", "decord"),
                        default=["opencv", "decord"])
    parser.add_argument("--frames", type=int, choices=(1, 16), default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-threads", type=int, default=1)
    parser.add_argument("--video-cache-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if min(args.batch_size, args.batches, args.video_threads, args.video_cache_size) < 1:
        parser.error("Batch size, measured batches, video threads/cache must be positive")
    if args.warmup < 0 or args.num_workers < 0:
        parser.error("Warmup and workers must be non-negative")
    torch.set_num_threads(1)
    for backend in args.backends:
        start = time.perf_counter()
        dataset = DG360LiteDataset(
            args.root, T=args.frames, target_offset=15 if args.frames == 1 else 0,
            clip_step=16, frame_stride=1, img_size=(224, 1120), storage="video",
            video_backend=backend, video_threads=args.video_threads,
            video_cache_size=args.video_cache_size,
        )
        setup_seconds = time.perf_counter() - start
        count = (args.warmup + args.batches) * args.batch_size
        if len(dataset) < count:
            raise ValueError(f"Need at least {count} samples; dataset has {len(dataset)}")
        # Identical shuffled sample IDs for each backend, independent of worker RNG.
        ids = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:count]
        subset = Subset(dataset, ids.tolist())
        options = {"prefetch_factor": 1} if args.num_workers else {}
        loader = DataLoader(subset, batch_size=args.batch_size,
                            num_workers=args.num_workers, **options)
        start = time.perf_counter()
        iterator = iter(loader)
        timings, startup = [], None
        try:
            for index in range(args.warmup + args.batches):
                batch = next(iterator)
                now = time.perf_counter()
                elapsed = now - start
                if index == 0:
                    startup = elapsed
                if index >= args.warmup:
                    timings.append(elapsed)
                print(f"{backend} batch {index + 1}: {elapsed:.3f}s", flush=True)
                del batch
                start = time.perf_counter()
            # Exhaust the fixed subset so multiprocessing workers exit normally.
            for _ in iterator:
                pass
        finally:
            del iterator, loader
            dataset.video_reader.close()
        print(json.dumps({
            "backend": backend, "setup_seconds": setup_seconds,
            "first_batch_seconds": startup, "measured_batches": len(timings),
            "median_batch_seconds": statistics.median(timings),
            "samples_per_second": args.batch_size * len(timings) / sum(timings),
            "note": "CPU loading only; warm disk/OS cache may favor later backends",
        }), flush=True)


if __name__ == "__main__":
    main()
