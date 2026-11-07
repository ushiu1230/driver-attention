"""Copy compressed recordings from mounted Drive to local Colab storage.

Never extract frames or remove source files. Interrupted copies are retried
one file at a time; completed files are reused using size/mtime receipts.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time


FILES = ("rgb.mp4", "saliency.mp4", "sim_gaze_df.csv")
GIB = 1024 ** 3


def inventory(source):
    records = []
    for directory, dirs, files in os.walk(source):
        # Do not traverse the potentially millions of already extracted JPGs.
        dirs[:] = sorted(d for d in dirs if d not in {"rgb", "saliency", "DT", "IS"}
                         and not d.startswith("."))
        if "sim_gaze_df.csv" not in files:
            continue
        missing = set(FILES) - set(files)
        if missing:
            raise FileNotFoundError(f"Incomplete recording {directory}: {sorted(missing)}")
        records.extend(Path(directory) / name for name in FILES)
    if not records:
        raise ValueError(f"No compressed recordings found under {source}")
    return records


def stage(source, destination, max_gib=160, reserve_gib=20, dry_run=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-nested directories")
    if max_gib <= 0 or reserve_gib < 0:
        raise ValueError("max-gib must be positive and reserve-gib non-negative")
    files = inventory(source)
    manifest_path = destination / ".stage_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and manifest.get("source") != str(source):
        raise ValueError("Destination belongs to another source; choose a fresh directory")
    receipts = manifest.get("files", {})
    entries, pending = {}, []
    for path in files:
        stat = path.stat()
        key = path.relative_to(source).as_posix()
        signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        entries[key] = signature
        target = destination / key
        if (receipts.get(key) == signature and target.is_file()
                and target.stat().st_size == stat.st_size):
            continue
        pending.append((path, target, key, signature))
    # A removed recording must not remain visible to the training loader.
    if set(receipts) - set(entries):
        raise ValueError("Source inventory changed; use a fresh destination to avoid stale recordings")
    total = sum(entry["size"] for entry in entries.values())
    needed = sum(entry[3]["size"] for entry in pending)
    probe = destination
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    print(f"Dataset: {total / GIB:.2f} GiB; pending: {needed / GIB:.2f} GiB; "
          f"free: {free / GIB:.2f} GiB", flush=True)
    if total > max_gib * GIB or needed + reserve_gib * GIB > free:
        raise RuntimeError("Insufficient staging budget/free space; no files copied")
    if dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    for index, (path, target, key, signature) in enumerate(pending, 1):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".stage-partial")
        start = time.perf_counter()
        with path.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        stat = path.stat()
        if (temporary.stat().st_size != signature["size"] or
                stat.st_size != signature["size"] or stat.st_mtime_ns != signature["mtime_ns"]):
            raise RuntimeError(f"Source changed or incomplete copy: {path}")
        temporary.replace(target)
        receipts[key] = signature
        receipt_tmp = manifest_path.with_suffix(".tmp")
        receipt_tmp.write_text(json.dumps({"source": str(source), "files": receipts}, indent=2))
        receipt_tmp.replace(manifest_path)
        speed = signature["size"] / 1e6 / max(time.perf_counter() - start, 1e-6)
        print(f"[{index}/{len(pending)}] {key}: {speed:.1f} MB/s", flush=True)
    print(f"Ready: {destination} (compressed video; no JPG extraction)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=Path("/content/drivergaze360"))
    parser.add_argument("--max-gib", type=float, default=160)
    parser.add_argument("--reserve-gib", type=float, default=20)
    parser.add_argument("--dry-run", action="store_true")
    stage(**vars(parser.parse_args()))
