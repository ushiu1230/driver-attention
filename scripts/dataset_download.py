import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

# Configuration
REPO_ID = "dfki-av/drivergaze360"
REPO_TYPE = "dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = PROJECT_ROOT / "data" / "raw" / "drivergaze360" / "trainval"
TEST_SAVE_DIR = PROJECT_ROOT / "data" / "raw" / "drivergaze360" / "test"
TRAIN_FRACTION = 1.0
VAL_FRACTION = 1.0
TEST_DRIVER_FRACTION = 0.10
OTHER_SPLIT_FRACTION = 0.20
SEED = 42

KEEP_FILES = {"rgb.mp4", "saliency.mp4", "sim_gaze_df.csv"}

parser = argparse.ArgumentParser(
    description="Download a reproducible DriverGaze360 subset."
)
parser.add_argument(
    "--train-fraction",
    type=float,
    default=TRAIN_FRACTION,
    help="Fraction of eligible training recordings to download (default: 1.0).",
)
parser.add_argument(
    "--val-fraction",
    type=float,
    default=VAL_FRACTION,
    help=(
        "Fraction of the official validation split to download "
        "(default: 1.0, the complete split)."
    ),
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=SAVE_DIR,
    help=f"Directory for training and validation data (default: {SAVE_DIR}).",
)
parser.add_argument(
    "--test-output-dir",
    type=Path,
    default=TEST_SAVE_DIR,
    help=f"Separate directory for held-out test drivers (default: {TEST_SAVE_DIR}).",
)
parser.add_argument(
    "--test-driver-fraction",
    type=float,
    default=TEST_DRIVER_FRACTION,
    help="Fraction of original training drivers held out for test (default: 0.10).",
)
parser.add_argument(
    "--val-only",
    action="store_true",
    help="Download only the selected official validation recordings.",
)
args = parser.parse_args()
if not 0 < args.train_fraction <= 1:
    parser.error("--train-fraction must be greater than 0 and at most 1")
if not 0 < args.val_fraction <= 1:
    parser.error("--val-fraction must be greater than 0 and at most 1")
if not 0 < args.test_driver_fraction < 1:
    parser.error("--test-driver-fraction must be greater than 0 and less than 1")
save_dir = args.output_dir
test_save_dir = args.test_output_dir

api = HfApi()

print("Scanning dataset file list...")
all_files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

# Group available files by recording and original split
recordings = defaultdict(dict)

for path in all_files:
    parts = path.split("/")

    # Expected structure:
    # dataset/train/C001/001/001-1/rgb.mp4
    if (
        len(parts) >= 6
        and parts[0] == "dataset"
        and parts[-1] in KEEP_FILES
    ):
        split = parts[1]
        recording_dir = "/".join(parts[:-1])
        recordings[(split, recording_dir)][parts[-1]] = path

# Keep only recordings containing all three requested files
eligible = defaultdict(list)

for (split, recording_dir), files in recordings.items():
    if KEEP_FILES.issubset(files.keys()):
        eligible[split].append((recording_dir, files))

# Keep non-training split sampling deterministic. Download the complete
# official validation split by default; other source splits remain sampled.
legacy_rng = random.Random(SEED)
selected_non_train = {}
for split in sorted(eligible):
    candidates = sorted(eligible[split], key=lambda x: x[0])
    fraction = args.val_fraction if split == "val" else OTHER_SPLIT_FRACTION
    n = max(1, math.ceil(len(candidates) * fraction))
    n = min(n, len(candidates))
    chosen = (
        candidates
        if fraction == 1.0
        else sorted(legacy_rng.sample(candidates, n), key=lambda x: x[0])
    )
    selected_non_train[split] = chosen

if args.val_only:
    validation_recordings = selected_non_train.get("val")
    if not validation_recordings:
        parser.error("The dataset release contains no eligible validation recordings")
    validation_files = [
        files[name]
        for _, files in validation_recordings
        for name in KEEP_FILES
    ]
    print(
        f"val: {len(validation_recordings)}/{len(eligible['val'])} recordings "
        "selected"
    )
    print(f"Downloading validation data to: {save_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=validation_files,
        local_dir=str(save_dir),
    )
    print("Validation download completed.")
    raise SystemExit(0)

train_candidates = sorted(eligible.get("train", []), key=lambda x: x[0])
train_by_driver = defaultdict(list)
for recording_dir, files in train_candidates:
    parts = recording_dir.split("/")
    if len(parts) < 3 or parts[0] != "dataset" or parts[1] != "train":
        raise ValueError(f"Unexpected training recording path: {recording_dir}")
    train_by_driver[parts[2]].append((recording_dir, files))

drivers = sorted(train_by_driver)
if len(drivers) < 2:
    raise RuntimeError(
        "At least two eligible training drivers are required to create "
        "disjoint train and test splits."
    )
test_driver_count = min(
    len(drivers) - 1,
    max(1, math.ceil(len(drivers) * args.test_driver_fraction)),
)
test_drivers = set(random.Random(SEED).sample(drivers, test_driver_count))
test_recordings = [
    recording
    for driver in sorted(test_drivers)
    for recording in train_by_driver[driver]
]
remaining_train = [
    recording
    for driver in drivers
    if driver not in test_drivers
    for recording in train_by_driver[driver]
]

train_count = min(
    len(remaining_train),
    max(1, math.ceil(len(remaining_train) * args.train_fraction)),
)
train_order = sorted(remaining_train, key=lambda x: x[0])
random.Random(SEED).shuffle(train_order)
selected_train = train_order[:train_count]

selected_files = []
for split in sorted(eligible):
    if split == "train":
        chosen = sorted(selected_train, key=lambda x: x[0])
    else:
        chosen = selected_non_train[split]

    print(f"{split}: {len(chosen)} recordings selected")
    if split == "val":
        print(
            f"  Official validation coverage: "
            f"{len(chosen)}/{len(eligible[split])} recordings"
        )

    for recording_dir, files in chosen:
        print(f"  Selected: {recording_dir}")
        selected_files.extend(files[name] for name in KEEP_FILES)

test_files = [
    files[name]
    for _, files in sorted(test_recordings, key=lambda x: x[0])
    for name in KEEP_FILES
]

if not selected_files or not test_files:
    raise RuntimeError(
        "No eligible train/validation or held-out test recordings found. "
        "Check the repository structure and dataset access terms."
    )

print(
    f"train: {len(selected_train)}/{len(remaining_train)} recordings from "
    f"{len(drivers) - len(test_drivers)}/{len(drivers)} drivers"
)
print(
    f"test: {len(test_recordings)} recordings from "
    f"{len(test_drivers)}/{len(drivers)} held-out drivers: "
    f"{', '.join(sorted(test_drivers))}"
)
print(f"\nTotal files to download: {len(selected_files)}")
print(f"Saving to: {save_dir}")
print(f"Test files to download: {len(test_files)}")
print(f"Saving test data to: {test_save_dir}")

for output_dir, files in (
    (save_dir, selected_files),
    (test_save_dir, test_files),
):
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=files,
        local_dir=str(output_dir),
    )

print("Download completed.")
