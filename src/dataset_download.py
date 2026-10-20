
import math
import random
from collections import defaultdict
from huggingface_hub import HfApi, snapshot_download

# Configuration
REPO_ID = "dfki-av/drivergaze360"
REPO_TYPE = "dataset"
SAVE_DIR = "drivergaze360_subset"
FRACTION = 0.20
SEED = 42

KEEP_FILES = {"rgb.mp4", "saliency.mp4", "sim_gaze_df.csv"}

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

# Select 20% independently within each original split
rng = random.Random(SEED)
selected_files = []

for split in sorted(eligible):
    candidates = sorted(eligible[split], key=lambda x: x[0])

    if not candidates:
        continue

    n = max(1, math.ceil(len(candidates) * FRACTION))
    n = min(n, len(candidates))

    chosen = sorted(rng.sample(candidates, n), key=lambda x: x[0])

    print(f"{split}: {n}/{len(candidates)} recordings selected")

    for recording_dir, files in chosen:
        print(f"  Selected: {recording_dir}")
        selected_files.extend(files[name] for name in KEEP_FILES)

if not selected_files:
    raise RuntimeError(
        "No eligible recordings found. Check the repository structure "
        "and whether you have accepted the dataset access terms."
    )

print(f"\nTotal files to download: {len(selected_files)}")
print(f"Saving to: {SAVE_DIR}")

snapshot_download(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    allow_patterns=selected_files,
    local_dir=SAVE_DIR,
)

print("Download completed.")
