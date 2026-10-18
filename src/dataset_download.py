import math
from huggingface_hub import HfApi, snapshot_download

# 1. Configure parameters
repo_id = "dfki-av/drivergaze360"
repo_type = "dataset"
save_dir = "./04_Projects/driver-attention/drivergaze360_subset"

# 2. Scan the file list on Hugging Face
api = HfApi()
all_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

# Analyze the folder structure from the file list
# Format: dataset/train/C001/001/001-1/rgb.mp4
train_participants = set()
val_participants = set()
test_participants = set()

recording_map = {}

for file_path in all_files:
    parts = file_path.split('/')
    if len(parts) >= 5 and parts[0] == "dataset":
        split_type = parts[1]  # train, val, or test
        driver_id = parts[2]   # C001, C002, ...
        recording_id = parts[3]  # 001, 002, ...
        sub_rec = parts[4]  # 001-1, ...

        rec_path = f"{split_type}/{driver_id}/{recording_id}/{sub_rec}"

        if split_type == "train":
            train_participants.add(driver_id)
        elif split_type == "val":
            val_participants.add(driver_id)
        elif split_type == "test":
            test_participants.add(driver_id)

        if driver_id not in recording_map:
            recording_map[driver_id] = []
        if rec_path not in recording_map[driver_id]:
            recording_map[driver_id].append(rec_path)

# Sort the driver list
train_list = sorted(list(train_participants))
val_list = sorted(list(val_participants))
test_list = sorted(list(test_participants))

# Step 1: Select 9 drivers according to the Balanced plan (6 Train, 2 Val, 1 Test)
selected_train = train_list[:6]

# Ensure there are no duplicate drivers across datasets
selected_val = [d for d in val_list if d not in selected_train][:2]
# If test_list is empty, we can fall back to selecting from other unused splits
selected_test = [d for d in test_list if d not in selected_train and d not in selected_val][:1]
if not selected_test:
    # Fallback: select any driver not already assigned to Train or Val
    all_drivers = sorted(list(recording_map.keys()))
    potential_test = [d for d in all_drivers if d not in selected_train and d not in selected_val]
    if potential_test:
        selected_test = [potential_test[0]]

print(f"Drivers selected for Train (6): {selected_train}")
print(f"Drivers selected for Val (2): {selected_val}")
print(f"Drivers selected for Test (1): {selected_test}")

# Step 2 & 4: Only keep exactly 1 representative recording for each driver and only keep rgb.mp4 + saliency.mp4
allow_patterns = []

all_selected_drivers = selected_train + selected_val + selected_test
for driver in all_selected_drivers:
    recs = sorted(recording_map.get(driver, []))
    if recs:
        # Select the first recording as the representative
        chosen_rec = recs[0]
        print(f"-> Driver {driver} selected representative recording: {chosen_rec}")
        allow_patterns.append(f"dataset/{chosen_rec}/rgb.mp4")
        allow_patterns.append(f"dataset/{chosen_rec}/saliency.mp4")
        allow_patterns.append(f"dataset/{chosen_rec}/sim_gaze_df.csv")

# 5. Download the exact data according to the optimal configuration
print("\nDownloading the optimal subset from Hugging Face...")
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    allow_patterns=allow_patterns,
    local_dir=save_dir
)
print("Subset download completed!")