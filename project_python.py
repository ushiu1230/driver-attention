import os

folders = [
    "src/",
    "data/",
    "data/raw/",
    "data/processed/",
    "checkpoints/",
    "results/",
]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")