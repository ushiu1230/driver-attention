from pathlib import Path

project_root = Path(__file__).resolve().parent
folders = (
    "data/raw",
    "data/processed",
    "checkpoints",
    "results/figures",
    "results/logs",
)

for folder in folders:
    path = project_root / folder
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")