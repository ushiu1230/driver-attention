from dataset.dataset import DG360LiteDataset

train_ds = DG360LiteDataset(
    root="drivergaze360_subset/dataset/train",
    T=16,                  # frames per clip
    frame_stride=1,        # spacing between frames in a clip
    overlap=8,              # actually the step between clip start positions
    img_size=(224, 1120),  # (height, width)
    skip_first=30,
    check=True,
)

print(len(train_ds))