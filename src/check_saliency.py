
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Change these paths to your dataset directories
RGB_DIR = Path("drivergaze360_subset/dataset/train/C011/001/001/rgb")
SALIENCY_DIR = Path("drivergaze360_subset/dataset/C011/001/001/saliency")

def check_frame(frame_id):
    filename = f"{frame_id:06d}.jpg"
    rgb_path = RGB_DIR / filename
    sal_path = SALIENCY_DIR / filename

    if not rgb_path.exists() or not sal_path.exists():
        print(f"Missing frame: {filename}")
        return

    # Load corresponding frames
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    sal = np.array(Image.open(sal_path))

    print(f"Frame: {filename}")
    print(f"RGB shape: {rgb.shape}")
    print(f"Saliency shape: {sal.shape}")
    print(f"Saliency range: {sal.min()} to {sal.max()}")

    # Convert saliency to grayscale for visualization
    if sal.ndim == 3:
        sal_gray = cv2.cvtColor(sal, cv2.COLOR_RGB2GRAY)
    else:
        sal_gray = sal

    # Resize only for visualization if dimensions differ
    if sal_gray.shape != rgb.shape[:2]:
        sal_gray = cv2.resize(
            sal_gray,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Normalize for visualization, not for training
    sal_float = sal_gray.astype(np.float32)
    sal_norm = cv2.normalize(
        sal_float, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(sal_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(rgb, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB frame")

    axes[1].imshow(sal_norm, cmap="hot")
    axes[1].set_title("Saliency map")

    axes[2].imshow(overlay)
    axes[2].set_title("RGB + saliency overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


for frame_id in [130, 500, 1000]:
    check_frame(frame_id)
