from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2

from dataset.dataset import resize_with_pad
from models import SingleFrameSaliencyNet


checkpoint_path = Path("best_single_frame.pt")
image_path = Path(r"dataset\drivergaze360_subset\dataset\test\C017\012\002\rgb\000008.jpg")
output_path = Path("saliency_prediction.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dùng pretrained=False vì trọng số đã được nạp từ checkpoint.
model = SingleFrameSaliencyNet(pretrained=False).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval()

image = read_image(str(image_path), mode=ImageReadMode.RGB)
image = resize_with_pad(image, (224, 1120))

preprocess = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
image = preprocess(image)

# Model nhận [B, T, C, H, W]; baseline này chỉ dùng một frame.
input_tensor = image.unsqueeze(0).unsqueeze(0).to(device)

with torch.inference_mode():
    prediction = model(input_tensor)

# prediction có dạng [1, 1, 224, 1120].
saliency = prediction[0, 0].cpu().numpy()

# Chuẩn hóa chỉ để xem heatmap; giữ `saliency` gốc nếu dùng tính metric.
visual = saliency - saliency.min()
visual = visual / (visual.max() + 1e-8)
Image.fromarray((visual * 255).astype(np.uint8)).save(output_path)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Validation KLD: {checkpoint['val_loss']:.4f}")
print(f"Prediction shape: {saliency.shape}")
print(f"Saved visualization: {output_path.resolve()}")