from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch

model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

for i, layer in enumerate(model.features):
    print(i, layer)

x = torch.randn(1, 3, 224, 1120)
stage_indices = (3, 6, 12, 16)

model.eval()

with torch.no_grad():
    for i, layer in enumerate(model.features):
        x = layer(x)

        if i in stage_indices:
            print(i, x.shape)