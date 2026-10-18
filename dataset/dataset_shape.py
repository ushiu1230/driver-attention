from PIL import Image
import numpy as np

path = "path/to/rgb/000001.jpg"

with Image.open(path) as img:
    print("Image size (W, H):", img.size)
    print("Image mode:", img.mode)

    arr = np.asarray(img)
    print("Array shape:", arr.shape)
    print("Data type:", arr.dtype)
    print("Min / max:", arr.min(), arr.max())