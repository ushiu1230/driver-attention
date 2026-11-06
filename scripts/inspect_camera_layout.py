"""CPU-only visual evidence for five equal-width camera panels; no gaze labels."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evidence = []
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    for index, source in enumerate(args.images):
        path = Path(source)
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        w, h = image.size
        if (w, h) != (6400, 720):
            raise ValueError(f"Unexpected geometry: {path}: {image.size}")
        pixels = np.asarray(image, dtype=float)
        column_jump = abs(pixels[:, 1:] - pixels[:, :-1]).mean(axis=(0, 2))
        seams = []
        for x in (1280, 2560, 3840, 5120):
            seams.append({"x": x, "adjacent_column_rgb_difference": float(column_jump[x-1]),
                          "local_peak_x_within_20px": int(x-20+np.argmax(column_jump[x-21:x+20]))})
        scaled = image.resize((1600, 180))
        canvas = Image.new("RGB", (1600, 250), "white")
        canvas.paste(scaled, (0, 40))
        draw = ImageDraw.Draw(canvas, "RGBA")
        for a, b in ((0, 320), (1280, 1599)):
            draw.rectangle((a, 40, b, 219), fill=(255, 0, 0, 45))
        names = ["1: rear-view candidate", "2: front-side", "3: front", "4: front-side", "5: rear-view candidate"]
        for i, label in enumerate(names):
            x = i*320
            draw.text((x+6, 9), label, fill="black", font=font)
            draw.line((x, 40, x, 219), fill="yellow", width=3)
            draw.text((x+6, 224), f"source x=[{i*1280},{(i+1)*1280})", fill="black", font=font)
        target = out / f"camera_panels_{index}.png"
        canvas.convert("RGB").save(target)
        evidence.append({"image": str(path.resolve()), "size": [w, h], "seams": seams,
                         "overall_median_column_jump": float(np.median(column_jump)),
                         "overlay": str(target.resolve())})
    (out / "camera_layout_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
