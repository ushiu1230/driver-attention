#!/usr/bin/env python3
"""
Audit DriverGaze360 dataset schema and RGB-saliency pairing.

Expected extracted layout (per recording):
  <recording>/rgb/000001.jpg
  <recording>/saliency/000001.jpg
  <recording>/sim_gaze_df.csv

Usage:
  python audit_drivergaze360.py "/path/to/DriverGaze360"
  python audit_drivergaze360.py "/path/to/DriverGaze360" --full-image-verify
  python audit_drivergaze360.py "D:/Master/CS2203_ComputerVision/04_Projects/driver-attention/dataset" --full-image-verify

Requires: pandas, Pillow
  pip install pandas pillow
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def frame_number(path: Path) -> Optional[int]:
    """Extract the last integer in a filename stem, e.g. 000123 -> 123."""
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def image_files(folder: Path) -> dict[int, Path]:
    if not folder.exists() or not folder.is_dir():
        return {}
    found = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            n = frame_number(p)
            if n is not None:
                # If duplicate formats share the same frame number, first sorted path wins.
                found.setdefault(n, p)
    return found


def image_header(path: Path):
    """Return basic image metadata without decoding the full pixel array."""
    with Image.open(path) as im:
        return {
            "width": int(im.width),
            "height": int(im.height),
            "mode": im.mode,
            "format": im.format,
        }


def full_image_check(path: Path):
    """Decode every pixel to catch truncated/corrupt files."""
    with Image.open(path) as im:
        im.load()
        return {"width": int(im.width), "height": int(im.height), "mode": im.mode}


def numeric_frame_ids(df: pd.DataFrame, frame_col: Optional[str]):
    if not frame_col:
        return set(), 0
    nums = pd.to_numeric(df[frame_col], errors="coerce")
    valid = nums.dropna()
    non_integer = int((valid % 1 != 0).sum())
    ids = set(valid[valid % 1 == 0].astype(int).tolist())
    return ids, non_integer


def make_preview(recording_name, pairs, preview_dir: Path, max_pairs: int = 12):
    """Create a visual contact sheet of RGB and saliency examples."""
    if not pairs:
        return None
    preview_dir.mkdir(parents=True, exist_ok=True)
    selected = pairs[:max_pairs]
    thumb_w, thumb_h = 560, 210
    label_h, gap = 26, 12
    rows = len(selected)
    canvas = Image.new("RGB", (thumb_w * 2 + gap * 3, rows * (thumb_h + label_h + gap) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    y = gap
    for frame_id, rgb_path, sal_path in selected:
        for x, path, label in [
            (gap, rgb_path, f"RGB | frame {frame_id:06d}"),
            (thumb_w + gap * 2, sal_path, f"Saliency | frame {frame_id:06d}"),
        ]:
            draw.text((x, y), label, fill="black", font=font)
            box = (thumb_w, thumb_h)
            try:
                with Image.open(path) as im:
                    im.load()
                    vis = im.convert("RGB")
                    vis = ImageOps.contain(vis, box, method=Image.Resampling.LANCZOS)
                    # center thumbnail
                    px = x + (thumb_w - vis.width) // 2
                    py = y + label_h + (thumb_h - vis.height) // 2
                    canvas.paste(vis, (px, py))
            except Exception as exc:
                draw.text((x, y + label_h), f"Could not decode: {exc}", fill="red", font=font)
        y += thumb_h + label_h + gap
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", recording_name)
    out = preview_dir / f"{safe_name}_rgb_saliency_preview.jpg"
    canvas.save(out, quality=90)
    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Audit DriverGaze360 CSV schema and RGB-saliency frame matching.")
    parser.add_argument("root", type=Path, help="Dataset root directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Report output directory (default: <root>/audit_report)")
    parser.add_argument("--full-image-verify", action="store_true",
                        help="Decode every paired image to check for corrupt/truncated files; can be slow on large datasets.")
    parser.add_argument("--preview-pairs", type=int, default=12, help="Number of paired frames shown in each contact sheet.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (args.out_dir or (root / "audit_report")).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"ERROR: dataset root not found or not a directory: {root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"

    csv_paths = sorted(root.rglob("sim_gaze_df.csv"))
    if not csv_paths:
        print("No sim_gaze_df.csv found. Check root path or whether you are looking at the original packaged files.")
        print("Original release may contain rgb.mp4 and saliency.mp4; see official preparation script in the repository.")
        raise SystemExit(2)

    schema_rows = []
    audit_rows = []
    image_rows = []
    notes = []

    for csv_path in csv_paths:
        rec = csv_path.parent
        rel = str(rec.relative_to(root))
        rgb_dir = rec / "rgb"
        sal_dir = rec / "saliency"
        rgb_map = image_files(rgb_dir)
        sal_map = image_files(sal_dir)

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            audit_rows.append({
                "recording": rel, "status": "CSV_READ_ERROR", "csv_rows": None,
                "frame_column": None, "csv_unique_frames": None, "rgb_files": len(rgb_map),
                "saliency_files": len(sal_map), "matched_rgb_saliency": 0,
                "missing_rgb_vs_csv": None, "missing_saliency_vs_csv": None,
                "rgb_without_saliency": None, "saliency_without_rgb": None,
                "duplicate_frame_rows": None, "non_integer_frame_values": None,
                "rgb_saliency_shape_mismatch": None, "corrupt_images": None,
                "notes": str(exc),
            })
            continue

        df.columns = [str(c).strip() for c in df.columns]
        frame_col = next((c for c in df.columns if c.lower() == "frame"), None)
        frame_ids, non_integer_frames = numeric_frame_ids(df, frame_col)

        # Schema report: one row per CSV column.
        for col in df.columns:
            ser = df[col]
            schema_rows.append({
                "recording": rel,
                "column": col,
                "dtype_inferred": str(ser.dtype),
                "rows": len(df),
                "missing_count": int(ser.isna().sum()),
                "missing_percent": round(float(ser.isna().mean() * 100), 4) if len(df) else 0,
                "unique_count": int(ser.nunique(dropna=True)),
                "example_values": " | ".join(map(str, ser.dropna().head(3).tolist())),
            })

        duplicate_frame_rows = int(df[frame_col].duplicated(keep=False).sum()) if frame_col else None
        rgb_ids = set(rgb_map)
        sal_ids = set(sal_map)
        matched_ids = sorted(rgb_ids & sal_ids)
        missing_rgb = sorted(frame_ids - rgb_ids) if frame_col else []
        missing_sal = sorted(frame_ids - sal_ids) if frame_col else []
        rgb_without_sal = sorted(rgb_ids - sal_ids)
        sal_without_rgb = sorted(sal_ids - rgb_ids)

        rgb_sal_shape_mismatch = 0
        corrupt_images = 0
        sample_pairs = []
        first_rgb_info = None
        first_sal_info = None
        observed_rgb_shapes = set()
        observed_sal_shapes = set()
        ids_to_check = sorted(rgb_ids | sal_ids)
        if not args.full_image_verify:
            ids_to_decode = set(matched_ids[:max(args.preview_pairs, 1)])
        else:
            ids_to_decode = set(matched_ids)

        # Read headers for all images; fully decode selected samples or all samples under full mode.
        for fid in ids_to_check:
            rp, sp = rgb_map.get(fid), sal_map.get(fid)
            ri = si = None
            if rp:
                try:
                    ri = image_header(rp)
                    observed_rgb_shapes.add((ri["height"], ri["width"], ri["mode"]))
                    if first_rgb_info is None:
                        first_rgb_info = ri
                except Exception as exc:
                    corrupt_images += 1
                    notes.append(f"{rel}: RGB frame {fid} header error: {exc}")
            if sp:
                try:
                    si = image_header(sp)
                    observed_sal_shapes.add((si["height"], si["width"], si["mode"]))
                    if first_sal_info is None:
                        first_sal_info = si
                except Exception as exc:
                    corrupt_images += 1
                    notes.append(f"{rel}: saliency frame {fid} header error: {exc}")
            if ri and si and (ri["width"] != si["width"] or ri["height"] != si["height"]):
                rgb_sal_shape_mismatch += 1
            if fid in ids_to_decode:
                for p in (rp, sp):
                    if p:
                        try:
                            full_image_check(p)
                        except Exception as exc:
                            corrupt_images += 1
                            notes.append(f"{rel}: full decode error {p.name}: {exc}")
                if rp and sp:
                    sample_pairs.append((fid, rp, sp))

        preview_path = make_preview(rel, sample_pairs, preview_dir, args.preview_pairs)
        status = "PASS" if (
            frame_col and len(rgb_ids) > 0 and len(sal_ids) > 0
            and not missing_rgb and not missing_sal
            and not rgb_without_sal and not sal_without_rgb
            and not duplicate_frame_rows and not non_integer_frames
            and rgb_sal_shape_mismatch == 0 and corrupt_images == 0
        ) else "CHECK"

        audit_rows.append({
            "recording": rel,
            "status": status,
            "csv_rows": len(df),
            "csv_columns": len(df.columns),
            "frame_column": frame_col,
            "csv_unique_frames": len(frame_ids),
            "frame_min": min(frame_ids) if frame_ids else None,
            "frame_max": max(frame_ids) if frame_ids else None,
            "rgb_files": len(rgb_ids),
            "saliency_files": len(sal_ids),
            "matched_rgb_saliency": len(matched_ids),
            "missing_rgb_vs_csv": len(missing_rgb),
            "missing_saliency_vs_csv": len(missing_sal),
            "rgb_without_saliency": len(rgb_without_sal),
            "saliency_without_rgb": len(sal_without_rgb),
            "duplicate_frame_rows": duplicate_frame_rows,
            "non_integer_frame_values": non_integer_frames,
            "rgb_saliency_shape_mismatch": rgb_sal_shape_mismatch,
            "corrupt_images_checked": corrupt_images,
            "first_rgb_info": json.dumps(first_rgb_info, ensure_ascii=False) if first_rgb_info else None,
            "first_saliency_info": json.dumps(first_sal_info, ensure_ascii=False) if first_sal_info else None,
            "distinct_rgb_shapes_H_W_mode": json.dumps(sorted([list(x) for x in observed_rgb_shapes])),
            "distinct_saliency_shapes_H_W_mode": json.dumps(sorted([list(x) for x in observed_sal_shapes])),
            "preview_path": preview_path,
            "notes": "",
        })

        image_rows.append({
            "recording": rel,
            "first_rgb_info": json.dumps(first_rgb_info, ensure_ascii=False) if first_rgb_info else None,
            "first_saliency_info": json.dumps(first_sal_info, ensure_ascii=False) if first_sal_info else None,
            "rgb_shapes_seen": json.dumps(sorted([list(x) for x in observed_rgb_shapes])),
            "saliency_shapes_seen": json.dumps(sorted([list(x) for x in observed_sal_shapes])),
            "shape_mismatch_pairs": rgb_sal_shape_mismatch,
            "preview_path": preview_path,
        })

        print(f"[{status}] {rel}: CSV rows={len(df)}, RGB={len(rgb_ids)}, saliency={len(sal_ids)}, pairs={len(matched_ids)}, "
              f"missing RGB={len(missing_rgb)}, missing saliency={len(missing_sal)}, shape mismatch={rgb_sal_shape_mismatch}")

    pd.DataFrame(audit_rows).to_csv(out_dir / "pair_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(schema_rows).to_csv(out_dir / "csv_schema.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(image_rows).to_csv(out_dir / "image_shape_audit.csv", index=False, encoding="utf-8-sig")
    (out_dir / "audit_notes.txt").write_text(
        "Interpretation guide\n"
        "===================\n"
        "PASS means the mechanical checks in this script found no mismatch for the checked files.\n"
        "It does NOT prove the saliency annotation semantically represents driver gaze correctly.\n"
        "Visually inspect the preview images and verify annotation meaning against the paper/dataset card.\n"
        "Check that Frame values are one-based and correspond to filenames 000001.jpg, 000002.jpg, ...\n"
        "If frames are not extracted yet, use the official preparation script from the DriverGaze360 repository.\n\n"
        + "\n".join(notes),
        encoding="utf-8",
    )
    print("\nReports written to:", out_dir)
    print(" - pair_audit.csv")
    print(" - csv_schema.csv")
    print(" - image_shape_audit.csv")
    print(" - previews/")
    print(" - audit_notes.txt")


if __name__ == "__main__":
    main()
