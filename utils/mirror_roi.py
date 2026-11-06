"""ROI geometry and binary detection metrics, independent of GPU inference."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_roi(path, require_confirmed=True):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    polygons = config.get("polygons", [])
    if require_confirmed and (config.get("confirmed") is not True or
                              not config.get("review_note", "").strip() or not polygons):
        raise ValueError("ROI is not reviewed: supply polygons, review_note and confirmed=true.")
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("Each normalized ROI polygon needs at least three [x,y] vertices.")
        if not np.isfinite(points).all() or ((points < 0) | (points > 1)).any():
            raise ValueError("ROI vertices must be finite and within [0,1].")
        area = abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) -
                   np.dot(points[:, 1], np.roll(points[:, 0], 1))) / 2
        if area <= 1e-12:
            raise ValueError("ROI polygon has zero area.")
    return config


def contains(points, polygons):
    """Union of simple polygons; normalized boundary points count as inside."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    x, y = points.T
    union = np.zeros(len(points), dtype=bool)
    for polygon in polygons:
        vertices = np.asarray(polygon, dtype=float)
        inside = np.zeros(len(points), dtype=bool)
        boundary = np.zeros(len(points), dtype=bool)
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            cross = (x-a[0])*(b[1]-a[1]) - (y-a[1])*(b[0]-a[0])
            boundary |= ((abs(cross) <= 1e-10) &
                         (x >= min(a[0], b[0])-1e-10) & (x <= max(a[0], b[0])+1e-10) &
                         (y >= min(a[1], b[1])-1e-10) & (y <= max(a[1], b[1])+1e-10))
            if b[1] != a[1]:
                intersects = ((a[1] > y) != (b[1] > y))
                inside ^= intersects & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
        union |= inside | boundary
    return union & np.isfinite(points).all(axis=1)


def roi_mask(polygons, height, width):
    if min(height, width) <= 1:
        raise ValueError("ROI raster needs dimensions greater than one.")
    yy, xx = np.mgrid[:height, :width]
    return contains(np.column_stack((xx.ravel()/width, yy.ravel()/height)),
                    polygons).reshape(height, width)


def gaze_label(metadata, frame_ids, size, polygons):
    """Only supplied input IDs are used; every input frame must have valid gaze."""
    width, height = size
    if len(frame_ids) != 16 or len(set(frame_ids)) != 16:
        raise ValueError("Plan 2 requires exactly 16 distinct input frame IDs.")
    subset = metadata.set_index("Frame").reindex(frame_ids)
    xy = subset[["gaze_x", "gaze_y"]].apply(
        __import__("pandas").to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < width) &
             (xy[:, 1] >= 0) & (xy[:, 1] < height))
    if not valid.all():
        return None
    return int(contains(xy / np.array([width, height]), polygons).any())


def binary_metrics(labels, scores, threshold):
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(scores) >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    precision = tp / max(1, tp+fp)
    recall = tp / max(1, tp+fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": 2*tp/max(1, 2*tp+fp+fn),
            "prevalence": float(y.mean()) if len(y) else None}


def pr_metrics(labels, scores):
    """Tie-aware AP and trapezoidal PR-AUC, positive class = mirror."""
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both classes are required.")
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.cumsum(y)[ends]
    precision, recall = tp/(ends+1), tp/y.sum()
    ap = np.sum(np.diff(np.r_[0., recall])*precision)
    area = np.sum(np.diff(np.r_[0., recall]) *
                  (np.r_[1., precision[:-1]] + precision)/2)
    return {"ap": float(ap), "pr_auc_trapezoid": float(area)}


def choose_threshold(labels, scores):
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if set(y.tolist()) != {0, 1} or not np.isfinite(s).all():
        raise ValueError("Calibration requires both classes and finite scores.")
    # Evaluate all distinct cutoffs in O(N log N), including tied score groups.
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.cumsum(y)[ends]
    fp, fn = ends+1-tp, y.sum()-tp
    f1 = 2*tp/np.maximum(1, 2*tp+fp+fn)
    precision = tp/(ends+1)
    best = max(range(len(ends)), key=lambda i: (f1[i], precision[i], s[ends[i]]))
    return float(s[ends[best]])
