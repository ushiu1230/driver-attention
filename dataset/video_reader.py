"""Bounded, process-local video handles; never cache whole decoded videos."""
import os
from collections import OrderedDict

import cv2
import torch


class VideoReader:
    def __init__(self, max_open=4):
        self.max_open = max_open
        self.pid = os.getpid()
        self.handles = OrderedDict()

    def close(self):
        for capture in self.handles.values():
            capture.release()
        self.handles.clear()

    def __getstate__(self):
        # Windows/spawn workers must open their own handles.
        return {"max_open": self.max_open, "pid": None, "handles": OrderedDict()}

    def read(self, path, frame_id, grayscale=False):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        path = str(path)
        capture = self.handles.pop(path, None)
        if capture is None:
            if len(self.handles) >= self.max_open:
                _, old = self.handles.popitem(last=False)
                old.release()
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Cannot open video: {path}")
        self.handles[path] = capture
        # CSV IDs are 1-based; decoder offsets are 0-based. Consecutive RGB
        # frames use read() without repeatedly seeking to a keyframe.
        target = int(frame_id) - 1
        position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if position != target:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                raise RuntimeError(f"Cannot seek to frame {frame_id}: {path}")
            if int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != target:
                raise RuntimeError(f"Inexact seek to frame {frame_id}: {path}")
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {frame_id}: {path}")
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return torch.from_numpy(frame).unsqueeze(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame).permute(2, 0, 1).contiguous()

    def __del__(self):
        self.close()


def validate_video(path, frame_count):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        # Existing preparation trims videos to CSV length; allow trailing frames.
        if count < frame_count:
            raise ValueError(f"Video {path} has {count} frames; CSV needs {frame_count}")
    finally:
        capture.release()
