"""Bounded, process-local video handles; never cache whole decoded videos."""
import os
from collections import OrderedDict

import cv2
import torch


class VideoReader:
    def __init__(self, max_open=4):
        if max_open < 1:
            raise ValueError("Video cache size must be positive")
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

    def read_many(self, path, frame_ids):
        for frame_id in frame_ids:
            yield self.read(path, frame_id)

    def __del__(self):
        if hasattr(self, "handles"):
            self.close()


class DecordVideoReader:
    """CPU decoder with accurate indexed access and bounded chunk allocations.

    No output resizing here: fixation mapping still uses source dimensions.
    No global Decord bridge changes and no CUDA contexts in loader workers.
    """
    def __init__(self, max_open=4, num_threads=1):
        if max_open < 1 or num_threads < 1:
            raise ValueError("Video cache size and decoder threads must be positive")
        try:
            import decord  # noqa: F401
        except ImportError as error:
            raise ImportError("Decord backend requires: pip install decord") from error
        self.max_open = max_open
        self.num_threads = num_threads
        self.pid = os.getpid()
        self.handles = OrderedDict()

    def close(self):
        # Dropping the last reference frees Decord's native reader handle.
        self.handles.clear()

    def __getstate__(self):
        return {"max_open": self.max_open, "num_threads": self.num_threads,
                "pid": None, "handles": OrderedDict()}

    def _reader(self, path):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        path = str(path)
        reader = self.handles.pop(path, None)
        if reader is None:
            from decord import VideoReader as Reader, cpu
            if len(self.handles) >= self.max_open:
                self.handles.popitem(last=False)
            reader = Reader(path, ctx=cpu(0), num_threads=self.num_threads)
        self.handles[path] = reader
        return reader

    def read_many(self, path, frame_ids):
        reader = self._reader(path)
        indices = [int(frame_id) - 1 for frame_id in frame_ids]
        if any(index < 0 or index >= len(reader) for index in indices):
            raise IndexError(f"Frame IDs out of video bounds: {path}")
        # Avoid materializing all 16 panoramic source frames at once.
        for start in range(0, len(indices), 4):
            frames = reader.get_batch(indices[start:start + 4]).asnumpy()
            for frame in frames:
                yield torch.from_numpy(frame).permute(2, 0, 1).contiguous()

    def read(self, path, frame_id, grayscale=False):
        frame = next(self.read_many(path, [frame_id]))
        if grayscale:
            image = frame.permute(1, 2, 0).numpy()
            return torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)).unsqueeze(0)
        return frame

    def __del__(self):
        if hasattr(self, "handles"):
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
