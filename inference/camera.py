from __future__ import annotations

from dataclasses import dataclass
import sys
import time

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    ts_ms: int


class VideoSource:
    def __init__(self, source: str | int):
        self.source = source
        self._backend_candidates = self._backend_order(source)
        self._backend_idx = 0
        self.cap = self._open_capture_with_candidates(advance_backend=False)
        self._failed_reads = 0
        self._blank_reads = 0
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {source}")

    def read(self) -> FramePacket | None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._failed_reads += 1
            # Recover from flaky webcam backends (common on Windows MSMF).
            if self._failed_reads >= 8:
                self._reopen(advance_backend=True)
            return None
        self._failed_reads = 0
        if self._is_blank_frame(frame):
            self._blank_reads += 1
            if self._blank_reads >= 25:
                # Camera opened but is feeding blank frames. Try alternate backend.
                self._reopen(advance_backend=True)
                return None
        else:
            self._blank_reads = 0
        ts_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
        return FramePacket(frame=frame, ts_ms=ts_ms)

    def release(self) -> None:
        self.cap.release()

    def _reopen(self, advance_backend: bool) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(0.2)
        self.cap = self._open_capture_with_candidates(advance_backend=advance_backend)
        self._failed_reads = 0
        self._blank_reads = 0

    @staticmethod
    def _backend_order(source: str | int) -> list[int | None]:
        if isinstance(source, int) and sys.platform.startswith("win"):
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
        return [None]

    @staticmethod
    def _configure_webcam_capture(cap: cv2.VideoCapture) -> None:
        # Best-effort settings for stable Windows webcam stream.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    @staticmethod
    def _warmup_capture(cap: cv2.VideoCapture, count: int = 5) -> None:
        for _ in range(max(0, count)):
            ok, _ = cap.read()
            if not ok:
                break

    @staticmethod
    def _is_blank_frame(frame: np.ndarray) -> bool:
        if frame.size == 0:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < 4.0 and float(gray.std()) < 4.0

    @staticmethod
    def _open_capture(source: str | int, backend: int | None = None) -> cv2.VideoCapture:
        if isinstance(source, int):
            cap = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
            if cap.isOpened():
                VideoSource._configure_webcam_capture(cap)
                VideoSource._warmup_capture(cap)
            return cap

        source_str = str(source).strip()
        if source_str.lower().startswith("rtsp://"):
            cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
        return cv2.VideoCapture(source_str)

    def _open_capture_with_candidates(self, advance_backend: bool) -> cv2.VideoCapture:
        candidates = self._backend_candidates or [None]
        start = self._backend_idx
        if advance_backend and len(candidates) > 1:
            start = (self._backend_idx + 1) % len(candidates)
        for i in range(len(candidates)):
            idx = (start + i) % len(candidates)
            cap = self._open_capture(self.source, backend=candidates[idx])
            if cap.isOpened():
                self._backend_idx = idx
                return cap
            cap.release()
        # Last fallback
        return self._open_capture(self.source)
