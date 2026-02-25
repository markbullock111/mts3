from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    ts_ms: int


class VideoSource:
    def __init__(self, source: str | int):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {source}")

    def read(self) -> FramePacket | None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        ts_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
        return FramePacket(frame=frame, ts_ms=ts_ms)

    def release(self) -> None:
        self.cap.release()
