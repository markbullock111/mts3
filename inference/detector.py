from __future__ import annotations

from pathlib import Path

import numpy as np

from .tracker import Detection

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


class YOLOPersonDetector:
    def __init__(self, model_path: str | Path, device: str | None = None, conf: float = 0.25):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[Detection]:
        kwargs = {"verbose": False, "classes": [0], "conf": self.conf}
        if self.device:
            kwargs["device"] = self.device
        res = self.model.predict(frame, **kwargs)
        if not res:
            return []
        boxes = res[0].boxes
        if boxes is None:
            return []
        out: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        for b, c, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in b.tolist()]
            out.append(Detection(bbox=(x1, y1, x2, y2), conf=float(c), cls_id=int(cls_id)))
        return out
