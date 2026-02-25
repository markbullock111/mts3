from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import numpy as np


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    conf: float
    cls_id: int = 0


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    conf: float
    age: int = 0
    hits: int = 1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class TrackerBase:
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:  # pragma: no cover
        raise NotImplementedError


class IoUTracker(TrackerBase):
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15, min_conf: float = 0.2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_conf = min_conf
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter + 1e-9
        return inter / union

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        active_ids = set(self._tracks.keys())
        dets = [d for d in detections if d.conf >= self.min_conf]
        matched_track_ids: set[int] = set()
        used_det_idx: set[int] = set()

        # Greedy matching by IoU for simplicity
        pairs: list[tuple[float, int, int]] = []
        track_items = list(self._tracks.items())
        for ti, (track_id, track) in enumerate(track_items):
            for di, det in enumerate(dets):
                iou = self._iou(track.bbox, det.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, track_id, di))
        pairs.sort(reverse=True, key=lambda x: x[0])
        for iou, track_id, di in pairs:
            if track_id in matched_track_ids or di in used_det_idx:
                continue
            det = dets[di]
            t = self._tracks[track_id]
            t.bbox = det.bbox
            t.conf = det.conf
            t.age = 0
            t.hits += 1
            matched_track_ids.add(track_id)
            used_det_idx.add(di)

        for di, det in enumerate(dets):
            if di in used_det_idx:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(track_id=tid, bbox=det.bbox, conf=det.conf)
            matched_track_ids.add(tid)

        expired: list[int] = []
        for tid, trk in self._tracks.items():
            if tid not in matched_track_ids:
                trk.age += 1
            if trk.age > self.max_age:
                expired.append(tid)
        for tid in expired:
            self._tracks.pop(tid, None)

        return list(self._tracks.values())


class ByteTrackWrapper(TrackerBase):
    """Best-effort adapter to Ultralytics BYTETracker.

    Falls back by raising RuntimeError if imports/signature mismatch. Caller should catch and use IoUTracker.
    """

    def __init__(self, conf_thresh: float = 0.25, track_thresh: float = 0.25, match_thresh: float = 0.8, fps: int = 30):
        try:
            from types import SimpleNamespace
            from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"ByteTrack unavailable: {exc}") from exc

        args = SimpleNamespace(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=30,
            frame_rate=fps,
            mot20=False,
            conf=conf_thresh,
        )
        self._BYTETracker = BYTETracker
        self._args = args
        self._tracker = BYTETracker(args, frame_rate=fps)

    def _update_internal(self, detections: list[Detection], frame: np.ndarray):
        # Try common BYTETracker signatures used across ultralytics/yolox variants.
        arr = np.array([[*d.bbox, d.conf, d.cls_id] for d in detections], dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)
        h, w = frame.shape[:2]
        errors: list[Exception] = []
        for call in (
            lambda: self._tracker.update(arr[:, :5], (h, w), (h, w)),
            lambda: self._tracker.update(arr[:, :5], frame),
            lambda: self._tracker.update(arr[:, :6], (h, w), (h, w)),
            lambda: self._tracker.update(arr),
        ):
            try:
                return call()
            except Exception as exc:
                errors.append(exc)
        raise RuntimeError(f"BYTETracker signature mismatch: {errors[-1] if errors else 'unknown'}")

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        outputs = self._update_internal(detections, frame)
        tracks: list[Track] = []
        for obj in outputs or []:
            tid = int(getattr(obj, "track_id", getattr(obj, "id", -1)))
            tlbr = getattr(obj, "tlbr", None)
            if tlbr is None:
                tlwh = getattr(obj, "tlwh", None)
                if tlwh is not None:
                    x, y, w, h = tlwh
                    tlbr = [x, y, x + w, y + h]
            if tid < 0 or tlbr is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in tlbr]
            score = float(getattr(obj, "score", 0.0))
            tracks.append(Track(track_id=tid, bbox=(x1, y1, x2, y2), conf=score))
        return tracks


def build_tracker(prefer_bytetrack: bool = True, **kwargs: Any) -> TrackerBase:
    if prefer_bytetrack:
        try:
            return ByteTrackWrapper(**kwargs)
        except Exception:
            pass
    return IoUTracker()
