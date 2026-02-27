from __future__ import annotations

from dataclasses import dataclass
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

        high = float(track_thresh)
        low = max(0.05, min(0.99, high * 0.5))
        args = SimpleNamespace(
            # Newer ultralytics ByteTracker arg names
            track_high_thresh=high,
            track_low_thresh=low,
            new_track_thresh=high,
            match_thresh=float(match_thresh),
            track_buffer=30,
            fuse_score=True,
            mot20=False,
            # Legacy compatibility keys used by older variants
            track_thresh=high,
            conf=float(conf_thresh),
            frame_rate=fps,
        )
        self._BYTETracker = BYTETracker
        self._args = args
        self._tracker = BYTETracker(args, frame_rate=fps)

    @staticmethod
    def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
        if xyxy.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + (w / 2.0)
        cy = y1 + (h / 2.0)
        return np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    @staticmethod
    def _build_ultralytics_results_like(arr: np.ndarray):
        # Newer ultralytics BYTETracker.update expects an object with .xywh/.conf/.cls attributes.
        class _ResultsLike:
            def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
                xyxy_arr = np.asarray(xyxy, dtype=np.float32)
                if xyxy_arr.ndim == 1:
                    xyxy_arr = xyxy_arr.reshape(1, -1)
                self.xyxy = xyxy_arr[:, :4] if xyxy_arr.size else np.empty((0, 4), dtype=np.float32)
                self.xywh = ByteTrackWrapper._xyxy_to_xywh(self.xyxy)
                self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
                self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

            def __len__(self) -> int:
                return int(self.conf.shape[0])

            def __getitem__(self, idx):
                return _ResultsLike(self.xyxy[idx], self.conf[idx], self.cls[idx])

        xyxy = arr[:, :4] if arr.size else np.empty((0, 4), dtype=np.float32)
        conf = arr[:, 4] if arr.size else np.empty((0,), dtype=np.float32)
        cls = arr[:, 5] if arr.size else np.empty((0,), dtype=np.float32)
        return _ResultsLike(xyxy=xyxy, conf=conf, cls=cls)

    def _update_internal(self, detections: list[Detection], frame: np.ndarray):
        # Try common BYTETracker signatures used across ultralytics/yolox variants.
        arr = np.array([[*d.bbox, d.conf, d.cls_id] for d in detections], dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)
        result_like = self._build_ultralytics_results_like(arr)
        h, w = frame.shape[:2]
        errors: list[Exception] = []
        for call in (
            lambda: self._tracker.update(result_like, frame),
            lambda: self._tracker.update(result_like),
            lambda: self._tracker.update(arr[:, :5], (h, w), (h, w)),
            lambda: self._tracker.update(arr[:, :5], frame),
            lambda: self._tracker.update(arr[:, :6], (h, w), (h, w)),
            lambda: self._tracker.update(arr),
        ):
            try:
                return call()
            except Exception as exc:
                errors.append(exc)
        msg = "; ".join(str(e) for e in errors[-3:]) if errors else "unknown"
        raise RuntimeError(f"BYTETracker signature mismatch: {msg}")

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        outputs = self._update_internal(detections, frame)
        tracks: list[Track] = []
        if isinstance(outputs, np.ndarray):
            rows = outputs.tolist()
            for row in rows:
                if len(row) < 5:
                    continue
                x1, y1, x2, y2 = [float(v) for v in row[:4]]
                tid = int(row[4])
                score = float(row[5]) if len(row) > 5 else 0.0
                if tid < 0:
                    continue
                tracks.append(Track(track_id=tid, bbox=(x1, y1, x2, y2), conf=score))
            return tracks
        if isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], (list, tuple, np.ndarray)):
            for row in outputs:
                vals = list(row)
                if len(vals) < 5:
                    continue
                x1, y1, x2, y2 = [float(v) for v in vals[:4]]
                tid = int(vals[4])
                score = float(vals[5]) if len(vals) > 5 else 0.0
                if tid < 0:
                    continue
                tracks.append(Track(track_id=tid, bbox=(x1, y1, x2, y2), conf=score))
            return tracks
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


class FallbackTracker(TrackerBase):
    def __init__(self, primary: TrackerBase, fallback: TrackerBase):
        self.primary = primary
        self.fallback = fallback
        self._using_fallback = False

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        if self._using_fallback:
            return self.fallback.update(detections, frame)
        try:
            return self.primary.update(detections, frame)
        except Exception as exc:
            print(f"[tracker] ByteTrack failed ({exc}); switching to IoU tracker.")
            self._using_fallback = True
            return self.fallback.update(detections, frame)


def build_tracker(prefer_bytetrack: bool = True, **kwargs: Any) -> TrackerBase:
    if prefer_bytetrack:
        try:
            return FallbackTracker(primary=ByteTrackWrapper(**kwargs), fallback=IoUTracker())
        except Exception:
            pass
    return IoUTracker()
