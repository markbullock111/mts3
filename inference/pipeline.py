from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .backend_client import BackendClient
from .camera import VideoSource
from .config import RuntimeConfig
from .event_logic import DailyFirstCheckInDeduper, is_within_morning_window
from .face_recognizer import FaceRecognizer, FaceTrackBuffer
from .gallery_sync import GalleryState, GallerySyncService
from .geometry import crossed_entry_line, point_in_polygon
from .reid_recognizer import BodyTrackBuffer, ReIDRecognizer
from .tracker import Detection, Track, build_tracker
from .detector import YOLOPersonDetector


@dataclass
class TrackMemory:
    track_id: int
    first_seen_ts: float
    last_seen_ts: float
    last_center: tuple[float, float] | None = None
    current_center: tuple[float, float] | None = None
    face_buffer: FaceTrackBuffer = field(default_factory=lambda: FaceTrackBuffer(max_items=12))
    body_buffer: BodyTrackBuffer = field(default_factory=lambda: BodyTrackBuffer(max_items=8))
    crossed: bool = False
    finalized: bool = False
    posted: bool = False
    resolved_employee_id: int | None = None
    resolved_employee_name: str | None = None
    resolved_method: str | None = None
    resolved_confidence: float = 0.0
    live_employee_id: int | None = None
    live_employee_name: str | None = None
    live_method: str | None = None
    live_confidence: float = 0.0
    last_live_infer_ts: float = 0.0
    last_live_match_ts: float = 0.0
    event_ts_utc: datetime | None = None
    last_post_attempt_ts: float = 0.0
    post_attempts: int = 0
    last_post_error: str | None = None


class AttendancePipeline:
    def __init__(
        self,
        cfg: RuntimeConfig,
        backend_url: str,
        camera_source: str | int,
        show: bool = False,
        save_snapshots: bool | None = None,
        enroll_employee_id: int | None = None,
        enroll_kind: str = "face",
    ):
        self.cfg = cfg
        self.backend = BackendClient(backend_url)
        self.camera = VideoSource(camera_source)
        self.show = show
        self.save_snapshots = cfg.inference.save_snapshots_default if save_snapshots is None else bool(save_snapshots)
        self.snapshot_dir = cfg.repo_root / cfg.inference.snapshot_dir
        if self.save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.enroll_employee_id = enroll_employee_id
        self.enroll_kind = enroll_kind
        self._last_enroll_upload = 0.0

        device = "0" if (cfg.inference.use_gpu_if_available and self._torch_cuda()) else None
        self.detector = YOLOPersonDetector(cfg.repo_root / cfg.inference.yolo_model, device=device, conf=0.25)
        self.tracker = build_tracker(prefer_bytetrack=True, fps=30)
        self.face_recognizer = FaceRecognizer(
            cfg.repo_root / cfg.inference.face_model_dir,
            model_name=cfg.inference.face_model_name,
            use_gpu=cfg.inference.use_gpu_if_available,
        )
        reid_device = "cuda" if cfg.inference.use_gpu_if_available and self._torch_cuda() else "cpu"
        self.reid_recognizer = ReIDRecognizer(cfg.repo_root / cfg.inference.reid_model_path, device=reid_device)
        self.gallery = GallerySyncService(self.backend, refresh_seconds=cfg.inference.gallery_refresh_seconds)
        self.track_mem: dict[int, TrackMemory] = {}
        self.runtime_dedup = DailyFirstCheckInDeduper()
        self.camera_id = cfg.inference.camera_id
        self.window_name = "attendance"
        self._display_window_ready = False

    @staticmethod
    def _torch_cuda() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _screen_size() -> tuple[int, int]:
        if os.name != "nt":
            return (0, 0)
        try:
            import ctypes

            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            w = int(user32.GetSystemMetrics(0))
            h = int(user32.GetSystemMetrics(1))
            return (w, h)
        except Exception:
            return (0, 0)

    def _init_display_window(self, frame_shape: tuple[int, ...]) -> None:
        if self._display_window_ready or not self.show:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        sw, sh = self._screen_size()
        if sw > 0 and sh > 0:
            target_w = max(960, sw - 40)
            target_h = max(540, sh - 120)
        else:
            h, w = frame_shape[:2]
            target_w = max(1280, int(w))
            target_h = max(720, int(h))
        cv2.resizeWindow(self.window_name, int(target_w), int(target_h))
        try:
            cv2.moveWindow(self.window_name, 0, 0)
        except Exception:
            pass
        self._display_window_ready = True

    @staticmethod
    def _is_blank_frame(frame: np.ndarray) -> bool:
        if frame.size == 0:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < 4.0 and float(gray.std()) < 4.0

    @staticmethod
    def _draw_blank_frame_warning(frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        x1, y1 = 20, 20
        x2, y2 = min(w - 20, 980), min(h - 20, 150)
        if x2 > x1 and y2 > y1:
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, out, 0.45, 0.0, out)
        cv2.putText(
            out,
            "Blank camera frame detected.",
            (30, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "Try another source: --camera 1 (or close apps using webcam).",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def run(self) -> None:
        self.gallery.start()
        fps_hist: list[float] = []
        try:
            while True:
                t0 = time.perf_counter()
                packet = self.camera.read()
                if packet is None:
                    time.sleep(0.05)
                    continue
                frame = packet.frame
                if self._is_blank_frame(frame):
                    if self.show:
                        if not self._display_window_ready:
                            self._init_display_window(frame.shape)
                        warn = self._draw_blank_frame_warning(frame)
                        cv2.imshow(self.window_name, warn)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                    time.sleep(0.02)
                    continue
                now_ts = time.time()
                detections = self.detector.detect(frame)
                detections = [d for d in detections if self._bbox_area(d.bbox) >= self.cfg.inference.min_box_area]
                tracks = self.tracker.update(detections, frame)
                gallery_state = self.gallery.get_state()
                self._update_track_buffers(frame, tracks, packet.ts_ms, now_ts, gallery_state)

                if self.enroll_employee_id is not None:
                    self._maybe_enroll(frame, tracks)
                else:
                    self._finalize_crossed_tracks(frame, packet.ts_ms)

                self._cleanup_tracks(now_ts)
                dt = time.perf_counter() - t0
                if dt > 0:
                    fps_hist.append(1.0 / dt)
                    if len(fps_hist) > 30:
                        fps_hist.pop(0)
                if self.show:
                    if not self._display_window_ready:
                        self._init_display_window(frame.shape)
                    self._draw_overlays(frame, tracks, fps=float(sum(fps_hist) / max(1, len(fps_hist))))
                    cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        finally:
            self.gallery.stop()
            self.camera.release()
            if self.show:
                cv2.destroyAllWindows()

    @staticmethod
    def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _crop(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    def _update_track_buffers(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        ts_ms: int,
        now_ts: float,
        gallery_state: GalleryState,
    ) -> None:
        for tr in tracks:
            mem = self.track_mem.get(tr.track_id)
            if mem is None:
                mem = TrackMemory(track_id=tr.track_id, first_seen_ts=now_ts, last_seen_ts=now_ts)
                mem.face_buffer = FaceTrackBuffer(max_items=self.cfg.inference.face_buffer_size)
                self.track_mem[tr.track_id] = mem
            mem.last_seen_ts = now_ts
            mem.last_center = mem.current_center
            mem.current_center = tr.center

            person_crop = self._crop(frame, tr.bbox)
            if person_crop.size == 0:
                continue
            mem.body_buffer.add(person_crop, ts_ms)
            try:
                self.face_recognizer.add_best_face_from_person_crop(person_crop, mem.face_buffer, ts_ms)
            except Exception:
                pass

            self._update_live_identity(mem, now_ts, gallery_state)

            if not mem.crossed and mem.last_center is not None and mem.current_center is not None:
                if crossed_entry_line(
                    mem.last_center,
                    mem.current_center,
                    self.cfg.roi.entry_line_p1,
                    self.cfg.roi.entry_line_p2,
                    self.cfg.roi.roi_polygon,
                ):
                    mem.crossed = True
            # Fallback trigger: if person is inside ROI for a short dwell time, finalize anyway.
            # This prevents missed check-ins when line geometry does not match the live camera framing.
            if (not mem.crossed) and mem.current_center is not None:
                if point_in_polygon(mem.current_center, self.cfg.roi.roi_polygon):
                    if (now_ts - mem.first_seen_ts) >= 1.2:
                        mem.crossed = True

    def _update_live_identity(self, mem: TrackMemory, now_ts: float, state: GalleryState) -> None:
        # Keep inference overhead bounded; update live identity at most twice per second per track.
        if now_ts - mem.last_live_infer_ts < 0.5:
            return
        mem.last_live_infer_ts = now_ts

        found = False
        if state.face_matcher is not None:
            try:
                face_match = self.face_recognizer.identify_from_buffer(
                    mem.face_buffer,
                    state.face_matcher,
                    threshold=self.cfg.inference.face_threshold,
                    top_k_frames=max(1, min(2, self.cfg.inference.face_top_k_frames)),
                )
            except Exception:
                face_match = None
            if face_match is not None:
                mem.live_employee_id = face_match.employee_id
                mem.live_employee_name = (face_match.employee_name or "").strip() or None
                mem.live_method = "face"
                mem.live_confidence = float(face_match.score)
                mem.last_live_match_ts = now_ts
                found = True

        if not found and state.reid_matcher is not None:
            try:
                reid_match = self.reid_recognizer.identify_from_buffer(
                    mem.body_buffer,
                    state.reid_matcher,
                    threshold=self.cfg.inference.reid_threshold,
                    top_k_frames=2,
                )
            except Exception:
                reid_match = None
            if reid_match is not None:
                mem.live_employee_id = reid_match.employee_id
                mem.live_employee_name = (reid_match.employee_name or "").strip() or None
                mem.live_method = "reid"
                mem.live_confidence = float(reid_match.score)
                mem.last_live_match_ts = now_ts
                found = True

        # Short stickiness avoids rapid flicker between name and "Detecting...".
        if (not found) and (now_ts - mem.last_live_match_ts > 1.5):
            mem.live_employee_id = None
            mem.live_employee_name = None
            mem.live_method = None
            mem.live_confidence = 0.0

    def _finalize_crossed_tracks(self, frame: np.ndarray, ts_ms: int) -> None:
        state = self.gallery.get_state()
        now_ts = time.time()
        for trk_id, mem in list(self.track_mem.items()):
            if mem.finalized or not mem.crossed:
                continue
            if mem.last_post_attempt_ts and (now_ts - mem.last_post_attempt_ts) < 1.0:
                continue
            if state.face_matcher is None or state.reid_matcher is None:
                continue
            employee_id = None
            method = "unknown"
            confidence = 0.0

            face_match = None
            try:
                face_match = self.face_recognizer.identify_from_buffer(
                    mem.face_buffer,
                    state.face_matcher,
                    threshold=self.cfg.inference.face_threshold,
                    top_k_frames=self.cfg.inference.face_top_k_frames,
                )
            except Exception:
                face_match = None

            if face_match is not None:
                employee_id = face_match.employee_id
                method = "face"
                confidence = float(face_match.score)
                mem.resolved_employee_id = employee_id
                mem.resolved_employee_name = (face_match.employee_name or "").strip() or None
                mem.resolved_method = method
                mem.resolved_confidence = confidence
            else:
                try:
                    reid_match = self.reid_recognizer.identify_from_buffer(
                        mem.body_buffer,
                        state.reid_matcher,
                        threshold=self.cfg.inference.reid_threshold,
                        top_k_frames=3,
                    )
                except Exception:
                    reid_match = None
                if reid_match is not None:
                    employee_id = reid_match.employee_id
                    method = "reid"
                    confidence = float(reid_match.score)
                    mem.resolved_employee_id = employee_id
                    mem.resolved_employee_name = (reid_match.employee_name or "").strip() or None
                    mem.resolved_method = method
                    mem.resolved_confidence = confidence

            event_ts = mem.event_ts_utc or datetime.now(timezone.utc)
            mem.event_ts_utc = event_ts
            if employee_id is not None:
                d = self.runtime_dedup.consider(employee_id, event_ts)
                if not d.keep_new:
                    mem.posted = True
                    mem.finalized = True
                    continue

            snap_path = None
            if self.save_snapshots:
                snap_path = self._save_snapshot(frame, trk_id, method, event_ts)

            payload = {
                "employee_id": employee_id,
                "ts": event_ts.isoformat(),
                "method": method,
                "confidence": float(max(0.0, min(1.0, confidence))),
                "camera_id": self.camera_id,
                "track_uid": f"{self.camera_id}-{trk_id}-{int(mem.first_seen_ts * 1000)}",
                "image_path": snap_path,
                "meta": {
                    "inside_morning_window": is_within_morning_window(
                        event_ts,
                        self.cfg.roi.morning_window_start,
                        self.cfg.roi.morning_window_end,
                    )
                },
            }
            try:
                self.backend.post_event(payload)
                mem.posted = True
                mem.finalized = True
                mem.last_post_error = None
            except Exception as exc:
                mem.posted = False
                mem.finalized = False
                mem.post_attempts += 1
                mem.last_post_attempt_ts = now_ts
                err_msg = f"{type(exc).__name__}: {exc}"
                if err_msg != mem.last_post_error:
                    print(f"[events] failed to post track={trk_id} attempt={mem.post_attempts}: {err_msg}")
                mem.last_post_error = err_msg

    def _save_snapshot(self, frame: np.ndarray, track_id: int, method: str, ts: datetime) -> str | None:
        try:
            day_dir = self.snapshot_dir / ts.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{ts.strftime('%H%M%S')}_{self.camera_id}_{track_id}_{method}.jpg"
            path = day_dir / fname
            cv2.imwrite(str(path), frame)
            return str(path)
        except Exception:
            return None

    def _cleanup_tracks(self, now_ts: float) -> None:
        stale_ids = [tid for tid, mem in self.track_mem.items() if now_ts - mem.last_seen_ts > 4.0]
        for tid in stale_ids:
            self.track_mem.pop(tid, None)

    def _draw_overlays(self, frame: np.ndarray, tracks: list[Track], fps: float) -> None:
        poly = np.array(self.cfg.roi.roi_polygon, dtype=np.int32)
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
        p1 = tuple(int(v) for v in self.cfg.roi.entry_line_p1)
        p2 = tuple(int(v) for v in self.cfg.roi.entry_line_p2)
        cv2.line(frame, p1, p2, (255, 0, 255), 2)
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            mem = self.track_mem.get(tr.track_id)
            color = (0, 255, 0) if mem and mem.crossed else (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if mem and mem.resolved_employee_name:
                label = mem.resolved_employee_name
            elif mem and mem.live_employee_name:
                label = mem.live_employee_name
            elif mem and mem.resolved_employee_id is not None:
                label = f"Employee {mem.resolved_employee_id}"
            elif mem and mem.live_employee_id is not None:
                label = f"Employee {mem.live_employee_id}"
            elif mem and mem.finalized:
                label = "Unknown"
            else:
                label = "Detecting..."
            method = mem.resolved_method if mem and mem.resolved_method else (mem.live_method if mem else None)
            score = (
                mem.resolved_confidence
                if mem and mem.resolved_method
                else (mem.live_confidence if mem and mem.live_method else 0.0)
            )
            if method:
                label += f" ({method}:{score:.2f})"
            if mem and mem.finalized:
                label += " [posted]" if mem.posted else " [finalized]"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"FPS {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def _maybe_enroll(self, frame: np.ndarray, tracks: list[Track]) -> None:
        now = time.time()
        if now - self._last_enroll_upload < 1.5:
            return
        if not tracks:
            return
        best = max(tracks, key=lambda t: self._bbox_area(t.bbox))
        crop = self._crop(frame, best.bbox)
        if crop.size == 0:
            return
        image = frame if self.enroll_kind == "face" else crop
        try:
            self.backend.upload_enroll_frame(self.enroll_employee_id or 0, self.enroll_kind, image)
            self._last_enroll_upload = now
        except Exception:
            pass
