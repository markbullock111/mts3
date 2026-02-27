from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import time as pytime
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import and_, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from .config import settings as app_settings
from .db import SessionLocal, engine, get_db
from .embedding_extractors import FaceEmbedder, ReIDEmbedder
from inference.detector import YOLOPersonDetector
from inference.matcher import EmbeddingMatcher
from inference.tracker import build_tracker
from .models import (
    AttendanceEvent,
    AttendanceMethod,
    Camera,
    Employee,
    EmployeeFaceEmbedding,
    EmployeeReIDEmbedding,
    EmployeeUploadedImage,
    EmployeeStatus,
)
from .schemas import EmployeeCreate, EmployeeUpdate, EventCreate, EventOverride, SettingsUpdate
from .settings_service import bump_gallery_version, ensure_default_settings, get_settings_dict, load_defaults_from_yaml, upsert_settings

app = FastAPI(title="Offline Attendance Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.ui_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_DIR = app_settings.repo_root / "ui"
_DATA_DIR = app_settings.repo_root / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_EMPLOYEE_UPLOADS_DIR = _DATA_DIR / "employee_uploads"
_EMPLOYEE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_face_embedder: FaceEmbedder | None = None
_reid_embedder: ReIDEmbedder | None = None
_preview_workers: dict[int, "CameraPreviewWorker"] = {}
_preview_workers_lock = threading.Lock()


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD") from exc


def _day_bounds_utc(d: date) -> tuple[datetime, datetime]:
    start = datetime.combine(d, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def _date_range_bounds_utc(date_from: date, date_to: date) -> tuple[datetime, datetime]:
    if date_to < date_from:
        raise HTTPException(status_code=400, detail="date_to must be >= date_from")
    start = datetime.combine(date_from, time.min, tzinfo=timezone.utc)
    end = datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return start, end


def _current_week_dates_utc(now: datetime | None = None) -> tuple[date, date]:
    now = now or datetime.now(timezone.utc)
    d = now.astimezone(timezone.utc).date()
    start = d - timedelta(days=d.weekday())  # Monday
    return start, d


def _media_url_from_data_rel_path(rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    rel = str(rel_path).replace("\\", "/").lstrip("/")
    return f"/media/{rel}"


def _event_image_url_from_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    try:
        p = Path(path_str)
        if p.is_absolute():
            p_rel = p.relative_to(_DATA_DIR)
            return _media_url_from_data_rel_path(str(p_rel))
        # already relative to data/
        return _media_url_from_data_rel_path(path_str)
    except Exception:
        return None


def _event_to_dict(evt: AttendanceEvent) -> dict[str, Any]:
    emp = evt.employee
    return {
        "id": evt.id,
        "employee_id": evt.employee_id,
        "ts": evt.ts,
        "method": evt.method.value if hasattr(evt.method, "value") else str(evt.method),
        "confidence": float(evt.confidence),
        "camera_id": evt.camera_id,
        "track_uid": evt.track_uid,
        "image_path": evt.image_path,
        "image_url": _event_image_url_from_path(evt.image_path),
        "employee_name": emp.full_name if emp else None,
        "employee_code": emp.employee_code if emp else None,
    }


def _read_upload_image(upload: UploadFile) -> tuple[bytes, np.ndarray]:
    raw = upload.file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}")
    return raw, img


def _safe_filename_component(name: str, fallback: str = "image") -> str:
    base = Path(name or fallback).stem or fallback
    base = re.sub(r"[^\w.-]+", "_", base, flags=re.UNICODE).strip("._")
    return base[:80] or fallback


def _safe_extension(name: str | None) -> str:
    ext = (Path(name or "").suffix or "").lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return ext
    return ".jpg"


def _store_employee_uploaded_image(
    db: Session,
    employee_id: int,
    kind: str,
    upload: UploadFile,
    raw_bytes: bytes,
) -> EmployeeUploadedImage:
    ts = datetime.now(timezone.utc)
    kind_dir = "reid" if kind == "reid" else "face"
    ext = _safe_extension(upload.filename)
    base = _safe_filename_component(upload.filename or "image")
    file_name = f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}_{base}{ext}"
    rel_path = Path("employee_uploads") / str(employee_id) / kind_dir / file_name
    abs_path = _DATA_DIR / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(raw_bytes)
    row = EmployeeUploadedImage(
        employee_id=employee_id,
        kind=kind_dir,
        original_filename=upload.filename,
        file_path=rel_path.as_posix(),
    )
    db.add(row)
    return row


def _delete_data_file(rel_path: str | None) -> None:
    if not rel_path:
        return
    try:
        base = _DATA_DIR.resolve()
        abs_path = (base / rel_path).resolve()
        abs_path.relative_to(base)
    except Exception:
        return
    try:
        if abs_path.is_file():
            abs_path.unlink()
    except Exception:
        return
    # best-effort cleanup of empty directories up to data/
    for parent in abs_path.parents:
        if parent == base:
            break
        try:
            parent.rmdir()
        except OSError:
            break


def _employee_photo_to_dict(row: EmployeeUploadedImage) -> dict[str, Any]:
    return {
        "id": row.id,
        "kind": row.kind,
        "original_filename": row.original_filename,
        "file_path": row.file_path,
        "media_url": _media_url_from_data_rel_path(row.file_path),
        "created_at": row.created_at,
    }


def _employee_to_dict(emp: Employee, include_lists: bool = False) -> dict[str, Any]:
    data = {
        "id": emp.id,
        "full_name": emp.full_name,
        "employee_code": emp.employee_code,
        "birth_date": emp.birth_date,
        "job_title": emp.job_title,
        "address": emp.address,
        "status": emp.status.value if hasattr(emp.status, "value") else str(emp.status),
        "created_at": emp.created_at,
        "face_embeddings_count": len(getattr(emp, "face_embeddings", []) or []),
        "reid_embeddings_count": len(getattr(emp, "reid_embeddings", []) or []),
        "uploaded_images_count": len(getattr(emp, "uploaded_images", []) or []),
    }
    if include_lists:
        images = sorted(list(getattr(emp, "uploaded_images", []) or []), key=lambda r: (r.created_at, r.id), reverse=True)
        data["uploaded_images"] = [_employee_photo_to_dict(r) for r in images]
    return data


def get_face_embedder() -> FaceEmbedder:
    global _face_embedder
    if _face_embedder is None:
        cfg_path = app_settings.repo_root / "config" / "app.yaml"
        use_gpu = True
        model_dir = app_settings.repo_root / "models" / "insightface"
        model_name = "buffalo_l"
        if cfg_path.exists():
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            inf = (cfg.get("inference") or {})
            use_gpu = bool(inf.get("use_gpu_if_available", True))
            model_dir = app_settings.repo_root / str(inf.get("face_model_dir", "models/insightface"))
            model_name = str(inf.get("face_model_name", "buffalo_l"))
        try:
            _face_embedder = FaceEmbedder(model_root=model_dir, model_name=model_name, use_gpu=use_gpu)
        except Exception:
            _face_embedder = FaceEmbedder(model_root=model_dir, model_name=model_name, use_gpu=False)
    return _face_embedder


def get_reid_embedder() -> ReIDEmbedder:
    global _reid_embedder
    if _reid_embedder is None:
        cfg_path = app_settings.repo_root / "config" / "app.yaml"
        use_gpu = False
        weights_path = app_settings.repo_root / "models" / "reid" / "mobilenet_v3_small-047dcff4.pth"
        if cfg_path.exists():
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            inf = (cfg.get("inference") or {})
            use_gpu = bool(inf.get("use_gpu_if_available", True))
            weights_path = app_settings.repo_root / str(inf.get("reid_model_path", "models/reid/mobilenet_v3_small-047dcff4.pth"))
        device = "cuda" if use_gpu and _torch_cuda_available() else "cpu"
        _reid_embedder = ReIDEmbedder(weights_path=weights_path, device=device)
    return _reid_embedder


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _color_from_track_id(track_id: int) -> tuple[int, int, int]:
    # Deterministic, high-contrast palette for multi-person overlays.
    palette = [
        (255, 56, 56),
        (56, 255, 56),
        (56, 56, 255),
        (255, 196, 56),
        (255, 56, 196),
        (56, 224, 255),
        (180, 56, 255),
        (56, 255, 170),
        (255, 122, 56),
    ]
    return palette[track_id % len(palette)]


class CameraPreviewWorker:
    def __init__(self, camera_id: int, camera_name: str, rtsp_url: str):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._last_frame_ts = 0.0
        self._last_gallery_refresh = 0.0
        self._face_threshold = 0.42
        self._matcher: EmbeddingMatcher | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-preview-{camera_id}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _load_runtime_cfg(self) -> tuple[Path, bool]:
        cfg_path = app_settings.repo_root / "config" / "app.yaml"
        yolo_model = app_settings.repo_root / "models" / "yolo" / "yolov8n.pt"
        use_gpu = False
        if cfg_path.exists():
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            inf = cfg.get("inference") or {}
            yolo_model = app_settings.repo_root / str(inf.get("yolo_model", "models/yolo/yolov8n.pt"))
            use_gpu = bool(inf.get("use_gpu_if_available", False))
        return yolo_model, use_gpu and _torch_cuda_available()

    def _refresh_face_gallery(self) -> None:
        now = pytime.time()
        if now - self._last_gallery_refresh < 20:
            return
        self._last_gallery_refresh = now
        db = SessionLocal()
        try:
            settings_map = get_settings_dict(db)
            try:
                self._face_threshold = float(settings_map.get("face_threshold", 0.42))
            except Exception:
                self._face_threshold = 0.42
            rows = (
                db.execute(
                    select(EmployeeFaceEmbedding, Employee)
                    .join(Employee, Employee.id == EmployeeFaceEmbedding.employee_id)
                    .where(Employee.status == EmployeeStatus.active)
                )
                .all()
            )
            face_rows = [
                {
                    "embedding_id": emb.id,
                    "employee_id": emp.id,
                    "employee_code": emp.employee_code,
                    "employee_name": emp.full_name,
                    "embedding": emb.embedding_vector,
                }
                for emb, emp in rows
                if emb.embedding_vector
            ]
            self._matcher = EmbeddingMatcher(face_rows, prefer_faiss=False)
        finally:
            db.close()

    def _put_latest_jpeg(self, data: bytes) -> None:
        with self._lock:
            self._latest_jpeg = data
            self._last_frame_ts = pytime.time()

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    @staticmethod
    def _parse_camera_source(source: str) -> str | int:
        src = str(source or "").strip()
        low = src.lower()
        if low in {"webcam", "camera", "cam"}:
            return 0
        if ":" in low:
            prefix, idx = low.split(":", 1)
            if prefix in {"webcam", "camera", "cam"}:
                try:
                    return int(idx.strip())
                except Exception:
                    return 0
        if low.lstrip("-").isdigit():
            try:
                return int(low)
            except Exception:
                pass
        return src

    @staticmethod
    def _configure_webcam_capture(cap: cv2.VideoCapture) -> None:
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
    def _backend_name(backend_choice: int | None | str) -> str:
        if backend_choice == cv2.CAP_DSHOW:
            return "CAP_DSHOW"
        if backend_choice == cv2.CAP_MSMF:
            return "CAP_MSMF"
        if backend_choice is None:
            return "DEFAULT"
        return str(backend_choice)

    @staticmethod
    def _open_capture(source: str | int, backend_choice: int | None | str = "auto") -> cv2.VideoCapture:
        if isinstance(source, int):
            if os.name == "nt" and backend_choice != "auto":
                cap = cv2.VideoCapture(source) if backend_choice is None else cv2.VideoCapture(source, backend_choice)
                if cap.isOpened():
                    CameraPreviewWorker._configure_webcam_capture(cap)
                    CameraPreviewWorker._warmup_capture(cap)
                return cap
            if os.name == "nt":
                for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, None):
                    cap = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
                    if cap.isOpened():
                        CameraPreviewWorker._configure_webcam_capture(cap)
                        CameraPreviewWorker._warmup_capture(cap)
                        return cap
                    cap.release()
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                CameraPreviewWorker._configure_webcam_capture(cap)
                CameraPreviewWorker._warmup_capture(cap)
            return cap

        s = str(source).strip()
        if s.lower().startswith("rtsp://"):
            cap = cv2.VideoCapture(s, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return cv2.VideoCapture(s)

    def _publish_status_frame(self, title: str, detail: str = "") -> None:
        canvas = np.zeros((360, 760, 3), dtype=np.uint8)
        canvas[:, :] = (28, 27, 25)
        cv2.putText(canvas, title[:80], (24, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 210, 255), 2, cv2.LINE_AA)
        if detail:
            cv2.putText(canvas, detail[:100], (24, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Use RTSP URL or webcam source (webcam, webcam:0, 0, 1)",
            (24, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(canvas, f"Camera ID {self.camera_id}: {self.camera_name}", (24, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 180, 180), 2, cv2.LINE_AA)
        ok_enc, enc = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok_enc:
            self._put_latest_jpeg(enc.tobytes())

    def _run(self) -> None:
        try:
            yolo_model, use_gpu = self._load_runtime_cfg()
            detector = YOLOPersonDetector(model_path=yolo_model, device=("0" if use_gpu else None), conf=0.25)
            tracker = build_tracker(prefer_bytetrack=True, fps=25)
            face_embedder = FaceEmbedder(
                model_root=app_settings.repo_root / "models" / "insightface",
                model_name="buffalo_l",
                use_gpu=use_gpu,
            )
        except Exception as exc:
            err_frame = np.zeros((320, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "Preview init failed", (18, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(err_frame, str(exc)[:70], (18, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2, cv2.LINE_AA)
            ok_enc, enc = cv2.imencode(".jpg", err_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok_enc:
                self._put_latest_jpeg(enc.tobytes())
            while not self._stop.is_set():
                pytime.sleep(0.5)
            return
        labels_by_track: dict[int, tuple[str, float]] = {}
        source = self._parse_camera_source(self.rtsp_url)
        webcam_backends: list[int | None | str] = ["auto"]
        backend_idx = 0
        if isinstance(source, int) and os.name == "nt":
            webcam_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

        while not self._stop.is_set():
            backend_choice = webcam_backends[backend_idx] if webcam_backends else "auto"
            cap = self._open_capture(source, backend_choice=backend_choice)
            if not cap.isOpened():
                self._publish_status_frame(
                    "Unable to open camera source",
                    f"{self.rtsp_url} ({self._backend_name(backend_choice)})",
                )
                if len(webcam_backends) > 1:
                    backend_idx = (backend_idx + 1) % len(webcam_backends)
                pytime.sleep(1.0)
                continue

            read_failures = 0
            blank_failures = 0

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures >= 20:
                        self._publish_status_frame("Camera stream read failed", str(self.rtsp_url))
                        pytime.sleep(0.2)
                        break
                    pytime.sleep(0.03)
                    continue
                read_failures = 0
                if self._is_blank_frame(frame):
                    blank_failures += 1
                    if blank_failures >= 20:
                        self._publish_status_frame(
                            "Camera returned blank frames",
                            f"{self.rtsp_url} ({self._backend_name(backend_choice)})",
                        )
                        pytime.sleep(0.2)
                        break
                    pytime.sleep(0.02)
                    continue
                blank_failures = 0

                self._refresh_face_gallery()
                dets = detector.detect(frame)
                tracks = tracker.update(dets, frame)

                for tr in tracks:
                    x1, y1, x2, y2 = [int(v) for v in tr.bbox]
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    person_crop = frame[y1:y2, x1:x2]
                    label = labels_by_track.get(tr.track_id, (f"Track {tr.track_id}", 0.0))[0]
                    if person_crop.size > 0 and self._matcher is not None:
                        try:
                            faces = face_embedder.extract_from_bgr(person_crop)
                            if faces:
                                best_face = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]
                                match = self._matcher.best_match(best_face.embedding, min_score=self._face_threshold, top_k=5)
                                if match is not None:
                                    name = match.employee_name or match.employee_code or f"ID {match.employee_id}"
                                    label = f"{name} ({match.score:.2f})"
                                    labels_by_track[tr.track_id] = (label, pytime.time())
                                else:
                                    labels_by_track.setdefault(tr.track_id, (label, pytime.time()))
                        except Exception:
                            pass

                    color = _color_from_track_id(tr.track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # label background
                    text = label
                    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    ty = max(th + 6, y1 - 4)
                    tx2 = min(w - 1, x1 + tw + 10)
                    cv2.rectangle(frame, (x1, ty - th - 6), (tx2, ty + baseline - 4), color, -1)
                    cv2.putText(frame, text, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                # prune old labels
                now = pytime.time()
                stale = [tid for tid, (_, ts) in labels_by_track.items() if now - ts > 10]
                for tid in stale:
                    labels_by_track.pop(tid, None)

                cv2.putText(
                    frame,
                    f"{self.camera_name} | Camera ID {self.camera_id}",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                ok_enc, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_enc:
                    self._put_latest_jpeg(enc.tobytes())

            cap.release()
            if len(webcam_backends) > 1:
                backend_idx = (backend_idx + 1) % len(webcam_backends)

    def mjpeg_generator(self):
        boundary = b"--frame\r\n"
        while not self._stop.is_set():
            frame = self.get_latest_jpeg()
            if frame is None:
                pytime.sleep(0.05)
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            pytime.sleep(0.03)


def _get_or_create_preview_worker(camera: Camera) -> CameraPreviewWorker:
    with _preview_workers_lock:
        worker = _preview_workers.get(camera.id)
        if worker is not None and worker.is_alive():
            return worker
        worker = CameraPreviewWorker(camera_id=camera.id, camera_name=camera.name, rtsp_url=camera.rtsp_url)
        _preview_workers[camera.id] = worker
        return worker


def _stop_preview_worker(camera_id: int) -> bool:
    with _preview_workers_lock:
        worker = _preview_workers.pop(camera_id, None)
    if worker is None:
        return False
    try:
        worker.stop()
    except Exception:
        pass
    return True


@app.on_event("startup")
def startup() -> None:
    defaults = load_defaults_from_yaml(app_settings.repo_root)
    db = SessionLocal()
    try:
        ensure_default_settings(db, defaults)
    finally:
        db.close()


@app.on_event("shutdown")
def shutdown() -> None:
    with _preview_workers_lock:
        workers = list(_preview_workers.values())
        _preview_workers.clear()
    for worker in workers:
        try:
            worker.stop()
        except Exception:
            pass


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        return {"status": "degraded", "db": False, "error": str(exc)}
    return {"status": "ok", "db": db_ok}


@app.post("/employees")
def create_employee(payload: EmployeeCreate, db: Session = Depends(get_db)) -> dict[str, Any]:
    emp = Employee(
        full_name=payload.full_name,
        employee_code=payload.employee_code,
        birth_date=payload.birth_date,
        job_title=payload.job_title,
        address=payload.address,
        status=EmployeeStatus(payload.status),
    )
    db.add(emp)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Employee code already exists") from exc
    db.refresh(emp)
    return _employee_to_dict(emp)


@app.get("/employees")
def list_employees(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = (
        db.execute(
            select(Employee)
            .options(joinedload(Employee.face_embeddings), joinedload(Employee.reid_embeddings), joinedload(Employee.uploaded_images))
            .order_by(Employee.id.asc())
        )
        .unique()
        .scalars()
        .all()
    )
    return [_employee_to_dict(emp) for emp in rows]


@app.get("/employees/gallery")
def get_gallery(db: Session = Depends(get_db)) -> dict[str, Any]:
    settings_map = get_settings_dict(db)
    version = int(settings_map.get("gallery_version", 1))
    face_rows = (
        db.execute(
            select(EmployeeFaceEmbedding, Employee)
            .join(Employee, Employee.id == EmployeeFaceEmbedding.employee_id)
            .where(Employee.status == EmployeeStatus.active)
        )
        .all()
    )
    reid_rows = (
        db.execute(
            select(EmployeeReIDEmbedding, Employee)
            .join(Employee, Employee.id == EmployeeReIDEmbedding.employee_id)
            .where(Employee.status == EmployeeStatus.active)
        )
        .all()
    )
    face_embeddings = [
        {
            "embedding_id": emb.id,
            "employee_id": emp.id,
            "employee_code": emp.employee_code,
            "employee_name": emp.full_name,
            "embedding": emb.embedding_vector,
        }
        for emb, emp in face_rows
    ]
    reid_embeddings = [
        {
            "embedding_id": emb.id,
            "employee_id": emp.id,
            "employee_code": emp.employee_code,
            "employee_name": emp.full_name,
            "embedding": emb.embedding_vector,
        }
        for emb, emp in reid_rows
    ]
    return {"version": version, "face_embeddings": face_embeddings, "reid_embeddings": reid_embeddings}


@app.get("/employees/{employee_id}")
def get_employee(employee_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    emp = (
        db.execute(
            select(Employee)
            .options(
                joinedload(Employee.face_embeddings),
                joinedload(Employee.reid_embeddings),
                joinedload(Employee.uploaded_images),
            )
            .where(Employee.id == employee_id)
        )
        .unique()
        .scalars()
        .first()
    )
    if emp is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    week_start, week_end = _current_week_dates_utc()
    return {
        **_employee_to_dict(emp, include_lists=True),
        "history_default_date_from": week_start.isoformat(),
        "history_default_date_to": week_end.isoformat(),
    }


@app.put("/employees/{employee_id}")
def update_employee(employee_id: int, payload: EmployeeUpdate, db: Session = Depends(get_db)) -> dict[str, Any]:
    emp = db.get(Employee, employee_id)
    if emp is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    data = payload.model_dump(exclude_unset=True)
    if "full_name" in data:
        emp.full_name = str(data["full_name"])
    if "birth_date" in data:
        emp.birth_date = data["birth_date"]
    if "job_title" in data:
        emp.job_title = data["job_title"]
    if "address" in data:
        emp.address = data["address"]
    if "status" in data and data["status"] is not None:
        emp.status = EmployeeStatus(data["status"])
    db.commit()
    db.refresh(emp)
    return _employee_to_dict(emp)


@app.get("/employees/{employee_id}/photos")
def list_employee_photos(
    employee_id: int,
    kind: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if db.get(Employee, employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    q = select(EmployeeUploadedImage).where(EmployeeUploadedImage.employee_id == employee_id)
    if kind in {"face", "reid"}:
        q = q.where(EmployeeUploadedImage.kind == kind)
    rows = db.execute(q.order_by(EmployeeUploadedImage.created_at.desc(), EmployeeUploadedImage.id.desc())).scalars().all()
    return {"items": [_employee_photo_to_dict(r) for r in rows]}


@app.get("/employees/{employee_id}/attendance")
def list_employee_attendance_history(
    employee_id: int,
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    emp = db.get(Employee, employee_id)
    if emp is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    if date_from is None and date_to is None:
        d_from, d_to = _current_week_dates_utc()
    else:
        d_from = _parse_date(date_from) if date_from else _parse_date(date_to or "")
        d_to = _parse_date(date_to) if date_to else d_from
    start, end = _date_range_bounds_utc(d_from, d_to)

    rows = (
        db.execute(
            select(AttendanceEvent)
            .options(joinedload(AttendanceEvent.employee))
            .where(
                AttendanceEvent.employee_id == employee_id,
                AttendanceEvent.ts >= start,
                AttendanceEvent.ts < end,
            )
            .order_by(AttendanceEvent.ts.asc())
        )
        .scalars()
        .all()
    )
    return {
        "employee_id": employee_id,
        "employee_name": emp.full_name,
        "date_from": d_from.isoformat(),
        "date_to": d_to.isoformat(),
        "items": [_event_to_dict(r) for r in rows],
    }


@app.post("/employees/{employee_id}/enroll/face")
def enroll_face(
    employee_id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    emp = db.get(Employee, employee_id)
    if emp is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    embedder = get_face_embedder()
    inserted = 0
    detections = 0
    saved_images = 0
    for upload in files:
        raw, img = _read_upload_image(upload)
        faces = embedder.extract_from_bgr(img)
        if not faces:
            continue
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
        detections += len(faces)
        best = faces[0]
        img_row = _store_employee_uploaded_image(db, employee_id=employee_id, kind="face", upload=upload, raw_bytes=raw)
        db.flush()
        saved_images += 1
        db.add(
            EmployeeFaceEmbedding(
                employee_id=employee_id,
                source_image_id=img_row.id,
                embedding_vector=best.embedding.tolist(),
            )
        )
        inserted += 1
    if inserted == 0:
        raise HTTPException(status_code=400, detail="No face embeddings extracted from uploaded images")
    db.commit()
    total = db.scalar(select(func.count()).select_from(EmployeeFaceEmbedding).where(EmployeeFaceEmbedding.employee_id == employee_id)) or 0
    bump_gallery_version(db)
    return {
        "employee_id": employee_id,
        "inserted_embeddings": inserted,
        "detections_seen": detections,
        "uploaded_images_saved": saved_images,
        "total_face_embeddings": int(total),
        "meets_minimum_recommended": bool(total >= 5),
        "recommended_minimum": 5,
    }


@app.post("/employees/{employee_id}/enroll/reid")
def enroll_reid(
    employee_id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    emp = db.get(Employee, employee_id)
    if emp is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    embedder = get_reid_embedder()
    inserted = 0
    saved_images = 0
    for upload in files:
        raw, img = _read_upload_image(upload)
        emb = embedder.extract_from_bgr(img)
        img_row = _store_employee_uploaded_image(db, employee_id=employee_id, kind="reid", upload=upload, raw_bytes=raw)
        db.flush()
        saved_images += 1
        db.add(
            EmployeeReIDEmbedding(
                employee_id=employee_id,
                source_image_id=img_row.id,
                embedding_vector=emb.tolist(),
            )
        )
        inserted += 1
    if inserted == 0:
        raise HTTPException(status_code=400, detail="No ReID embeddings extracted from uploaded images")
    db.commit()
    total = db.scalar(select(func.count()).select_from(EmployeeReIDEmbedding).where(EmployeeReIDEmbedding.employee_id == employee_id)) or 0
    bump_gallery_version(db)
    return {
        "employee_id": employee_id,
        "inserted_embeddings": inserted,
        "uploaded_images_saved": saved_images,
        "total_reid_embeddings": int(total),
    }


def _dedup_known_event(db: Session, payload: EventCreate) -> tuple[AttendanceEvent | None, bool]:
    if payload.employee_id is None:
        return None, False
    ts = payload.ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    d = ts.astimezone(timezone.utc).date()
    start, end = _day_bounds_utc(d)
    existing = (
        db.execute(
            select(AttendanceEvent)
            .where(
                AttendanceEvent.employee_id == payload.employee_id,
                AttendanceEvent.ts >= start,
                AttendanceEvent.ts < end,
            )
            .order_by(AttendanceEvent.ts.asc())
        )
        .scalars()
        .first()
    )
    if existing is None:
        return None, False
    if existing.ts <= ts:
        return existing, True
    existing.ts = ts
    existing.method = AttendanceMethod(payload.method)
    existing.confidence = payload.confidence
    existing.camera_id = payload.camera_id
    existing.track_uid = payload.track_uid
    existing.image_path = payload.image_path
    db.commit()
    db.refresh(existing)
    return existing, True


@app.post("/events")
def create_event(payload: EventCreate, db: Session = Depends(get_db)) -> dict[str, Any]:
    payload_ts = payload.ts if payload.ts.tzinfo else payload.ts.replace(tzinfo=timezone.utc)
    payload = payload.model_copy(update={"ts": payload_ts})

    existing_track = (
        db.execute(
            select(AttendanceEvent).where(
                AttendanceEvent.camera_id == payload.camera_id,
                AttendanceEvent.track_uid == payload.track_uid,
                AttendanceEvent.ts == payload.ts,
            )
        )
        .scalars()
        .first()
    )
    if existing_track:
        return {"created": False, "deduplicated": True, "event": _event_to_dict(existing_track)}

    dedup_evt, was_dedup = _dedup_known_event(db, payload)
    if was_dedup and dedup_evt is not None:
        return {"created": False, "deduplicated": True, "event": _event_to_dict(dedup_evt)}

    evt = AttendanceEvent(
        employee_id=payload.employee_id,
        ts=payload.ts,
        method=AttendanceMethod(payload.method),
        confidence=payload.confidence,
        camera_id=payload.camera_id,
        track_uid=payload.track_uid,
        image_path=payload.image_path,
    )
    db.add(evt)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Duplicate event") from exc
    db.refresh(evt)
    return {"created": True, "deduplicated": False, "event": _event_to_dict(evt)}


@app.delete("/employees/{employee_id}/photos/{photo_id}")
def delete_employee_photo(employee_id: int, photo_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    photo = db.get(EmployeeUploadedImage, photo_id)
    if photo is None or photo.employee_id != employee_id:
        raise HTTPException(status_code=404, detail="Photo not found")

    embeddings_deleted = 0
    if photo.kind == "face":
        embeddings_deleted = (
            db.query(EmployeeFaceEmbedding)
            .filter(EmployeeFaceEmbedding.employee_id == employee_id, EmployeeFaceEmbedding.source_image_id == photo_id)
            .delete(synchronize_session=False)
        )
    elif photo.kind == "reid":
        embeddings_deleted = (
            db.query(EmployeeReIDEmbedding)
            .filter(EmployeeReIDEmbedding.employee_id == employee_id, EmployeeReIDEmbedding.source_image_id == photo_id)
            .delete(synchronize_session=False)
        )

    file_path = photo.file_path
    db.delete(photo)
    db.commit()
    _delete_data_file(file_path)

    if embeddings_deleted > 0:
        bump_gallery_version(db)

    return {
        "deleted": True,
        "employee_id": employee_id,
        "photo_id": photo_id,
        "kind": photo.kind,
        "embeddings_deleted": int(embeddings_deleted),
    }


@app.get("/events")
def list_events(date: str = Query(...), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    d = _parse_date(date)
    start, end = _day_bounds_utc(d)
    rows = (
        db.execute(
            select(AttendanceEvent)
            .options(joinedload(AttendanceEvent.employee))
            .where(AttendanceEvent.ts >= start, AttendanceEvent.ts < end)
            .order_by(AttendanceEvent.ts.asc())
        )
        .scalars()
        .all()
    )
    return [_event_to_dict(r) for r in rows]


@app.get("/reports/daily.csv")
def export_daily_csv(date: str = Query(...), db: Session = Depends(get_db)) -> Response:
    d = _parse_date(date)
    start, end = _day_bounds_utc(d)
    rows = (
        db.execute(
            select(AttendanceEvent)
            .options(joinedload(AttendanceEvent.employee))
            .where(AttendanceEvent.ts >= start, AttendanceEvent.ts < end)
            .order_by(AttendanceEvent.ts.asc())
        )
        .scalars()
        .all()
    )
    sio = io.StringIO()
    writer = csv.writer(sio)
    writer.writerow(["id", "ts", "employee_id", "employee_code", "employee_name", "method", "confidence", "camera_id", "track_uid", "image_path"])
    for r in rows:
        emp = r.employee
        writer.writerow(
            [
                r.id,
                r.ts.isoformat(),
                r.employee_id or "",
                emp.employee_code if emp else "",
                emp.full_name if emp else "",
                r.method.value if hasattr(r.method, "value") else str(r.method),
                f"{r.confidence:.4f}",
                r.camera_id,
                r.track_uid,
                r.image_path or "",
            ]
        )
    return Response(
        content=sio.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="attendance-{d.isoformat()}.csv"'},
    )


@app.post("/events/{event_id}/override")
def override_event(event_id: int, payload: EventOverride, db: Session = Depends(get_db)) -> dict[str, Any]:
    evt = db.get(AttendanceEvent, event_id)
    if evt is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if payload.employee_id is not None and db.get(Employee, payload.employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    if payload.employee_id is not None:
        d = evt.ts.astimezone(timezone.utc).date()
        start, end = _day_bounds_utc(d)
        existing = (
            db.execute(
                select(AttendanceEvent)
                .where(
                    AttendanceEvent.employee_id == payload.employee_id,
                    AttendanceEvent.id != evt.id,
                    AttendanceEvent.ts >= start,
                    AttendanceEvent.ts < end,
                )
                .order_by(AttendanceEvent.ts.asc())
            )
            .scalars()
            .first()
        )
        if existing and existing.ts <= evt.ts:
            raise HTTPException(
                status_code=409,
                detail=f"Employee already has earlier event on {d.isoformat()} (event_id={existing.id})",
            )
        if existing and existing.ts > evt.ts:
            db.delete(existing)
            db.flush()

    evt.employee_id = payload.employee_id
    db.commit()
    db.refresh(evt)
    return {"updated": True, "event": _event_to_dict(evt)}


@app.get("/settings")
def get_settings(db: Session = Depends(get_db)) -> dict[str, Any]:
    return {"values": get_settings_dict(db)}


@app.put("/settings")
def put_settings(payload: SettingsUpdate, db: Session = Depends(get_db)) -> dict[str, Any]:
    gallery_keys = {"face_threshold", "reid_threshold"}
    bump = any(k in gallery_keys for k in payload.values.keys())
    values = upsert_settings(db, payload.values, bump_gallery=bump)
    return {"values": values}


@app.get("/cameras")
def list_cameras(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.execute(select(Camera).order_by(Camera.id.asc())).scalars().all()
    return [
        {"id": c.id, "name": c.name, "rtsp_url": c.rtsp_url, "location": c.location, "enabled": c.enabled}
        for c in rows
    ]


@app.post("/cameras")
def create_camera(name: str = Form(...), rtsp_url: str = Form(...), location: str = Form(default=""), enabled: bool = Form(default=True), db: Session = Depends(get_db)) -> dict[str, Any]:
    cam = Camera(name=name, rtsp_url=rtsp_url, location=location or None, enabled=enabled)
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return {"id": cam.id, "name": cam.name, "rtsp_url": cam.rtsp_url, "location": cam.location, "enabled": cam.enabled}


@app.delete("/cameras/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    cam = db.get(Camera, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    db.delete(cam)
    db.commit()
    _stop_preview_worker(camera_id)
    return {"deleted": True, "camera_id": camera_id}


@app.get("/cameras/{camera_id}/preview.mjpeg")
def camera_preview_stream(camera_id: int, db: Session = Depends(get_db)) -> StreamingResponse:
    cam = db.get(Camera, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not cam.enabled:
        raise HTTPException(status_code=400, detail="Camera is disabled")
    worker = _get_or_create_preview_worker(cam)
    return StreamingResponse(
        worker.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


if _DATA_DIR.exists():
    app.mount("/media", StaticFiles(directory=str(_DATA_DIR)), name="media")

if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")
