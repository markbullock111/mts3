from __future__ import annotations

import csv
import io
import json
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import and_, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from .config import settings as app_settings
from .db import SessionLocal, engine, get_db
from .embedding_extractors import FaceEmbedder, ReIDEmbedder
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


@app.on_event("startup")
def startup() -> None:
    defaults = load_defaults_from_yaml(app_settings.repo_root)
    db = SessionLocal()
    try:
        ensure_default_settings(db, defaults)
    finally:
        db.close()


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
        _store_employee_uploaded_image(db, employee_id=employee_id, kind="face", upload=upload, raw_bytes=raw)
        saved_images += 1
        db.add(EmployeeFaceEmbedding(employee_id=employee_id, embedding_vector=best.embedding.tolist()))
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
        _store_employee_uploaded_image(db, employee_id=employee_id, kind="reid", upload=upload, raw_bytes=raw)
        saved_images += 1
        db.add(EmployeeReIDEmbedding(employee_id=employee_id, embedding_vector=emb.tolist()))
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


if _DATA_DIR.exists():
    app.mount("/media", StaticFiles(directory=str(_DATA_DIR)), name="media")

if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")
