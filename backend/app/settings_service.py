from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Setting


DEFAULT_SETTINGS: dict[str, Any] = {
    "face_threshold": 0.42,
    "reid_threshold": 0.78,
    "standard_attendance_time": "09:00",
    "attendance_today_override_date": "",
    "attendance_today_override_time": "",
    "morning_window_start": "04:00",
    "morning_window_end": "11:30",
    "roi_polygon": [[100, 100], [1180, 100], [1180, 680], [100, 680]],
    "entry_line": {"p1": [250, 420], "p2": [1050, 420]},
    "snapshot_retention_days": 7,
    "save_snapshots_default": False,
    "gallery_version": 1,
}


def _encode(value: Any) -> str:
    return json.dumps(value)


def _decode(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_defaults_from_yaml(repo_root: Path) -> dict[str, Any]:
    roi_path = repo_root / "config" / "roi.yaml"
    if not roi_path.exists():
        return DEFAULT_SETTINGS.copy()
    data = yaml.safe_load(roi_path.read_text(encoding="utf-8")) or {}
    merged = DEFAULT_SETTINGS.copy()
    if "morning_window" in data:
        mw = data["morning_window"] or {}
        merged["morning_window_start"] = mw.get("start", merged["morning_window_start"])
        merged["morning_window_end"] = mw.get("end", merged["morning_window_end"])
    if "roi_polygon" in data:
        merged["roi_polygon"] = data["roi_polygon"]
    if "entry_line" in data:
        merged["entry_line"] = data["entry_line"]
    if "snapshot_retention_days" in data:
        merged["snapshot_retention_days"] = data["snapshot_retention_days"]
    return merged


def ensure_default_settings(db: Session, defaults: dict[str, Any]) -> None:
    existing = {row.key for row in db.execute(select(Setting)).scalars().all()}
    changed = False
    for key, value in defaults.items():
        if key not in existing:
            db.add(Setting(key=key, value=_encode(value)))
            changed = True
    if changed:
        db.commit()


def get_settings_dict(db: Session) -> dict[str, Any]:
    rows = db.execute(select(Setting).order_by(Setting.key)).scalars().all()
    return {r.key: _decode(r.value) for r in rows}


def upsert_settings(db: Session, values: dict[str, Any], bump_gallery: bool = False) -> dict[str, Any]:
    rows = {r.key: r for r in db.execute(select(Setting)).scalars().all()}
    for key, value in values.items():
        if key in rows:
            rows[key].value = _encode(value)
        else:
            db.add(Setting(key=key, value=_encode(value)))
    if bump_gallery:
        current = rows.get("gallery_version")
        if current is not None:
            try:
                v = int(_decode(current.value)) + 1
            except Exception:
                v = 1
            current.value = _encode(v)
        else:
            db.add(Setting(key="gallery_version", value=_encode(1)))
    db.commit()
    return get_settings_dict(db)


def bump_gallery_version(db: Session) -> int:
    data = upsert_settings(db, {}, bump_gallery=True)
    return int(data.get("gallery_version", 1))
