from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ROIConfig:
    roi_polygon: list[list[float]]
    entry_line_p1: tuple[float, float]
    entry_line_p2: tuple[float, float]
    morning_window_start: str
    morning_window_end: str
    snapshot_retention_days: int


@dataclass
class InferenceSettings:
    camera_id: str
    yolo_model: str
    face_model_dir: str
    face_model_name: str
    reid_model_path: str
    use_gpu_if_available: bool
    face_threshold: float
    reid_threshold: float
    face_top_k_frames: int
    face_buffer_size: int
    gallery_refresh_seconds: int
    snapshot_dir: str
    save_snapshots_default: bool
    line_cross_direction: str
    min_box_area: int


@dataclass
class RuntimeConfig:
    repo_root: Path
    inference: InferenceSettings
    roi: ROIConfig


def load_runtime_config(repo_root: Path, roi_config_path: str | Path) -> RuntimeConfig:
    app_cfg = yaml.safe_load((repo_root / "config" / "app.yaml").read_text(encoding="utf-8")) or {}
    roi_cfg = yaml.safe_load(Path(roi_config_path).read_text(encoding="utf-8")) or {}
    inf = app_cfg.get("inference") or {}
    inference = InferenceSettings(
        camera_id=str(inf.get("camera_id", "entrance-1")),
        yolo_model=str(inf.get("yolo_model", "models/yolo/yolov8n.pt")),
        face_model_dir=str(inf.get("face_model_dir", "models/insightface")),
        face_model_name=str(inf.get("face_model_name", "buffalo_l")),
        reid_model_path=str(inf.get("reid_model_path", "models/reid/mobilenet_v3_small-047dcff4.pth")),
        use_gpu_if_available=bool(inf.get("use_gpu_if_available", True)),
        face_threshold=float(inf.get("face_threshold", 0.42)),
        reid_threshold=float(inf.get("reid_threshold", 0.78)),
        face_top_k_frames=int(inf.get("face_top_k_frames", 3)),
        face_buffer_size=int(inf.get("face_buffer_size", 12)),
        gallery_refresh_seconds=int(inf.get("gallery_refresh_seconds", 30)),
        snapshot_dir=str(inf.get("snapshot_dir", "data/snapshots")),
        save_snapshots_default=bool(inf.get("save_snapshots_default", False)),
        line_cross_direction=str(inf.get("line_cross_direction", "into_roi")),
        min_box_area=int(inf.get("min_box_area", 2500)),
    )
    entry_line = roi_cfg.get("entry_line") or {"p1": [250, 420], "p2": [1050, 420]}
    mw = roi_cfg.get("morning_window") or {"start": "04:00", "end": "11:30"}
    roi = ROIConfig(
        roi_polygon=[list(map(float, p)) for p in (roi_cfg.get("roi_polygon") or [[100, 100], [1180, 100], [1180, 680], [100, 680]])],
        entry_line_p1=(float(entry_line["p1"][0]), float(entry_line["p1"][1])),
        entry_line_p2=(float(entry_line["p2"][0]), float(entry_line["p2"][1])),
        morning_window_start=str(mw.get("start", "04:00")),
        morning_window_end=str(mw.get("end", "11:30")),
        snapshot_retention_days=int(roi_cfg.get("snapshot_retention_days", 7)),
    )
    return RuntimeConfig(repo_root=repo_root, inference=inference, roi=roi)
