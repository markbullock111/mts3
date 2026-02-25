from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_runtime_config
from .pipeline import AttendancePipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline attendance inference runtime")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", help="Use webcam. Set to 'webcam' or webcam index like 0")
    src.add_argument("--rtsp", help="RTSP URL")
    p.add_argument("--backend", default="http://127.0.0.1:8000")
    p.add_argument("--show", action="store_true")
    p.add_argument("--roi-config", default="config/roi.yaml")
    p.add_argument("--save-snapshots", type=int, choices=[0, 1], default=None)
    p.add_argument("--enroll-employee-id", type=int, default=None, help="Optional capture mode: upload frames to enrollment endpoint")
    p.add_argument("--enroll-kind", choices=["face", "reid"], default="face")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_runtime_config(repo_root, args.roi_config)
    source: str | int
    if args.rtsp:
        source = args.rtsp
    else:
        if args.camera == "webcam":
            source = 0
        else:
            try:
                source = int(args.camera)
            except (TypeError, ValueError):
                source = args.camera
    pipeline = AttendancePipeline(
        cfg=cfg,
        backend_url=args.backend.rstrip("/"),
        camera_source=source,
        show=bool(args.show),
        save_snapshots=(None if args.save_snapshots is None else bool(args.save_snapshots)),
        enroll_employee_id=args.enroll_employee_id,
        enroll_kind=args.enroll_kind,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
