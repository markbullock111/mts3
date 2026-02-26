# Offline Attendance System (Windows, No Docker)

Offline attendance system for a single entrance using YOLO person detection, tracking, InsightFace face recognition, and ReID fallback. Designed for Windows 10/11 with a local PC acting as server.

Python target: `3.13` (supported via version-specific dependency pins in `requirements-base.txt`).

## Repo Layout
- `backend/` FastAPI + PostgreSQL + Alembic + static UI serving
- `inference/` Camera pipeline (RTSP/webcam) and recognition pipeline
- `ui/` Minimal admin UI (HTML/CSS/JS)
- `scripts/` Windows setup/run scripts and model downloader
  - includes `import_employees_from_folders.py` for bulk enrollment from `one-folder-per-person`
- `config/` YAML runtime defaults (ROI, thresholds)
- `docs/` Windows setup guide
- `tests/` Unit tests for matching and event logic
- `models/` Local model files (download once; runtime offline)

See `docs/SETUP_WINDOWS.md` for complete setup instructions.

## Bulk Folder Import (Employees)

Use a local folder tree where each subfolder name is the employee name (Unicode/Asian names supported), then run:

```powershell
python scripts\import_employees_from_folders.py --root "D:\attendance_import" --backend http://127.0.0.1:8000 --kind face --existing use
```
