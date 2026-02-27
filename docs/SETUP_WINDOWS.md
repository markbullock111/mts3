# Windows Setup (No Docker, Offline Runtime)

This guide installs and runs the attendance system on Windows 10/11 using a local PostgreSQL server and a Python virtual environment. Runtime is fully offline after the one-time model download step.

## 1) Prerequisites

- Windows 10 or 11
- Python 3.13 (recommended). Python 3.10+ still works, but this repo now includes Python-version-specific pins for 3.13.
- PostgreSQL 14+ installed locally
- Visual C++ Build Tools may be needed for some Python packages on some machines
- Optional NVIDIA GPU (for faster inference)

## 2) Clone / open repo

Use this repo folder locally (no Docker required).

## 3) Create database (Postgres)

PowerShell:

```powershell
psql -U postgres -h 127.0.0.1 -c "CREATE DATABASE attendance_db;"
```

cmd.exe:

```cmd
psql -U postgres -h 127.0.0.1 -c "CREATE DATABASE attendance_db;"
```

If your password is not `postgres`, update `.env` with your local Postgres credentials.

## 4) Python environment (choose ONE mode)

### A. CPU-only install (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-cpu.txt
```

### A. CPU-only install (cmd.exe)

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-cpu.txt
```

### B. NVIDIA GPU install (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-base.txt
pip install onnxruntime-gpu==1.20.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### B. NVIDIA GPU install (cmd.exe)

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-base.txt
pip install onnxruntime-gpu==1.20.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Notes:

- If your CUDA runtime/driver differs, install the matching PyTorch wheel from the official PyTorch index.
- `insightface` uses `onnxruntime` / `onnxruntime-gpu` providers depending on availability.
- Default ReID fallback model is a lightweight Torch MobileNetV3 embedding model (\"OSNet or similar\" requirement).
- `requirements-base.txt` includes Python-version-specific pins so Python 3.13 uses a newer `numpy` and `torch/torchvision` pair automatically.

## 5) One-time model download (internet required only for this step)

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\download_models.py
```

cmd.exe:

```cmd
.venv\Scripts\activate.bat
python scripts\download_models.py
```

Model files are stored under `models/`. Runtime must remain offline after this step.

## 6) Configure environment

Create or edit `.env` in the repo root and adjust Postgres credentials if needed.

Example:

```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/attendance_db
ATTENDANCE_TIMEZONE=UTC
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
UI_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

## 7) Database migration (Alembic)

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python alembic -c backend\alembic.ini upgrade head
```

cmd.exe:

```cmd
.venv\Scripts\activate.bat
set PYTHONPATH=.
python alembic -c backend\alembic.ini upgrade head
```

## 8) Run backend (FastAPI + UI)

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

cmd.exe:

```cmd
.venv\Scripts\activate.bat
set PYTHONPATH=.
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open UI: `http://127.0.0.1:8000/ui/`

UI notes:

- `Employees` tab: add employee with `Full Name`, `Birth`, `Job`, `Address`
- `Employee Details` tab: view/edit profile, see uploaded enrollment pictures, and view entrance-time history (defaults to this week, selectable date range)

## 9) Run inference

### Webcam (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python -m inference.run --camera webcam --backend http://127.0.0.1:8000 --roi-config config\roi.yaml --show
```

### Webcam (cmd.exe)

```cmd
.venv\Scripts\activate.bat
set PYTHONPATH=.
python -m inference.run --camera webcam --backend http://127.0.0.1:8000 --roi-config config\roi.yaml --show
```

### RTSP camera (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python -m inference.run --rtsp "rtsp://user:pass@camera-ip/stream" --backend http://127.0.0.1:8000 --roi-config config\roi.yaml --show
```

### RTSP camera (cmd.exe)

```cmd
.venv\Scripts\activate.bat
set PYTHONPATH=.
python -m inference.run --rtsp "rtsp://user:pass@camera-ip/stream" --backend http://127.0.0.1:8000 --roi-config config\roi.yaml --show
```

### Optional enrollment capture mode (inference uploads frames to backend)

```powershell
python -m inference.run --camera webcam --backend http://127.0.0.1:8000 --roi-config config\roi.yaml --show --enroll-employee-id 12 --enroll-kind face
```

## 10) Bulk import employees from folders (person folder = person name)

You can bulk import employee photos from a local folder tree.

Example structure:

```text
D:\attendance_import\
  张伟\
    1.jpg
    2.jpg
    3.jpg
  山田太郎\
    a.png
    b.png
  Maria Santos\
    img1.jpg
```

Rules:

- Each immediate subfolder under the import root is one person
- Subfolder name becomes `employees.full_name` (Unicode preserved; Asian names supported)
- Images inside that subfolder are uploaded for enrollment (`face` by default)
- If a filename starts with `re_` (case-insensitive, e.g. `RE_side.jpg`, `Re_body.png`), it is uploaded as `ReID` instead of face
- Employee code is auto-generated (`EMP0001`, `EMP0002`, ...)

PowerShell (recommended for Unicode names/path display):

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python scripts\import_employees_from_folders.py --root "D:\attendance_import" --backend http://127.0.0.1:8000 --kind face --existing use
```

PowerShell wrapper:

```powershell
scripts\import_employees_from_folders.ps1 -Root "D:\attendance_import" -Kind face -Existing use
```

cmd.exe:

```cmd
.venv\Scripts\activate.bat
set PYTHONPATH=.
python scripts\import_employees_from_folders.py --root "D:\attendance_import" --backend http://127.0.0.1:8000 --kind face --existing use
```

Useful options:

- `--existing use` reuse existing employee with same `full_name` and upload more photos
- `--existing skip` skip folders whose names already exist in DB
- `--existing error` stop on first existing-name match
- `--recursive` scan nested subfolders for images
- `--dry-run` preview actions without creating/uploading

## 11) ROI and thresholds configuration

- Edit `config/roi.yaml` for initial ROI polygon and entry line.
- Use the UI `Settings` page to update thresholds and ROI JSON in the database.

## 12) Troubleshooting (Windows)

- After updating this repo, employee profile fields/photos table may require migrations:

  - Run `alembic -c backend\\alembic.ini upgrade head` again

- `ExecutionPolicy` blocks PowerShell venv activate:
  - Run PowerShell as current user and use:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
- `psycopg2` install issues:
  - Use `psycopg2-binary` (already included) and confirm Python architecture matches Postgres client tools.
- `insightface` import/runtime errors:
  - Verify `onnxruntime` (CPU) or `onnxruntime-gpu` (GPU) is installed.
  - Confirm model files exist under `models\insightface\models\buffalo_l\`.
  - `insightface` is distributed as source on PyPI; on some Windows/Python 3.13 setups you may need Visual C++ Build Tools.
- GPU not used:
  - Check `torch.cuda.is_available()` in Python.
  - Ensure matching NVIDIA driver + CUDA-compatible PyTorch build.
- ByteTrack adapter errors on Windows:
  - The runtime falls back automatically to the built-in IoU tracker.
- RTSP instability:
  - Test camera URL in VLC first.
  - Try lower resolution/bitrate on the camera.

## Performance Targets (practical)

- GPU: ~10-20 FPS (depends on camera resolution and hardware)
- CPU-only: ~5-10 FPS (use lower resolution for stability)
