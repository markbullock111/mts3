@echo off
if "%~1"=="" (
  echo Usage: scripts\run_inference_rtsp.bat "rtsp://..."
  exit /b 1
)
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=.
python -m inference.run --rtsp %1 --backend http://127.0.0.1:8000 --roi-config config/roi.yaml --show
