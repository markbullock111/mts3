@echo off
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=.
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
