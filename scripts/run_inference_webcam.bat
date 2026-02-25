@echo off
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=.
python -m inference.run --camera webcam --backend http://127.0.0.1:8000 --roi-config config/roi.yaml --show
