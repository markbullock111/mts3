@echo off
set VENV=.venv
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3.13 -m venv %VENV%
) else (
  python -m venv %VENV%
)
call %VENV%\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-base.txt
pip install onnxruntime-gpu==1.20.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo GPU setup complete. Verify CUDA compatibility.
