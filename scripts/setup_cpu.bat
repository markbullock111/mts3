@echo off
set VENV=.venv
python -m venv %VENV%
call %VENV%\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-cpu.txt
echo CPU setup complete.
