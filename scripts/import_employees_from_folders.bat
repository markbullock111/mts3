@echo off
if "%~1"=="" (
  echo Usage: scripts\import_employees_from_folders.bat "C:\path\to\people_folders"
  echo Each subfolder name is the person full name; images inside are uploaded as FACE enrollment.
  exit /b 1
)
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=.
python scripts\import_employees_from_folders.py --root "%~1" --backend http://127.0.0.1:8000 --kind face --existing use
