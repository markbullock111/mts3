param(
  [string]$Host = "127.0.0.1",
  [int]$Port = 8000
)

if (Test-Path ".venv\Scripts\Activate.ps1") { & ".venv\Scripts\Activate.ps1" }
$env:PYTHONPATH = "."
python -m uvicorn backend.app.main:app --host $Host --port $Port
