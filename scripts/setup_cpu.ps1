param(
  [string]$Venv = ".venv"
)

python -m venv $Venv
& "$Venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-cpu.txt
Write-Host "CPU setup complete. Activate with: $Venv\Scripts\Activate.ps1"
