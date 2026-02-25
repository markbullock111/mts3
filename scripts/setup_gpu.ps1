param(
  [string]$Venv = ".venv"
)

python -m venv $Venv
& "$Venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-base.txt
pip install onnxruntime-gpu==1.20.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
Write-Host "GPU setup complete. Verify CUDA compatibility for your GPU/driver."
