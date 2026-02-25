param(
  [string]$Backend = "http://127.0.0.1:8000",
  [Parameter(Mandatory=$true)][string]$RtspUrl,
  [switch]$Show,
  [int]$SaveSnapshots = -1
)

if (Test-Path ".venv\Scripts\Activate.ps1") { & ".venv\Scripts\Activate.ps1" }
$env:PYTHONPATH = "."
$cmd = "python -m inference.run --rtsp `"$RtspUrl`" --backend $Backend --roi-config config/roi.yaml"
if ($Show) { $cmd += " --show" }
if ($SaveSnapshots -eq 0 -or $SaveSnapshots -eq 1) { $cmd += " --save-snapshots $SaveSnapshots" }
Invoke-Expression $cmd
