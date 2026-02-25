param(
  [Parameter(Mandatory=$true)][string]$Root,
  [string]$Backend = "http://127.0.0.1:8000",
  [ValidateSet("face","reid")][string]$Kind = "face",
  [switch]$Recursive,
  [ValidateSet("use","skip","error")][string]$Existing = "use",
  [switch]$DryRun
)

if (Test-Path ".venv\Scripts\Activate.ps1") { & ".venv\Scripts\Activate.ps1" }
$env:PYTHONPATH = "."
$cmd = "python scripts\import_employees_from_folders.py --root `"$Root`" --backend $Backend --kind $Kind --existing $Existing"
if ($Recursive) { $cmd += " --recursive" }
if ($DryRun) { $cmd += " --dry-run" }
Invoke-Expression $cmd
