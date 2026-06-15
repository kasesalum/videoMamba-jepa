# Activate the videoMamba-jepa development environment (PowerShell)
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $VenvActivate) {
    . $VenvActivate
} else {
    Write-Error "Virtual environment not found at $VenvActivate. Run scripts/install.ps1 first."
    return
}

# Repo root: imports like `src.models`, `app`, `evals`
# src/: imports like `mamba2.mamba_ssm`, `utils.logger`
$env:PYTHONPATH = "$RepoRoot;$RepoRoot\src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "Activated videoMamba-jepa environment"
Write-Host "  Python: $(python --version)"
Write-Host "  PYTHONPATH: $env:PYTHONPATH"
