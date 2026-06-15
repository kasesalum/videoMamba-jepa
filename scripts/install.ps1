# Install videoMamba-jepa dependencies (Windows)
# Run from repo root: powershell -ExecutionPolicy Bypass -File scripts/install.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Require-Python311 {
    $py311 = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if (-not $py311) {
        throw "Python 3.11 not found. Install with: winget install Python.Python.3.11"
    }
    return $py311
}

Write-Host "==> Initializing git submodules"
git submodule update --init --recursive src/mamba2

Write-Host "==> Creating virtual environment (.venv)"
if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

$pip = Join-Path $RepoRoot ".venv\Scripts\pip.exe"
$python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "==> Upgrading pip/setuptools/wheel"
& $python -m pip install --upgrade pip setuptools wheel

Write-Host "==> Installing PyTorch 2.3.0 (CUDA 12.1; required for Windows mamba wheels)"
& $pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

Write-Host "==> Installing core Python dependencies"
& $pip install -r requirements-core.txt
& $pip install "transformers>=4.36,<4.46"

Write-Host "==> Installing causal-conv1d (Windows community wheel)"
& $pip install --no-deps "https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/causal_conv1d-1.1.1-cp311-cp311-win_amd64.whl"

Write-Host "==> Installing mamba_ssm (Windows community wheel)"
& $pip install --no-deps "https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/mamba_ssm-1.1.3-cp311-cp311-win_amd64.whl"

Write-Host "==> Registering repo on PYTHONPATH (.pth)"
$pth = Join-Path $RepoRoot ".venv\Lib\site-packages\videomamba_jepa_path.pth"
@(
    $RepoRoot
    (Join-Path $RepoRoot "src")
) | Set-Content -Encoding ascii $pth

Write-Host ""
Write-Host "Installation complete. Activate with:"
Write-Host "  . .\scripts\activate.ps1"
Write-Host ""
Write-Host "Verify with:"
Write-Host "  python scripts/verify_install.py"
