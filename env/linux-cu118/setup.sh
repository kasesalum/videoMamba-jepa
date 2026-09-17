#!/usr/bin/env bash
set -euo pipefail

# Ordered environment setup for Linux + NVIDIA GPU + CUDA 11.8.
# This script is intentionally explicit so builds are reproducible.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-jepa-cu118}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

echo "[1/8] Creating conda environment: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y

echo "[2/8] Installing PyTorch CUDA 11.8 wheels"
conda run -n "${ENV_NAME}" pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.2 torchvision==0.16.2

echo "[3/8] Installing pinned Python dependencies"
conda run -n "${ENV_NAME}" pip install -r env/linux-cu118/requirements-core.txt

echo "[4/8] Installing CUDA extension package: causal-conv1d"
conda run -n "${ENV_NAME}" pip install -v --no-build-isolation ./src/mamba/causal-conv1d

echo "[5/8] Installing CUDA extension package: mamba_ssm"
conda run -n "${ENV_NAME}" pip install -v --no-build-isolation ./src/mamba

echo "[6/8] Installing project package"
conda run -n "${ENV_NAME}" pip install -e . --no-deps

echo "[7/8] Verifying CUDA + imports"
conda run -n "${ENV_NAME}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('CUDA OK:', torch.version.cuda)"
conda run -n "${ENV_NAME}" python -c "import causal_conv1d_cuda; print('causal_conv1d_cuda import OK')"
conda run -n "${ENV_NAME}" python -c "import selective_scan_cuda; print('selective_scan_cuda import OK')"
conda run -n "${ENV_NAME}" python -c "import src.models.videomamba as vm; print('videomamba import OK')"
conda run -n "${ENV_NAME}" python -c "import src.models.videomamba_predictor as vmp; print('videomamba_predictor import OK')"

echo "[8/8] Environment setup complete"
echo "Activate with: conda activate ${ENV_NAME}"
