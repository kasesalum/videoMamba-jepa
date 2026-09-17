# Linux/CUDA Environment Spec (VideoMamba runnable)

This spec is for running VideoMamba + V-JEPA on a Linux training machine where CUDA extensions must compile.

## Target stack

- OS: Linux x86_64
- Python: `3.10`
- CUDA toolkit (nvcc): `11.8` (required for extension build path)
- NVIDIA driver: compatible with CUDA 11.8
- PyTorch: `2.1.2+cu118`
- torchvision: `0.16.2+cu118`

## System prerequisites

Install these before running the setup script:

- `conda` (or Miniconda)
- `gcc` / `g++` (recommend GCC 10-11)
- CUDA toolkit with `nvcc` available in `PATH`

Quick checks:

```bash
nvidia-smi
nvcc --version
```

## Exact install order

From repo root:

```bash
bash env/linux-cu118/setup.sh
```

This script does, in order:

1. Creates conda env (`jepa-cu118` by default)
2. Installs pinned `torch`/`torchvision` CUDA 11.8 wheels
3. Installs pinned Python deps from `requirements-core.txt`
4. Builds and installs local `causal-conv1d`
5. Builds and installs local `mamba_ssm`
6. Installs project package (`pip install -e . --no-deps`)
7. Verifies CUDA visibility and extension imports:
   - `causal_conv1d_cuda`
   - `selective_scan_cuda`
   - `src.models.videomamba`

## Optional overrides

You can override env name and python version:

```bash
ENV_NAME=my-jepa PYTHON_VERSION=3.10 bash env/linux-cu118/setup.sh
```

## Notes

- `apex`, `deepspeed`, and `xformers` are not required for the exact SSv2 VideoMamba/ViT experiments in this repo.
- If extension builds fail, the most common causes are:
  - `nvcc` not found
  - CUDA/PyTorch mismatch
  - unsupported compiler version
