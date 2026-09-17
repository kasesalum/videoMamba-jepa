#!/bin/bash
#SBATCH --job-name=vjepa-smoke
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err

# One-GPU VideoMamba smoke on UT Rocket.
# Submit from the repo root on the login node:
#   mkdir -p logs
#   sbatch hpc/rocket_smoke.sbatch
# Edit --account before first submit (sacctmgr show assoc where user=$USER -p).
# Full install / troubleshooting: SETUP.md (UT Rocket section).

set -euo pipefail

# Prefer the directory used at sbatch time (repo root).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv/bin/python not found under $REPO_ROOT" >&2
  echo "Complete the Linux install in SETUP.md before submitting." >&2
  exit 1
fi

module purge
module load gcc/11
module load cuda/12.1

export LIBSTDCPP_DIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$LIBSTDCPP_DIR:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
export PYTHONPATH="$(pwd):$(pwd)/src"

mkdir -p logs output/videomambaT16_pretrain

# Optional fast import gate before the training smoke.
.venv/bin/python scripts/verify_install.py

# Do not wrap with srun for a single-task job (avoids CPU-binding errors).
.venv/bin/python -m app.main \
  --fname configs/pretrain/videomambaT16.yaml \
  --devices cuda:0
