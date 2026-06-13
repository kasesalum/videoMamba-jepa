# Setup Tutorial

This guide walks through everything needed to clone **videoMamba-jepa**, install dependencies, prepare datasets, and get the environment ready for training.

For the paper experiments (VideoMamba-Tiny and ViT-Tiny with V-JEPA on Something-Something V2), the recommended platform is **Linux with an NVIDIA GPU and CUDA toolkit**. Windows is supported for development and smoke tests using community Mamba wheels, but full paper-scale reproduction is intended for multi-GPU distributed training.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the repository](#2-clone-the-repository)
3. [Create a Python environment](#3-create-a-python-environment)
4. [Install dependencies](#4-install-dependencies)
   - [Linux (recommended)](#linux-recommended)
   - [Windows](#windows)
5. [Activate the environment](#5-activate-the-environment)
6. [Verify the installation](#6-verify-the-installation)
7. [Download datasets](#7-download-datasets)
   - [Something-Something V2 (required for paper experiments)](#something-something-v2-required-for-paper-experiments)
   - [Kinetics-400 (optional)](#kinetics-400-optional)
   - [ImageNet-1K (optional, image probe only)](#imagenet-1k-optional-image-probe-only)
8. [Prepare CSV file lists](#8-prepare-csv-file-lists)
9. [Update experiment configs](#9-update-experiment-configs)
10. [Smoke test before full training](#10-smoke-test-before-full-training)
11. [Training workflow overview](#11-training-workflow-overview)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **OS** | Linux (Ubuntu 20.04+ recommended). Windows works with extra steps (see below). |
| **GPU** | NVIDIA GPU with CUDA support. VideoMamba requires Mamba CUDA kernels. |
| **CUDA** | CUDA 11.6+ on Linux. PyTorch cu121 wheels are used in this guide. |
| **Python** | 3.9–3.11 (3.11 tested). |
| **Git** | Required for cloning and the `mamba2` submodule. |
| **Disk space** | ~50 GB+ for SSv2 videos; more for Kinetics / ImageNet. |
| **RAM** | 16 GB+ recommended for data loading. |

**Paper-scale hardware:** Pretraining configs assume large distributed jobs (e.g. 16 nodes × 8 GPUs). Local single-GPU runs are useful for debugging but will not reproduce paper numbers without scaling batch size, learning rate, and epoch count accordingly.

---

## 2. Clone the repository

Clone with submodules so the `mamba2` dependency is included:

```bash
git clone --recurse-submodules https://github.com/<your-org>/videoMamba-jepa.git
cd videoMamba-jepa
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive src/mamba2
```

---

## 3. Create a Python environment

### Option A: venv (Linux or Windows)

```bash
python3.11 -m venv .venv
```

### Option B: conda (Linux)

```bash
conda create -n jepa python=3.11 pip -y
conda activate jepa
```

---

## 4. Install dependencies

Use **`requirements-core.txt`** for training. The full `requirements.txt` includes optional packages (`apex`, `deepspeed`, `tensorflow`, `xformers`) that are not required for the SSv2 experiments and often fail to build.

### Linux (recommended)

```bash
# Activate venv first if using one
source .venv/bin/activate   # Linux
# conda activate jepa       # if using conda

pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 12.1 (adjust index URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core Python packages
pip install -r requirements-core.txt

# Build Mamba from the submodule (requires nvcc / CUDA toolkit)
pip install --no-build-isolation src/mamba/causal-conv1d
pip install --no-build-isolation src/mamba2

# Register repo paths (or use activate script below)
cat > .venv/lib/python3.11/site-packages/videomamba_jepa_path.pth << EOF
$(pwd)
$(pwd)/src
EOF
```

If `pip install src/mamba2` fails, ensure the CUDA toolkit is installed and `nvcc` is on your `PATH`:

```bash
nvcc --version
export CUDA_HOME=/usr/local/cuda   # adjust if needed
pip install --no-build-isolation src/mamba2
```

### Windows

Official Mamba CUDA extensions are not published for Windows. Use the automated installer, which installs community wheels compatible with **PyTorch 2.3.0+cu121**:

```powershell
cd videoMamba-jepa
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

Or install manually:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip setuptools wheel
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-core.txt

pip install --no-deps https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/causal_conv1d-1.1.1-cp311-cp311-win_amd64.whl
pip install --no-deps https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/mamba_ssm-1.1.3-cp311-cp311-win_amd64.whl
```

Then activate (see next section).

**Windows limitations:**
- Uses older community `mamba-ssm` 1.1.3 wheels (repo submodule is 2.1.0).
- Full 300-epoch distributed pretraining is impractical on consumer GPUs.
- For exact paper parity, use Linux/WSL with a source-built `mamba2`.

---

## 5. Activate the environment

### Linux

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/src"
export KMP_DUPLICATE_LIB_OK=TRUE
```

Or add an alias/script:

```bash
source scripts/activate.sh   # if you create one mirroring activate.ps1
```

### Windows (PowerShell)

```powershell
. .\scripts\activate.ps1
```

This activates `.venv` and sets `PYTHONPATH` to the repo root and `src/`.

---

## 6. Verify the installation

```bash
python scripts/verify_install.py
```

Expected output: all checks `OK`, including `torch + cuda`, `mamba_ssm`, `src.models.videomamba`, and `src.models.vision_transformer`.

Quick manual check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from src.models.videomamba import videomamba_tiny; print('VideoMamba OK')"
python -c "from src.models.vision_transformer import vit_tiny; print('ViT OK')"
```

---

## 7. Download datasets

### Something-Something V2 (required for paper experiments)

**Source:** [Qualcomm AI Hub — Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads)

1. Create an account and request access to the dataset.
2. Download the video archives and label files.
3. Extract into `src/datasets/SSv2/` (gitignored; not committed). Expected layout:

**Repo-relative path (recommended):**

```
src/datasets/SSv2/
├── videos/
│   └── 20bn-something-something-v2/
│       ├── 1.webm
│       ├── 2.webm
│       └── ...
└── labels/
    ├── train.json
    ├── validation.json
    ├── test.json
    ├── labels.json
    └── test-answers.csv
```

**Windows absolute path (example):**

```
<repo-root>/src/datasets/SSv2/
```

If you still have an archive such as `something.tar` in that folder after extraction, you can delete it to reclaim disk space once you confirm the videos play correctly.

**Files you need:**

| File | Purpose |
|---|---|
| `train.json` | Training video IDs and action templates |
| `validation.json` | Validation split for attentive probe |
| `labels.json` | Maps template strings → integer class IDs (174 classes) |
| `*.webm` videos | One file per video ID in the JSON files |

**Alternative / mirror:** The original dataset page is [20bn.com](https://20bn.com/something-something). Some labs host mirrors; use whichever your institution provides.

---

### Kinetics-400 (optional)

Used by configs such as `configs/evals/vitl16_k400_16x8x3.yaml`.

**Sources:**
- Official: [DeepMind Kinetics](https://deepmind.google/discover/blog/open-sourcing-a-deepmind-dataset-kinetics/)
- Common practice: download preprocessed splits from [CVDF Foundation](https://github.com/cvdfoundation/kinetics-dataset) and create CSV file lists in the same format as SSv2 (absolute path + integer label per line).

---

### ImageNet-1K (optional, image probe only)

Used by `configs/evals/*_in1k.yaml` for frozen image classification probes.

**Source:** [ImageNet](https://www.image-net.org/download.php)

Organize in PyTorch `ImageFolder` layout:

```
/path/to/imagenet_full_size/061417/
├── train/
│   ├── n01440764/
│   │   ├── xxx.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

---

## 8. Prepare CSV file lists

The codebase expects **space-delimited CSV files** with no header. Video paths are **repo-relative** (portable across machines):

```
src/datasets/SSv2/videos/20bn-something-something-v2/1.webm 42
src/datasets/SSv2/videos/20bn-something-something-v2/2.webm 17
```

`generate_ssv2_filelist` writes repo-relative paths by default. Training code resolves them against the repository root automatically.

For V-JEPA **pretraining**, class labels are ignored (any integer is fine). For **attentive probe** training on SSv2, labels must be correct integers from `labels.json`.

### Generate SSv2 CSVs

From the repo root (with the environment activated). Set paths once, then run all three generators.

**Linux / macOS:**

```bash
SSV2="$(pwd)/src/datasets/SSv2"
VIDEOS="$SSV2/videos/20bn-something-something-v2"
LABELS="$SSV2/labels"

# Training split — probe training + VideoMamba pretrain
python -m src.datasets.generate_ssv2_filelist \
  --video-dir "$VIDEOS" \
  --labels-json "$LABELS/train.json" \
  --label-map "$LABELS/labels.json" \
  --output "$LABELS/SSv2_train_probe_filelist.csv"

# Validation split — probe evaluation
python -m src.datasets.generate_ssv2_filelist \
  --video-dir "$VIDEOS" \
  --labels-json "$LABELS/validation.json" \
  --label-map "$LABELS/labels.json" \
  --output "$LABELS/SSv2_valid_probe_filelist.csv"

# Unlabeled-style train list — ViT pretrain config
python -m src.datasets.generate_ssv2_filelist \
  --video-dir "$VIDEOS" \
  --labels-json "$LABELS/train.json" \
  --label-map "$LABELS/labels.json" \
  --output "$LABELS/SSv2_filelist.csv"
```

**PowerShell (Windows):**

```powershell
cd <repo-root>
. .\scripts\activate.ps1

$SSV2 = "src/datasets/SSv2"
$VIDEOS = "$SSV2/videos/20bn-something-something-v2"
$LABELS = "$SSV2/labels"

python -m src.datasets.generate_ssv2_filelist `
  --video-dir $VIDEOS `
  --labels-json "$LABELS\train.json" `
  --label-map "$LABELS\labels.json" `
  --output "$LABELS\SSv2_train_probe_filelist.csv"

python -m src.datasets.generate_ssv2_filelist `
  --video-dir $VIDEOS `
  --labels-json "$LABELS\validation.json" `
  --label-map "$LABELS\labels.json" `
  --output "$LABELS\SSv2_valid_probe_filelist.csv"

python -m src.datasets.generate_ssv2_filelist `
  --video-dir $VIDEOS `
  --labels-json "$LABELS\train.json" `
  --label-map "$LABELS\labels.json" `
  --output "$LABELS\SSv2_filelist.csv"
```

Each command prints how many rows were written. Warnings about missing videos usually mean extraction is incomplete.

---

## 9. Update experiment configs

All hyperparameters live in YAML files under `configs/`. Paths should be **repo-relative** (not machine-specific absolute paths). The training and eval code resolves them against the repository root via `src/utils/paths.py`.

The SSv2 configs in this repo are already set up for the default dataset layout:

### Paper experiments on SSv2

| Model | Pretrain config | Probe / eval config |
|---|---|---|
| VideoMamba-Tiny | `configs/pretrain/videomambaT16.yaml` | `configs/evals/videomambaT16_ssv2_16x2x3.yaml` |
| ViT-Tiny | `configs/pretrain/vitt16.yaml` | `configs/evals/vitt16_ssv2_16x2x3.yaml` |

### Fields to update

If you change dataset or output locations, edit paths in the YAML files below. Use repo-relative paths such as `src/datasets/SSv2/...` and `output/...`.

**Pretrain YAML** (`configs/pretrain/*.yaml`):

```yaml
data:
  datasets:
    # VideoMamba: SSv2_train_probe_filelist.csv
    # ViT: SSv2_filelist.csv
    - src/datasets/SSv2/labels/SSv2_train_probe_filelist.csv

logging:
  folder: output/videomambaT16_pretrain
```

**Eval YAML** (`configs/evals/*_ssv2_*.yaml`):

```yaml
data:
  dataset_train: src/datasets/SSv2/labels/SSv2_train_probe_filelist.csv
  dataset_val: src/datasets/SSv2/labels/SSv2_valid_probe_filelist.csv

pretrain:
  folder: output/videomambaT16_pretrain   # must contain jepa-latest.pth.tar
```

Absolute paths still work if needed, but repo-relative paths are preferred so configs can be shared across machines.

### Paper probe duration

The paper reports results after **4** attentive-probe epochs. Repo configs default to `num_epochs: 20`. To match the paper, set:

```yaml
optimization:
  num_epochs: 4
```

in the eval YAML files.

---

## 10. Smoke test before full training

Before launching a multi-day distributed job, confirm the pipeline works on one GPU with a tiny run.

1. Copy a pretrain config (e.g. `videomambaT16.yaml`) and reduce:
   - `optimization.epochs: 1`
   - `data.batch_size: 2` (or `1` on 6–8 GB GPUs)
   - `optimization.ipe: 5` (optional; limits iterations per epoch)

2. Run locally:

```bash
python -m app.main \
  --fname configs/pretrain/videomambaT16.yaml \
  --devices cuda:0
```

3. Check that:
   - Videos load without errors
   - Training loss is printed
   - Checkpoint is written to `logging.folder` as `jepa-latest.pth.tar`

For ViT:

```bash
python -m app.main \
  --fname configs/pretrain/vitt16.yaml \
  --devices cuda:0
```

The training entrypoint automatically routes `videomamba_*` models to `train_videomamba.py` and `vit_*` models to `train.py`.

---

## 11. Training workflow overview

Once setup is complete, run experiments in this order:

```
1. V-JEPA pretrain (VideoMamba-Tiny)     →  configs/pretrain/videomambaT16.yaml
2. Attentive probe (VideoMamba-Tiny)     →  configs/evals/videomambaT16_ssv2_16x2x3.yaml
3. V-JEPA pretrain (ViT-Tiny)            →  configs/pretrain/vitt16.yaml
4. Attentive probe (ViT-Tiny)            →  configs/evals/vitt16_ssv2_16x2x3.yaml
```

### Local (single machine, debugging)

```bash
# Pretrain
python -m app.main --fname configs/pretrain/videomambaT16.yaml --devices cuda:0

# Attentive probe (frozen backbone + trainable probe head)
python -m evals.main --fname configs/evals/videomambaT16_ssv2_16x2x3.yaml --devices cuda:0
```

### Distributed (SLURM cluster)

```bash
python -m app.main_distributed \
  --fname configs/pretrain/videomambaT16.yaml \
  --folder /path/to/submitit_logs \
  --partition YOUR_PARTITION

python -m evals.main_distributed \
  --fname configs/evals/videomambaT16_ssv2_16x2x3.yaml \
  --folder /path/to/submitit_logs \
  --partition YOUR_PARTITION
```

Probe validation accuracy is logged each epoch to:

```
{pretrain.folder}/video_classification_frozen/ssv2-16x2x3/jepa_r0.csv
```

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'mamba2'` or `selective_scan_cuda`

- **Linux:** Build `src/mamba2` from source with `nvcc` available (`pip install --no-build-isolation src/mamba2`).
- **Windows:** Use `scripts/install.ps1` and keep PyTorch at **2.3.0+cu121**.

### `ModuleNotFoundError: No module named 'app'` or `src`

Set `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd):$(pwd)/src"
```

Or use `scripts/activate.ps1` (Windows) / register the `.pth` file as in `scripts/install.ps1`.

### `decord` / video loading errors

- Confirm video paths in the CSV are **absolute** and files exist.
- SSv2 videos are `.webm` by default; pass `--video-ext .mp4` if your files differ.

### CUDA out of memory

- Reduce `data.batch_size` in the config.
- For VideoMamba pretrain, reduce `optimization.accumulation_steps` effects by lowering batch size.
- Use a single GPU smoke config before scaling up.

### ViT pretrain runs VideoMamba code

Ensure you are on a recent checkout where `app/scaffold.py` routes `vit_*` model names to `train.py` and `videomamba_*` to `train_videomamba.py`.

### Config paths still point to `/content/` or `/scratch/`

These are Colab/cluster placeholders from the authors. Every path in the YAML must be edited to match your machine before training.

---

## Quick reference checklist

- [ ] Cloned repo with `git submodule update --init --recursive src/mamba2`
- [ ] Python 3.11 venv created and activated
- [ ] PyTorch + CUDA installed and `torch.cuda.is_available()` is `True`
- [ ] `requirements-core.txt` installed
- [ ] Mamba installed (source build on Linux, or Windows wheels via `scripts/install.ps1`)
- [ ] `python scripts/verify_install.py` passes
- [ ] SSv2 downloaded and extracted to `src/datasets/SSv2/` (`videos/` + `labels/`)
- [ ] CSV file lists generated in `src/datasets/SSv2/labels/` (`SSv2_*_filelist.csv`)
- [ ] Config YAML paths updated (data + output folders)
- [ ] Smoke test completed on 1 GPU

When all items are checked, the environment is ready for training.
