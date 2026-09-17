# HPC (Rocket first)

Cluster ops for the VideoMamba + V-JEPA pilot. Prefer **UT Rocket (CUDA)** for the first smoke and stage-1 matrix; treat LUMI/ROCm as a later option only after Rocket VideoMamba is healthy.

## Full setup

Install, modules, GLIBCXX fixes, interactive `srun`, and troubleshooting live in:

- [SETUP.md](../SETUP.md) — Linux install + **UT Rocket (Slurm)** section

Strategy and budget notes:

- [docs/hpc_lumi_usage.md](../docs/hpc_lumi_usage.md)
- [docs/research_handoff.md](../docs/research_handoff.md)

## Smoke (gate before any matrix)

1. Clone + build `.venv` and regenerate SSv2 CSVs **on Rocket** (see SETUP.md).
2. Set `#SBATCH --account=...` in `rocket_smoke.sbatch` (`sacctmgr show assoc where user=$USER -p`).
3. From the **repo root** on the login node:

```bash
mkdir -p logs
sbatch hpc/rocket_smoke.sbatch
```

4. Confirm loss lines in `logs/smoke_*.out` and checkpoints under `output/videomambaT16_pretrain/`.

`lumi_smoke.sbatch` is not provided yet; do not start LUMI until Rocket smoke + at least one VideoMamba stage-1 seed succeeds.

## Stage-1 matrix (after smoke)

| File | Role |
|------|------|
| [configs/hpc/stage1_base.yaml](../configs/hpc/stage1_base.yaml) | Shared geometry: 8192-clip CSV path, 4 frames, 64px, 20 epochs, 1 GPU |
| [configs/hpc/stage1_matrix.yaml](../configs/hpc/stage1_matrix.yaml) | Variants: ViT multiblock + VideoMamba row/tube (layer-norm and L2), seeds 234–236 |

Before submitting stage-1:

1. Replace `/path/to/ssv2_train_stage1.csv` in `stage1_base.yaml` with a Rocket-local 8192-clip manifest.
2. Materialize patched configs under `diagnostic_runs/configs/hpc_stage1/` (see `config_root` / `output_root` in the matrix file).
3. Submit one config at a time on 1 GPU; then run frozen probes on fixed balanced splits.

Do not jump to paper-scale multi-node configs until stage-1 probes are informative.
