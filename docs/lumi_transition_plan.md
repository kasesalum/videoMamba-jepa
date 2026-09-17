# LUMI Transition Plan

This document records when and how to move beyond local experiments.

For the step-by-step access process, current public price assumptions, billing
notes, and conference-level compute estimate, see `docs/hpc_lumi_usage.md`.
For concrete Slurm templates, see `hpc/README.md`.

## Current Local Conclusion

Local experiments have now covered:

- synthetic and generated-video smoke tests;
- 32/128/512-clip SSv2 sanity runs;
- 512-clip 10-epoch follow-up runs;
- 2048-clip SSv2 runs with more balanced probe splits;
- frozen-probe and probe-sensitivity checks.

The local results establish that VideoMamba can be trained with a V-JEPA-style
masked latent objective, but they do not establish useful held-out
representation quality. ViT loss improves more cleanly. VideoMamba is stable
but slower, has weaker prediction-variance behavior, and remains weak under
held-out frozen probes.

This is enough for a bounded diagnostic paper if framed honestly. It is not
enough for a positive performance paper.

## When To Move To LUMI

Move to LUMI when the next question is no longer "does the code and probe
pipeline work?" but one of:

- Does longer training change the VideoMamba failure mode?
- Does a larger real-video subset produce transferable features?
- Does the VideoMamba + V-JEPA objective need different predictor, target, EMA,
  or mask settings?
- Does the result hold over multiple seeds?

Do not run a broad grid first. Start with the best understood configurations:

1. ViT-Tiny V-JEPA baseline with multi-block masks.
2. VideoMamba-Tiny with attention predictor and row/tube masks.
3. One targeted VideoMamba modification, such as target projection or adjusted
   predictor capacity.

## LUMI Smoke Gate

Before spending real allocation:

1. Create the same Python environment or container.
2. Run import checks for PyTorch, Mamba dependencies, and project modules.
3. Run one VideoMamba forward pass.
4. Run one masked batch and one optimizer step.
5. Save and reload a checkpoint.
6. Run one frozen-probe feature extraction batch.

If Mamba kernels or dependencies fail under ROCm, either:

- use a CUDA system for the pilot runs; or
- frame ROCm/Mamba portability as research engineering work and do not spend
  large LUMI allocation until a fallback path is measured.

## First LUMI Experiment Shape

The first LUMI matrix should be small:

| Run | Purpose |
| --- | --- |
| ViT-Tiny, multi-block, larger subset | Calibration baseline |
| VideoMamba-Tiny, attention predictor, row/tube | Main combination test |
| VideoMamba-Tiny, one objective modification | Test whether local failure mode is fixable |

Each run must report:

- exact config path and git commit;
- data manifest construction rule;
- seed;
- node/GPU type;
- wall time and charged GPU-hours;
- loss curve;
- target and prediction variance;
- encoder and predictor gradient norms;
- throughput and memory;
- frozen-probe result on a fixed held-out split.

## Stop Conditions

Stop or redesign if:

- prediction variance remains low while ViT remains healthy;
- encoder gradients remain much weaker than ViT across repeated runs;
- held-out probes remain at chance after substantially more data or steps;
- ROCm/Mamba issues make each experiment slower than the compute budget allows.

If the local negative pattern persists at LUMI scale, the paper should be framed
as a controlled negative/diagnostic result. If a modification improves
stability or probe transfer, frame the paper around the implementation plus the
compute-bounded fix.

## Current Cost Gate

Under the public rates checked on 2026-06-20, EUR 1000 is approximately 2857
gross LUMI GPU-hours or 2000 Rocket GPU-hours before reserves. With 10% reserved
for setup and 20% for reruns/final checks, treat the usable training budget as
about 2000 LUMI GPU-hours or 1400 Rocket GPU-hours.

The minimum conference-oriented HPC stage should fit comfortably inside that
budget if the smoke benchmark is healthy: an 8192-clip, 20-epoch,
small-geometry matrix over ViT, VideoMamba layer-normalized targets, and
VideoMamba L2-normalized targets is estimated at about 190
single-GPU-equivalent hours for three seeds. A stronger follow-up should be
planned at 1500-1800 accounted LUMI GPU-hours including reserves, not as full
SOTA pretraining.
