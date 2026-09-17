# Experiment Timeline and Local Compute Report

This document records what has run locally, when it completed, and how long the
persisted logs say each run took. The goal is to keep enough operational detail
for a later paper methods section and for reproducibility.

Times are local machine times in Europe/Tallinn. For training runs, runtime is
the sum of per-step `wall-time(ms)` from the CSV logs. This excludes some Python
startup, model initialization, and checkpoint overhead, so shell elapsed time
can be slightly higher. `Device hours` are local single-GPU wall hours from the
CSV logs. Local device hours are not paid compute cost, but they are useful for
scaling estimates.

## Local Environment

Measured on 2026-06-18:

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU
- VRAM: 8192 MiB total
- Available VRAM at query time: 5623 MiB
- Disk: about 317 GB free on `C:` at query time
- OS shell: Windows PowerShell
- Primary device used in experiments: `cuda:0`

The local environment is good enough for bounded subset experiments, but it is
not a substitute for full-scale SSv2 pretraining.

## Local Feasibility Conclusion

The current laptop can carry the project a meaningful distance. Based on the
128-clip SSv2 timings:

- ViT-Tiny local throughput is about 10 clips/s.
- VideoMamba-Tiny with an attention predictor is about 2 clips/s.
- VideoMamba uses about 4-5x more local wall time than the ViT-Tiny baseline in
  this fallback environment.

This is sufficient for:

- 512-2048 clip SSv2 subset studies.
- short ViT-Tiny vs VideoMamba-Tiny ablations.
- Mamba predictor vs attention predictor comparisons.
- multi-block vs row/tube mask comparisons.
- target normalization checks.
- collapse, variance, gradient, throughput, and memory reporting.
- a real held-out frozen probe once a larger label-balanced subset is prepared.

It is probably not sufficient for:

- full SSv2 pretraining across many configurations.
- 224/384-resolution or 16-frame pretraining at meaningful scale.
- SOTA claims.
- reproducing the inherited 300-epoch distributed setup.

The practical paper path is therefore local-first: produce a clean bounded
implementation/evaluation study on SSv2 subsets, then use cluster or rented
CUDA only if the local results justify one stronger final run.

## Data Preparation

The local SSv2 data used so far came from the Hugging Face mirror
`morpheushoc/something-something-v2`.

Commands:

```bash
python -m diagnostics.fetch_ssv2_hf_part --parts 00
python -m diagnostics.extract_ssv2_hf_subset --num-clips 128
python -m diagnostics.make_subset --input diagnostic_runs/data/ssv2_subset/ssv2_train_128.csv --output diagnostic_runs/data/ssv2_subset/ssv2_train_32.csv --num-clips 32 --seed 234 --require-existing
```

Local generated data and real-video data are under `diagnostic_runs/`, which is
ignored by git. The repository intentionally tracks scripts, configs, and docs,
not the dataset, checkpoints, or run logs.

## Run Timeline

| Phase | Run | Completed Local Time | Steps | Final Loss | Min Loss | Wall Seconds | Device Hours | Mean Clips/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generated | ViT generated multiblock | 2026-06-18 14:24:20 | 2 | 0.81138 | 0.81138 | 1.0 | 0.00027 | 7.45 |
| generated | VideoMamba Mamba generated multiblock | 2026-06-18 14:21:12 | 2 | 0.81694 | 0.81676 | 2.6 | 0.00072 | 1.76 |
| generated | VideoMamba attention generated row/tube | 2026-06-18 14:24:47 | 2 | 0.80607 | 0.80607 | 2.3 | 0.00065 | 1.80 |
| 4-step SSv2 | ViT 32-clip multiblock | 2026-06-18 18:49:13 | 4 | 0.83497 | 0.81008 | 1.9 | 0.00053 | 8.96 |
| 4-step SSv2 | VideoMamba Mamba 32-clip multiblock | 2026-06-18 18:49:41 | 4 | 0.85765 | 0.81443 | 4.0 | 0.00111 | 2.05 |
| 4-step SSv2 | VideoMamba attention 32-clip row/tube | 2026-06-18 18:50:09 | 4 | 0.84220 | 0.81171 | 4.8 | 0.00133 | 1.75 |
| 32-clip | ViT multiblock layer-norm | 2026-06-18 18:58:42 | 48 | 0.75656 | 0.73006 | 10.3 | 0.00286 | 10.52 |
| 32-clip | VideoMamba Mamba multiblock layer-norm | 2026-06-18 18:59:56 | 48 | 0.75706 | 0.75100 | 47.6 | 0.01324 | 2.03 |
| 32-clip | VideoMamba Mamba multiblock raw | 2026-06-18 19:04:37 | 48 | 0.76255 | 0.73877 | 45.1 | 0.01255 | 2.14 |
| 32-clip | VideoMamba Mamba row/tube layer-norm | 2026-06-18 19:06:54 | 48 | 0.76994 | 0.74210 | 47.5 | 0.01321 | 2.05 |
| 32-clip | VideoMamba attention multiblock layer-norm | 2026-06-18 19:05:43 | 48 | 0.74248 | 0.72973 | 42.8 | 0.01190 | 2.27 |
| 32-clip | VideoMamba attention row/tube layer-norm | 2026-06-18 19:01:08 | 48 | 0.73873 | 0.73137 | 42.3 | 0.01175 | 2.30 |
| 128-clip | ViT multiblock layer-norm | 2026-06-18 19:09:55 | 128 | 0.71926 | 0.67479 | 25.5 | 0.00710 | 10.33 |
| 128-clip | VideoMamba attention multiblock layer-norm | 2026-06-18 19:12:13 | 128 | 0.67468 | 0.65823 | 114.3 | 0.03177 | 2.26 |
| 128-clip | VideoMamba attention row/tube layer-norm | 2026-06-18 19:16:11 | 127 | 0.68413 | 0.64509 | 129.1 | 0.03587 | 2.06 |
| 512-clip | ViT multiblock layer-norm | 2026-06-18 19:50:21 | 768 | 0.64137 | 0.42909 | 166.6 | 0.04628 | 9.33 |
| 512-clip | VideoMamba attention multiblock layer-norm | 2026-06-18 20:03:00 | 768 | 0.71896 | 0.41646 | 712.6 | 0.19795 | 2.20 |
| 512-clip | VideoMamba attention row/tube layer-norm | 2026-06-18 20:14:30 | 768 | 0.70946 | 0.48851 | 654.3 | 0.18174 | 2.36 |
| 512-clip 10-epoch | ViT multiblock layer-norm | 2026-06-18 20:34:58 | 2560 | 0.53267 | 0.16541 | 546.1 | 0.15204 | 9.50 |
| 512-clip 10-epoch | VideoMamba attention row/tube layer-norm | 2026-06-18 21:17:00 | 2558 | 0.53668 | 0.38167 | 2480.8 | 0.68946 | 2.08 |
| 2048-clip | ViT multiblock layer-norm, interrupted timeout probe | 2026-06-19 12:50:02 | 42 | 0.73445 | 0.70886 | 41.4 | 0.01152 | 2.29 |
| 2048-clip | ViT multiblock layer-norm, CSV lock interrupted | 2026-06-19 13:29:23 | 2559 | 0.26434 | 0.16233 | 1002.2 | 0.27873 | 5.18 |
| 2048-clip | ViT multiblock layer-norm, final full segment | 2026-06-19 13:49:09 | 3072 | 0.49298 | 0.19180 | 931.8 | 0.25927 | 6.80 |
| 2048-clip | VideoMamba attention row/tube layer-norm | 2026-06-19 14:54:26 | 3065 | 0.75317 | 0.35599 | 3818.5 | 1.06112 | 1.62 |

The row/tube 128-clip run completed both epochs and checkpointed, but the CSV is
missing one epoch-2 logging row. The table reports the persisted CSV evidence.
The 512-clip 10-epoch VideoMamba run completed and checkpointed, but the CSV is
missing two microstep rows. The table reports the persisted CSV evidence.
The first 2048-clip ViT row was an intentional short foreground run used to
expose errors after the background wrapper failed. The second 2048-clip ViT row
was interrupted by a transient Windows/OneDrive CSV append lock. `CSVLogger`
now retries append opens, and the final 2048-clip ViT row reports the complete
post-fix CSV segment.

## Frozen-Probe Timeline

The frozen probe uses the checkpoint `target_encoder`, freezes it, mean-pools
tokens, and trains only a linear classifier. The current local probe filters to
labels appearing at least twice in the 128-clip subset, leaving 72 clips across
28 classes.

| Run | Completed Local Time | Probe Split | Classes | Final Train Acc | Best Train Acc | Eval Acc | Feature Seconds | Total Seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT multiblock | 2026-06-18 19:24:50 | 72 | 28 | 0.9167 | 0.9167 | n/a | 1.85 | 3.34 |
| VideoMamba attention multiblock | 2026-06-18 19:25:15 | 72 | 28 | 0.6944 | 0.7639 | n/a | 4.04 | 7.69 |
| VideoMamba attention row/tube | 2026-06-18 19:25:15 | 72 | 28 | 0.6944 | 0.7639 | n/a | 4.00 | 7.69 |
| ViT multiblock, 512 balanced held-out | 2026-06-18 20:15:17 | 96 train / 32 eval | 32 | 0.8229 | 0.9271 | 0.0312 | 1.62 | 4.09 |
| VideoMamba attention multiblock, 512 balanced held-out | 2026-06-18 20:15:45 | 96 train / 32 eval | 32 | 0.8021 | 0.8438 | 0.0000 | 3.77 | 8.37 |
| VideoMamba attention row/tube, 512 balanced held-out | 2026-06-18 20:16:17 | 96 train / 32 eval | 32 | 0.7917 | 0.8333 | 0.0000 | 3.67 | 7.94 |
| ViT multiblock, 512 balanced 8-class held-out | 2026-06-18 20:19:03 | 48 train / 16 eval | 8 | 0.8333 | 0.8333 | 0.0625 | 1.09 | 3.12 |
| VideoMamba attention multiblock, 512 balanced 8-class held-out | 2026-06-18 20:19:07 | 48 train / 16 eval | 8 | 0.8333 | 0.8750 | 0.0625 | 2.82 | 7.18 |
| VideoMamba attention row/tube, 512 balanced 8-class held-out | 2026-06-18 20:19:07 | 48 train / 16 eval | 8 | 0.8125 | 0.8750 | 0.0625 | 2.84 | 7.21 |
| ViT multiblock, 512 10-epoch balanced held-out | 2026-06-18 21:17:16 | 96 train / 32 eval | 32 | 0.4792 | 0.4896 | 0.0625 | 2.13 | 5.70 |
| VideoMamba attention row/tube, 512 10-epoch balanced held-out | 2026-06-18 21:17:27 | 96 train / 32 eval | 32 | 0.7083 | 0.7396 | 0.0625 | 4.81 | 10.63 |
| ViT multiblock, 512 10-epoch balanced 8-class held-out | 2026-06-18 21:17:33 | 48 train / 16 eval | 8 | 0.6250 | 0.6250 | 0.1250 | 0.95 | 2.50 |
| VideoMamba attention row/tube, 512 10-epoch balanced 8-class held-out | 2026-06-18 21:17:45 | 48 train / 16 eval | 8 | 0.7083 | 0.7292 | 0.0000 | 2.25 | 5.81 |
| ViT multiblock, 512 10-epoch 32-class strong probe | 2026-06-18 21:23:51 | 96 train / 32 eval | 32 | 0.6354 | 0.7188 | 0.0312 | 2.36 | 9.50 |
| VideoMamba attention row/tube, 512 10-epoch 32-class strong probe | 2026-06-18 21:24:11 | 96 train / 32 eval | 32 | 0.8125 | 0.8854 | 0.0938 | 4.51 | 13.87 |
| ViT multiblock, 512 10-epoch 8-class strong probe | 2026-06-18 21:24:24 | 48 train / 16 eval | 8 | 0.8958 | 0.9375 | 0.1250 | 1.16 | 6.65 |
| VideoMamba attention row/tube, 512 10-epoch 8-class strong probe | 2026-06-18 21:24:41 | 48 train / 16 eval | 8 | 1.0000 | 1.0000 | 0.0625 | 2.78 | 10.43 |
| ViT 512 checkpoint on 2048 16-class split | 2026-06-19 13:00:36 | 240 train / 80 eval | 16 | 0.2125 | 0.2833 | 0.1000 | 6.99 | 17.00 |
| VideoMamba 512 checkpoint on 2048 16-class split | 2026-06-19 13:01:03 | 240 train / 80 eval | 16 | 0.3708 | 0.4208 | 0.0875 | 10.98 | 27.22 |
| ViT 512 checkpoint on 2048 32-class split | 2026-06-19 13:01:23 | 320 train / 160 eval | 32 | 0.2219 | 0.2219 | 0.0188 | 6.70 | 19.30 |
| VideoMamba 512 checkpoint on 2048 32-class split | 2026-06-19 13:02:01 | 320 train / 160 eval | 32 | 0.3344 | 0.3688 | 0.0437 | 17.14 | 38.05 |
| ViT 2048 checkpoint on 2048 16-class split | 2026-06-19 14:57:05 | 240 train / 80 eval | 16 | 0.2625 | 0.3625 | 0.0750 | 7.64 | 22.41 |
| VideoMamba 2048 checkpoint on 2048 16-class split | 2026-06-19 14:57:40 | 240 train / 80 eval | 16 | 0.3500 | 0.4292 | 0.0875 | 15.68 | 33.18 |
| ViT 2048 checkpoint on 2048 32-class split | 2026-06-19 14:58:05 | 320 train / 160 eval | 32 | 0.2906 | 0.2906 | 0.0125 | 7.81 | 24.92 |
| VideoMamba 2048 checkpoint on 2048 32-class split | 2026-06-19 14:58:47 | 320 train / 160 eval | 32 | 0.3563 | 0.3781 | 0.0375 | 19.76 | 42.10 |

The first three probe rows are probe plumbing and overfit checks, not held-out
accuracy. The 512 balanced rows use a held-out split, but the split has only
three train clips per class and one held-out clip per class. Accuracy is at
chance, so this verifies the pipeline but is not yet useful representation
evidence. The 8-class split increases examples per class but still gives only 1
of 16 correct for all models, so the current representations are not yet useful
for held-out probing.

The 10-epoch follow-up improves the ViT pretraining loss and increases
VideoMamba feature variance, but it still does not produce useful held-out
probe accuracy. This makes the next step a data/probe/objective-improvement
question, not simply "run the same 512 clips for longer."

The strong-probe rows use `--probe-lr 0.1` and `--probe-epochs 300`. They show
that the linear head can fit the tiny train split much more aggressively, but
held-out accuracy remains weak and held-out loss becomes very large. This is a
probe-overfitting warning, not a positive representation result.

The 2048 split rows use better class support than the earlier 512-derived
splits. They still do not produce useful held-out accuracy. This is the local
transition point: continue locally only for targeted objective debugging, or
move to LUMI for selective larger-scale tests.

## Next Planned Local Run

The 2048-clip subset and denser probe splits are done. They improved the
evaluation substrate but did not produce useful held-out frozen-probe
accuracy. Do not spend more local epochs on the same 512-clip or 2048-clip
setting.

Continue locally only for cheap diagnostics: probe-protocol checks,
random-feature or kNN/logistic sanity baselines, and at most one targeted
VideoMamba objective tweak (predictor capacity, target projection, mask
ratio, or EMA/LR). Start Rocket/LUMI access in parallel. Prefer Rocket/CUDA
for the first cluster pilot. Use LUMI only after `hpc/lumi_smoke.sbatch`
confirms that the ROCm environment can run the VideoMamba path.

The completed 512-clip, 10-epoch, and 2048-clip configs were:

- `configs/diagnostics/local_ssv2_512_vit.yaml`
- `configs/diagnostics/local_ssv2_512_videomamba_attention_multiblock.yaml`
- `configs/diagnostics/local_ssv2_512_videomamba_attention_rowtube.yaml`
- `configs/diagnostics/local_ssv2_512_long_vit.yaml`
- `configs/diagnostics/local_ssv2_512_long_videomamba_attention_rowtube.yaml`
- `configs/diagnostics/local_ssv2_2048_vit.yaml`
- `configs/diagnostics/local_ssv2_2048_videomamba_attention_rowtube.yaml`

The next paper-relevant compute step is the stage-1 cluster matrix in
`configs/hpc/stage1_matrix.yaml`: 8192 clips, 20 epochs, small geometry, ViT
baseline plus VideoMamba attention/row-tube with layer-norm and L2 targets.
Run only the best understood 2-3 configurations after the smoke gate.

## What Must Be Reported In Future Runs

For every run that might appear in a paper table, record:

- exact config path
- checkpoint path
- manifest path and subset construction rule
- seed
- GPU/device
- start/completion time
- wall seconds and local device hours
- throughput
- final and minimum loss
- target variance and prediction variance
- encoder and predictor gradient norms
- memory
- probe split details and whether the encoder was frozen
- whether the result is overfit/plumbing evidence or held-out evaluation
