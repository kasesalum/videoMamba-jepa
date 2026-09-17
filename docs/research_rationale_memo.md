# VideoMamba + V-JEPA Research Rationale

## Executive Summary

The project can reasonably be framed as a VideoMamba + V-JEPA study, provided
the claim is modest and evidence-driven:

> A first-pass implementation, feasibility study, and bounded empirical
> evaluation of VideoMamba trained with a V-JEPA-style masked latent prediction
> objective.

The main research question should be:

> Can a pure VideoMamba video backbone be pretrained with a V-JEPA-style masked
> latent prediction objective under realistic compute constraints?

This is different from claiming a state-of-the-art VideoMamba-JEPA model. The
project should not start by attempting full-scale pretraining. The inherited
long pretraining configs are large distributed jobs, and they are not a
reasonable first milestone for a local machine or a EUR 1000 exploratory
compute cap.

The diagnostic component remains essential, but it is no longer the headline of
the work. It is the mechanism that makes the combination study defensible:
matched baselines, controlled ablations, collapse checks, timing, memory, and
frozen-probe readiness.

The supporting diagnostic question is:

> Under matched compute, does the V-JEPA masked-latent objective interact poorly
> with a pure VideoMamba encoder, and if so, is the issue caused by the
> backbone, predictor, target representation, or masking strategy?

## Why Full Pretraining Is Not The First Step

The original VideoMamba-JEPA setup includes a VideoMamba-Tiny pretraining config
with 16 nodes, 8 tasks per node, 300 epochs, 300 iterations per epoch, and
gradient accumulation. That is a large distributed pretraining setup.

With a EUR 1000 compute cap and the working assumption of roughly EUR 0.35 per
LUMI GPU-hour, the total budget is about 2800 LUMI GPU-hours before reserves.
After reserving 10% for setup/smoke work and 20% for reruns/failures/final
verification, only about 2000 LUMI GPU-hours should be treated as usable for
the first training matrix.

A 16-node LUMI-G run consumes GPU-hours quickly. It would be very easy to spend
the budget before learning whether the model, masks, predictor, and target
normalization are even technically stable together.

## Publication Positioning

I did not find a formal research-venue paper for the exact contribution
"VideoMamba trained with V-JEPA / V-JEPA-style masked latent prediction" in a
quick literature check on 2026-06-18. The exact prototype still appears to be
the small course-project repository rather than a venue publication:

- https://github.com/rolson24/videoMamba-jepa

Related but non-identical work exists:

- V-JEPA / V-JEPA 2 use ViT-based masked latent video prediction:
  https://arxiv.org/abs/2404.08471 and https://arxiv.org/abs/2506.09985
- V-JEPA code: https://github.com/facebookresearch/jepa
- VideoMamba is an SSM/Mamba video backbone:
  https://arxiv.org/abs/2403.06977
- VideoMamba code: https://github.com/OpenGVLab/VideoMamba
- VideoMAP is Mamba-based video pretraining, but it uses a hybrid
  Mamba-Transformer backbone and frame-wise autoregressive pretraining rather
  than the exact V-JEPA-on-VideoMamba formulation:
  https://arxiv.org/abs/2503.12332
- V-JEPA 2 code: https://github.com/facebookresearch/vjepa2
- A Hugging Face project studies BiMamba/Transformer JEPA-style masked latent
  prediction on toy-scale image/video benchmarks, but it is not the same as
  VideoMamba + V-JEPA and does not appear to be a venue publication:
  https://huggingface.co/2264K/bimamba-jepa-masked-latent-prediction

This makes a modest workshop/short-paper contribution plausible, if the claim is
phrased carefully. The contribution should be:

> We implement and evaluate VideoMamba with a V-JEPA-style masked latent
> prediction objective under a strict compute budget, and we report the
> stability, efficiency, and representation-readiness consequences of that
> combination.

It should not be:

> We achieve SOTA with VideoMamba-JEPA.

It should also avoid "first ever" language unless a final literature check
immediately before submission still supports that exact claim.

The paper can be framed around the combination while still asking controlled
questions:

- Can VideoMamba be trained with a V-JEPA-style masked latent objective at all
  under bounded compute?
- Does V-JEPA-style masked latent prediction collapse more easily for
  VideoMamba than for a ViT baseline under matched compute?
- Is instability caused by the VideoMamba encoder, the Mamba predictor, target
  scale/normalization, or the mask order?
- Does a lightweight attention predictor stabilize VideoMamba-JEPA?
- Do row/tube masks align better with Mamba's sequential inductive bias than
  V-JEPA multi-block masks?
- Is the method too inefficient under realistic compute constraints to justify
  larger-scale pretraining?

## Implemented Study Harness

The repository now includes a phase-1 diagnostic harness designed to prevent
accidental expensive pretraining and to make short runs measurable.

Implemented components:

- `docs/diagnostic_study.md`: operational runbook.
- `docs/local_generated_results.md`: committed local fixture results.
- `configs/diagnostics/base_short_pretrain.yaml`: short real-data baseline
  config template.
- `configs/diagnostics/phase1_matrix.yaml`: compute-bounded phase-1 matrix.
- `configs/diagnostics/local_generated_*.yaml`: local generated-video fixture
  configs.
- `diagnostics.smoke`: one-step synthetic smoke test.
- `diagnostics.run_matrix`: config generation plus budget gate.
- `diagnostics.summarize`: summary reducer for diagnostic CSV logs.
- `diagnostics.summarize_selected`: summary reducer for explicit CSV paths.
- `diagnostics.make_subset`: fixed-seed subset manifest creator.
- `diagnostics.make_synthetic_videos`: generated MP4 fixture creator.
- `src/masks/row_tube.py`: row/tube mask collator.
- Trainer routing so ViT configs use the ViT trainer and VideoMamba configs use
  the VideoMamba trainer.
- Local single-process fallback when distributed training is unavailable.
- Local compatibility fallbacks around missing fused Mamba/Triton helpers.
- End-to-end local training paths for VideoMamba with a V-JEPA-style masked
  latent objective.
- Enriched CSV logging for:
  - loss and JEPA loss
  - regularization loss
  - target variance and prediction variance
  - target and prediction activation magnitude
  - encoder and predictor gradient norms
  - memory
  - clips per second
  - per-step device-hours
  - GPU time and wall time

The compatibility fallbacks are only for local correctness checks. Serious
throughput claims require the intended optimized Mamba kernel stack.

## Why These Changes Were Made

The main goal is now to make the VideoMamba + V-JEPA combination credible as a
bounded empirical paper. The old project path encouraged jumping directly from a
prototype to expensive pretraining. That is risky because it mixes several
unknowns:

1. Whether the code runs on the intended hardware.
2. Whether VideoMamba and V-JEPA masks produce compatible tensor shapes.
3. Whether the predictor can learn the target representation.
4. Whether target scale causes unstable prediction.
5. Whether Mamba-oriented mask order matters.
6. Whether the method is efficient enough to justify cluster use.

The new harness separates these questions. It first tests mechanics, then tiny
sanity behavior, then short real-data ablations, and only then cluster-scale
runs. That makes the combination paper more credible because every reported
configuration can include exact compute cost, timing, variance/collapse signals,
and baseline comparisons.

The ablation matrix maps directly to hypotheses:

- ViT-Tiny vs VideoMamba-Tiny tests whether the encoder family is the issue.
- Mamba predictor vs attention predictor tests whether the predictor is the
  bottleneck.
- Raw targets vs layer-normalized targets tests whether target scale is a
  stability issue.
- Multi-block masks vs row/tube masks tests whether masking conflicts with
  Mamba's sequential inductive bias.

## Local Results So Far

### Synthetic Smoke Tests

Synthetic one-step smoke tests pass locally on `cuda:0` for:

- ViT-Tiny with V-JEPA-style multi-block masks.
- VideoMamba-Tiny with Mamba predictor and multi-block masks.
- VideoMamba-Tiny with attention predictor and row/tube masks.

These tests validate imports, model initialization, masks, one masked batch, one
optimizer step, checkpoint save/load, and a tiny frozen-probe step.

They are not evidence of representation quality, but they show that the
VideoMamba + V-JEPA-style training path is implemented well enough to run,
backpropagate, checkpoint, and log metrics locally.

Latest local smoke rerun on 2026-06-18:

| Run | Loss | Probe Loss | Target Var | Pred Var | Mem MB | Wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, attention predictor, multi-block masks | 0.81192 | 1.55794 | 0.74155 | 0.00437 | 151.24 | 3959 |
| VideoMamba-Tiny, Mamba predictor, multi-block masks | 0.83452 | 1.32384 | 0.60854 | 0.00356 | 172.50 | 5072 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.83039 | 1.79923 | 0.62248 | 0.00521 | 172.94 | 4780 |

The current workspace now contains a small fetched SSv2 real-video subset under
`diagnostic_runs/data/ssv2_subset`. This removes the immediate data blocker for
local sanity runs. It does not replace the need for a larger/full SSv2 manifest
before making paper-level empirical claims.

### Phase-1 Budget Gate Dry Run

The phase-1 matrix generator was rerun locally on 2026-06-18 with a conservative
placeholder of 5000 ms per microstep:

```bash
python -m diagnostics.run_matrix --write-configs --measured-ms 5000
```

This produced six planned runs and estimated:

- total estimated LUMI GPU-hours: 3.33
- total estimated cost: EUR 1.17
- budget status: `budget_ok: true`

This confirms that the budget-gate tooling is working. It is not a real cost
estimate for the paper yet, because the 5000 ms value came from synthetic smoke
checks rather than a real SSv2 subset benchmark.

### Generated-Video Trainer Fixture

A generated-video fixture was added to test the real video loader and full
training loop without requiring SSv2. This fixture uses generated MP4 files and
the same manifest/data-loader path expected by real video data.

Current local fixture results:

| Run | Loss | Target Var | Pred Var | Enc Grad | Pred Grad | Mem MB | Clips/s | Last Wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 0.81138 | 0.64777 | 0.00365 | 0.00436 | 0.38372 | 150.03 | 12.41 | 161 |
| VideoMamba-Tiny, Mamba predictor, multi-block masks | 0.80513 | 0.55110 | 0.00337 | 0.00016 | 0.42523 | 188.30 | 1.60 | 1248 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.80607 | 0.65284 | 0.00517 | 0.00755 | 0.38367 | 178.67 | 2.21 | 904 |

Interpretation:

- All three runs complete end to end.
- Target variance is nonzero in all cases.
- Prediction variance starts much lower than target variance, which is expected
  at initialization and should be tracked as a collapse/stability signal.
- The local VideoMamba fallback path is substantially slower than ViT. This is a
  warning signal, not a final efficiency result, because optimized Mamba kernels
  are not available in this local environment.

### Local SSv2 Subset Sanity Runs

A small real-video SSv2 subset was fetched from the Hugging Face mirror
`morpheushoc/something-something-v2`. The local fetch downloaded annotations,
the research license/download instructions, and video archive part `00`, then
stream-extracted 128 train-labelled `.webm` clips. A fixed-seed 32-clip subset
was used for the first real-video sanity run.

Commands:

```bash
python -m diagnostics.fetch_ssv2_hf_part --parts 00
python -m diagnostics.extract_ssv2_hf_subset --num-clips 128
python -m diagnostics.make_subset --input diagnostic_runs/data/ssv2_subset/ssv2_train_128.csv --output diagnostic_runs/data/ssv2_subset/ssv2_train_32.csv --num-clips 32 --seed 234 --require-existing
python -m app.main --fname configs/diagnostics/local_ssv2_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_videomamba.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_videomamba_attention_rowtube.yaml --devices cuda:0
```

Final logged rows from the 4-iteration sanity runs:

| Run | Loss | Target Var | Pred Var | Enc Grad | Pred Grad | Mem MB | Clips/s | Wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 0.83497 | 0.50183 | 0.00455 | 0.01246 | 0.45387 | 150.03 | 11.57 | 172 |
| VideoMamba-Tiny, Mamba predictor, multi-block masks | 0.85765 | 0.43825 | 0.00309 | 0.00020 | 0.47839 | 194.04 | 2.34 | 856 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.84220 | 0.44273 | 0.00432 | 0.01307 | 0.47432 | 178.67 | 2.19 | 914 |

Interpretation:

- The combined VideoMamba + V-JEPA-style path now runs on real SSv2 videos, not
  only synthetic tensors or generated MP4 fixtures.
- Target variance is nonzero and prediction variance is low but nonzero after
  four iterations.
- The Mamba-predictor run shows a much smaller encoder gradient norm than the
  ViT baseline and the attention-predictor row/tube variant. This is a useful
  real-video signal that motivated the longer 32-clip ablations and 128-clip
  subset runs below.
- These runs are still implementation/stability evidence, not representation
  quality or accuracy evidence.

A later local run repeated the three core configurations for three epochs over
the 32-clip subset, with 16 iterations per epoch and 48 optimizer steps.

Epoch-average loss:

| Run | Epoch 1 | Epoch 2 | Epoch 3 | Last Loss | Median Enc Grad |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 0.79536 | 0.76451 | 0.75326 | 0.75656 | 0.00496 |
| VideoMamba-Tiny, Mamba predictor, multi-block masks | 0.80997 | 0.77763 | 0.76700 | 0.75706 | 0.00028 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.79826 | 0.75613 | 0.74290 | 0.73873 | 0.00412 |

This is the first evidence beyond one-step or four-step checks. It supports the
practical feasibility claim: VideoMamba + a V-JEPA-style objective can run and
reduce loss on real SSv2 clips under the local bounded setup. It also sharpens
the main diagnostic lead: the Mamba-predictor configuration keeps encoder
gradients roughly an order of magnitude or more below the ViT baseline and the
attention-predictor row/tube variant, despite reducing loss. That is now a
priority hypothesis supported by the controlled ablations below.

Those ablations were then run under the same 32-clip, 48-step setup:

| Run | Last Loss | Min Loss | Epoch 3 Avg | Median Enc Grad | Last Pred Var |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT, multi-block, layer-norm | 0.75656 | 0.73006 | 0.75326 | 0.00496 | 0.00387 |
| VideoMamba, Mamba predictor, multi-block, layer-norm | 0.75706 | 0.75100 | 0.76700 | 0.00027 | 0.00445 |
| VideoMamba, Mamba predictor, multi-block, raw | 0.76255 | 0.73877 | 0.76418 | 0.00027 | 0.00445 |
| VideoMamba, Mamba predictor, row/tube, layer-norm | 0.76994 | 0.74210 | 0.76661 | 0.00021 | 0.00310 |
| VideoMamba, attention predictor, multi-block, layer-norm | 0.74248 | 0.72973 | 0.74189 | 0.00433 | 0.00319 |
| VideoMamba, attention predictor, row/tube, layer-norm | 0.73873 | 0.73137 | 0.74290 | 0.00412 | 0.00435 |

Current interpretation:

- The combination is locally feasible on real SSv2 clips.
- The best next candidate is VideoMamba with the attention predictor, not the
  inherited Mamba predictor.
- Target normalization is not the main explanation for the Mamba-predictor
  gradient issue in these small runs.
- Row/tube masks alone do not fix the Mamba-predictor gradient issue.
- The attention predictor makes both mask strategies look viable at this scale,
  with row/tube only slightly ahead by final loss.

The 128-clip subset was then used for the current larger local sanity run. The
same 128 extracted SSv2 clips were used for each configuration, with batch size
2, 4 frames, 64x64 crops, 64 iterations per epoch, 2 epochs, seed 234, and
`cuda:0`.

Commands:

```bash
python -m app.main --fname configs/diagnostics/local_ssv2_128_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_128_videomamba_attention_multiblock.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_128_videomamba_attention_rowtube.yaml --devices cuda:0
python -m diagnostics.summarize_selected --csv diagnostic_runs/local_ssv2_128/vit_multiblock/local_ssv2_128_vit_multiblock_r0.csv diagnostic_runs/local_ssv2_128/videomamba_attention_multiblock/local_ssv2_128_videomamba_attention_multiblock_r0.csv diagnostic_runs/local_ssv2_128/videomamba_attention_rowtube/local_ssv2_128_videomamba_attention_rowtube_r0.csv --output diagnostic_runs/local_ssv2_128/summary_current.json
```

Summary:

| Run | Steps | First Loss | Last Loss | Min Loss | Epoch 1 Avg | Epoch 2 Avg | Last Target Var | Last Pred Var | Median Enc Grad | Mean Clips/s | Device Hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 128 | 0.81706 | 0.71926 | 0.67479 | 0.75508 | 0.71596 | 0.43263 | 0.00275 | 0.00452 | 10.33 | 0.00710 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 128 | 0.81709 | 0.67468 | 0.65823 | 0.74364 | 0.68772 | 0.33866 | 0.00218 | 0.00309 | 2.26 | 0.03177 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 127 | 0.81188 | 0.68413 | 0.64509 | 0.74311 | 0.68609 | 0.34748 | 0.00281 | 0.00234 | 2.06 | 0.03587 |

The row/tube run completed both epochs and checkpointed, but the persisted CSV
is missing one epoch-2 logging row. The table therefore uses the actual CSV
rows as the source of record.

Current interpretation:

- The VideoMamba + V-JEPA combination now has real-video evidence beyond smoke
  tests: it runs for two epochs on 128 SSv2 clips and reduces loss.
- The attention-predictor VideoMamba variants outperform the ViT-Tiny baseline
  on loss in this small local setting. This should be presented only as
  bounded feasibility evidence, not as a quality or SOTA result.
- Both attention-predictor mask variants are viable at this scale. Multi-block
  has the better final logged loss; row/tube has the better minimum loss and
  very similar epoch-average loss.
- The key diagnostic result from the 32-clip ablations still matters: the
  inherited Mamba predictor shows very small encoder gradients, while the
  lightweight attention predictor gives a stronger training signal.
- The most reasonable next research path is therefore not full pretraining. It
  is a larger but still bounded SSv2 subset run with the attention-predictor
  VideoMamba variants, plus a proper held-out frozen probe.
- Local throughput remains unfavorable for VideoMamba. The observed local
  slowdown is a hardware/software-stack warning, not a final claim about
  optimized VideoMamba efficiency.

A frozen linear-probe overfit diagnostic was then added and run on the same
128-clip subset checkpoints. It loads the checkpoint `target_encoder`, freezes
the encoder, extracts mean-pooled features, and trains only a linear classifier.
Because the current subset is label-sparse, the diagnostic filters to labels
appearing at least twice. This leaves 72 clips across 28 classes.

Commands:

```bash
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_128_vit.yaml --checkpoint diagnostic_runs/local_ssv2_128/vit_multiblock/local_ssv2_128_vit_multiblock-latest.pth.tar --output diagnostic_runs/local_ssv2_128/probes/vit_multiblock_probe.json --batch-size 8 --probe-epochs 40 --min-label-count 2 --device cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_128_videomamba_attention_multiblock.yaml --checkpoint diagnostic_runs/local_ssv2_128/videomamba_attention_multiblock/local_ssv2_128_videomamba_attention_multiblock-latest.pth.tar --output diagnostic_runs/local_ssv2_128/probes/videomamba_attention_multiblock_probe.json --batch-size 8 --probe-epochs 40 --min-label-count 2 --device cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_128_videomamba_attention_rowtube.yaml --checkpoint diagnostic_runs/local_ssv2_128/videomamba_attention_rowtube/local_ssv2_128_videomamba_attention_rowtube-latest.pth.tar --output diagnostic_runs/local_ssv2_128/probes/videomamba_attention_rowtube_probe.json --batch-size 8 --probe-epochs 40 --min-label-count 2 --device cuda:0
```

Probe summary:

| Run | Probe Clips | Classes | Feature Var | Final Train Loss | Final Train Acc | Best Train Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 72 | 28 | 0.25256 | 0.59026 | 0.9167 | 0.9167 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 72 | 28 | 0.20995 | 1.13809 | 0.6944 | 0.7639 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 72 | 28 | 0.21039 | 1.13901 | 0.6944 | 0.7639 |

This result is useful because it proves the frozen-probe plumbing now works for
both encoder families. It is also a caution: after only two very short
pretraining epochs, the ViT baseline is easier to linearly fit than the
VideoMamba checkpoints on this tiny repeated-label subset. That should be read
as a diagnostic signal, not as an accuracy claim. The next publishable probe
requires a larger subset with enough repeated labels to support a real held-out
split.

The next local experiment used 512 SSv2 clips and three pretraining epochs. A
balanced held-out probe split was created with 32 classes, 3 train clips per
class, and 1 held-out clip per class.

Training summary:

| Run | Last Loss | Min Loss | Epoch 3 Avg | Last Pred Var | Median Enc Grad | Mean Clips/s | Device Hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 0.64137 | 0.42909 | 0.59857 | 0.04680 | 0.07028 | 9.33 | 0.04628 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 0.71896 | 0.41646 | 0.66162 | 0.00662 | 0.00924 | 2.20 | 0.19795 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.70946 | 0.48851 | 0.66654 | 0.00654 | 0.00744 | 2.36 | 0.18174 |

Held-out probe summary:

| Run | Train Clips | Eval Clips | Classes | Best Train Acc | Eval Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 96 | 32 | 32 | 0.9271 | 0.0312 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 96 | 32 | 32 | 0.8438 | 0.0000 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 96 | 32 | 32 | 0.8333 | 0.0000 |

This strengthens two conclusions. First, local experimentation is viable: the
laptop completed the 512-clip three-epoch matrix and held-out probes in a
reasonable amount of time. Second, the current VideoMamba variants are not yet
showing a stronger representation signal than ViT. Their loss can decrease, but
their final prediction variance and encoder-gradient scale remain much lower
than ViT's. The held-out split is too small to be an accuracy claim, but the
chance-level probe result says the next probe needs fewer classes with more
examples per class, or a larger extracted subset.

A denser 8-class split was then tested from the same 512 clips, with 6 train
clips and 2 held-out clips per class. It still produced only 1/16 held-out
correct for each model:

| Run | Train Clips | Eval Clips | Classes | Best Train Acc | Eval Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 48 | 16 | 8 | 0.8333 | 0.0625 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 48 | 16 | 8 | 0.8750 | 0.0625 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 48 | 16 | 8 | 0.8750 | 0.0625 |

This changes the next-step diagnosis. The main issue is not only that the first
probe had too many classes. These short local checkpoints do not yet provide
features that transfer to held-out clips under a linear probe. The paper can
still report this as useful bounded evidence, but a positive representation
claim now requires longer pretraining, more data, better probe design, or some
combination of those.

A 10-epoch follow-up on the same 512 clips was then run for ViT-Tiny with
multi-block masks and VideoMamba-Tiny with the attention predictor and row/tube
masks. The purpose was to test the simplest alternative explanation: maybe the
3-epoch runs were just too short for the frozen probe.

Training summary:

| Run | Steps | Epoch 10 Avg Loss | Min Loss | Epoch 10 Pred Var Avg | Median Enc Grad | Mean Clips/s | Device Hours | Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks, 10 epochs | 2560 | 0.38251 | 0.16541 | 0.08338 | 0.20322 | 9.50 | 0.15204 | 546.1 |
| VideoMamba-Tiny, attention predictor, row/tube masks, 10 epochs | 2558 | 0.71806 | 0.38167 | 0.03197 | 0.06670 | 2.08 | 0.68946 | 2480.8 |

Held-out probe summary:

| Run | Train Clips | Eval Clips | Classes | Best Train Acc | Eval Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, 10 epochs, 32-class split | 96 | 32 | 32 | 0.4896 | 0.0625 |
| VideoMamba-Tiny row/tube, 10 epochs, 32-class split | 96 | 32 | 32 | 0.7396 | 0.0625 |
| ViT-Tiny, 10 epochs, 8-class split | 48 | 16 | 8 | 0.6250 | 0.1250 |
| VideoMamba-Tiny row/tube, 10 epochs, 8-class split | 48 | 16 | 8 | 0.7292 | 0.0000 |

This is a useful negative result. Longer ViT training clearly improves the
masked-latent loss on the 512-clip subset, but it does not produce a useful
held-out linear probe under the current split. Longer VideoMamba training is
stable, but its loss plateaus and remains noisy; its prediction variance
improves relative to the 3-epoch checkpoint, but the probe still fails. The
immediate conclusion is that merely running the same 512 clips longer is not
the right next step. The next experiment should either improve the data/probe
protocol or test a concrete VideoMamba + V-JEPA modification.

A probe-sensitivity check then reran the 10-epoch frozen probes with
`--probe-lr 0.1` and `--probe-epochs 300`. This stronger linear head fits the
tiny train splits much better, but it does not create useful held-out accuracy:

| Run | Train Clips | Eval Clips | Classes | Best Train Acc | Eval Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, 10 epochs, 32-class strong probe | 96 | 32 | 32 | 0.7188 | 0.0312 |
| VideoMamba-Tiny row/tube, 10 epochs, 32-class strong probe | 96 | 32 | 32 | 0.8854 | 0.0938 |
| ViT-Tiny, 10 epochs, 8-class strong probe | 48 | 16 | 8 | 0.9375 | 0.1250 |
| VideoMamba-Tiny row/tube, 10 epochs, 8-class strong probe | 48 | 16 | 8 | 1.0000 | 0.0625 |

This means the weak held-out results are not simply because the probe optimizer
was too timid. The probe can memorize the tiny training split, especially for
VideoMamba, but the learned linear boundary does not transfer to held-out clips.
That is a data/probe/generalization problem that must be solved before claiming
representation quality.

The next local step extracted a 2048-clip SSv2 subset from the available archive
part and built more balanced probe splits. This produced 174 classes total, 55
classes with at least 15 clips, and 27 classes with at least 20 clips. Two
probe splits were used: 16 classes with 15 train and 5 held-out clips per class,
and 32 classes with 10 train and 5 held-out clips per class.

Training summary:

| Run | Epoch 3 Avg Loss | Min Loss | Last Pred Var | Median Enc Grad | Mean Clips/s | Device Hours | Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, 2048 clips | 0.53257 | 0.19180 | 0.21356 | 0.17511 | 6.80 | 0.25927 | 931.8 |
| VideoMamba-Tiny row/tube, 2048 clips | 0.72566 | 0.35599 | 0.01102 | 0.07481 | 1.62 | 1.06112 | 3818.5 |

Held-out probe summary:

| Run | Split | Train Clips | Eval Clips | Classes | Best Train Acc | Eval Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny 2048 | 16c 15x5 | 240 | 80 | 16 | 0.3625 | 0.0750 |
| VideoMamba-Tiny row/tube 2048 | 16c 15x5 | 240 | 80 | 16 | 0.4292 | 0.0875 |
| ViT-Tiny 2048 | 32c 10x5 | 320 | 160 | 32 | 0.2906 | 0.0125 |
| VideoMamba-Tiny row/tube 2048 | 32c 10x5 | 320 | 160 | 32 | 0.3781 | 0.0375 |

This gives the clearest local conclusion so far. The implementation is viable
and the comparison is reproducible, but the current VideoMamba + V-JEPA setup
does not yet show useful held-out representation quality. Increasing the local
subset from 512 to 2048 clips and making the probe splits less sparse did not
solve the problem. ViT still has healthier loss and prediction-variance
behavior; VideoMamba remains stable but appears underdriven by the objective.

This is enough for a humble diagnostic/feasibility paper if the claim is
framed around implementation, bounded compute, and failure-mode evidence. It is
not enough for a positive "VideoMamba-JEPA works well" paper. The next research
step should be either a targeted objective modification or selective scaling on
LUMI.

## Local Versus Cluster Execution

The next stage should still be local or on a cheap CUDA machine. The 32-clip
and 128-clip SSv2 sanity sets now exist and have run, the frozen-probe plumbing
has been tested, and the 512-clip 10-epoch follow-up has completed locally. The
next purpose is to improve the evidence source, not to scale blindly. A larger
and more label-balanced subset, a better validated probe protocol, or a focused
objective modification is more informative than repeating the same 512 clips for
more epochs.

The measured local machine is an NVIDIA GeForce RTX 3070 Laptop GPU with 8 GB
VRAM. On the 128-clip subset, ViT-Tiny runs at about 10 clips/s and
VideoMamba-Tiny with an attention predictor runs at about 2 clips/s. That is
enough for 512-2048 clip subset experiments and a modest workshop/short-paper
study, but not enough for full SSv2 pretraining or SOTA claims. The detailed
timeline and local compute accounting are in `docs/experiment_timeline.md`.
Continuation guidance is summarized in `docs/research_handoff.md`.

LUMI should only be used after:

- Local or cheap-CUDA smoke tests pass.
- A real SSv2 subset run produces valid CSV logs.
- Measured microstep time is entered into `diagnostics.run_matrix`.
- The generated matrix remains under the post-reserve compute cap.
- A LUMI/ROCm smoke benchmark confirms that the needed Mamba dependencies work.

LUMI is specifically risky because the project inherits CUDA/Triton-oriented
Mamba components, while LUMI uses AMD MI250x/ROCm. If optimized kernels fail on
LUMI, the project should either use a CUDA environment for the pilot or treat
kernel portability as part of the research engineering work.

The practical access and compute-planning details are now separated into
`docs/hpc_lumi_usage.md`. That document records the public UT/Rocket/LUMI price
assumptions checked on 2026-06-20, the EUR 1000 budget conversion, and a
conference-oriented estimate: about 190 single-GPU-equivalent hours for an
8192-clip / 20-epoch / three-configuration / three-seed small-geometry stage,
with a more realistic final HPC attempt budgeted at 1500-1800 accounted LUMI
GPU-hours including setup, reruns, and verification.

The 2048-clip local runs now satisfy the local evidence gate: code, data,
logging, checkpointing, and probing are all exercised on real SSv2 video. If
the desired paper needs a positive performance result, the project should now
transition to the LUMI smoke gate described in `docs/lumi_transition_plan.md`.

## Next Steps

The working rule is: continue locally as long as local experiments can answer
pipeline, probe, data, and small-configuration questions. If these become
sufficient for a modest workshop/short-paper, stop and write. If they remain
insufficient but the bottleneck is clearly data scale, training duration,
sequence length, or seeds, transition to LUMI or another larger GPU environment
with only the best 2-3 configurations.

1. Replace the overfit probe with a held-out frozen probe.
   - The 512-clip held-out probe path now works, including a denser 8-class
     split, but the current checkpoints remain at chance-level evaluation.
   - A 10-epoch follow-up also failed to produce useful held-out probe results.
   - Do not spend effort only reshuffling these same checkpoints or running the
     same 512 clips longer.
   - Improve the evidence source first: larger/better-balanced data, a probe
     protocol check, or a targeted objective modification.
   - Keep the same fixed seed and manifest construction.
   - Create a balanced probe train/eval split with
     `diagnostics.make_balanced_probe_split`.
   - Report probe setup, labels used, train/eval split, and confirmation that
     the encoder is frozen.

2. Prepare a larger/full SSv2 manifest for paper-level runs.
   - Use `diagnostics.make_subset` for fixed-seed subsets.
   - Keep the seed fixed.
   - Verify all files exist with `--require-existing`.

3. Extend the real SSv2 subset runs only after frozen-probe wiring is clear.
   - Next practical subset size: 256-512 clips if storage and runtime are
     acceptable.
   - Prioritize ViT-Tiny multi-block, VideoMamba attention multi-block, and
     VideoMamba attention row/tube.
   - Re-run the inherited VideoMamba Mamba-predictor configuration only as a
     diagnostic comparison, because the current gradient signal is weak.

4. Check acceptance criteria.
   - No NaNs.
   - Target and prediction shapes align.
   - Prediction variance does not collapse to near zero across repeated steps.
   - Gradient norms are finite.
   - Checkpoints save and reload.
   - Microstep timing is stable after the first warmup step.

5. Keep the tiny overfit tests as regression checks.
   - The 32-clip and 128-clip runs are now the local reference.
   - Re-run them after trainer, mask, probe, or data-loader changes.
   - If a later VideoMamba run fails while ViT still succeeds, inspect predictor
     and mask variants before increasing compute.

6. Generate the phase-1 matrix.
   - Use measured real-video microstep time.
   - Stop if `diagnostics.run_matrix` reports the matrix exceeds budget.

7. Only then consider LUMI.
   - First run a LUMI smoke benchmark.
   - Verify ROCm compatibility and Mamba kernels.
   - Reserve at least 20% of the EUR 1000 budget for reruns and final checks.
   - If more LUMI compute becomes available, still scale selectively: run the
     best understood configurations first, with fixed manifests and early stop
     criteria, before expanding the matrix.

8. Draft the paper around the combination and use the diagnostics as evidence.
   - Main claim: VideoMamba can be evaluated under a V-JEPA-style masked latent
     objective with a reproducible bounded-compute protocol.
   - Results claim: report whichever outcome the SSv2 subset experiments
     support, whether stable training, a stabilizing modification, or a
     reproducible failure mode.
   - Related-work claim: exact venue publication not found as of the latest
     check, but final submission text should be rechecked before using strong
     novelty language.

## Paper Decision Rule

Proceed toward a workshop or short-paper submission only if phase 1 produces one
of the following:

- VideoMamba + V-JEPA trains stably enough to report bounded empirical results.
- A small modification, such as an attention predictor, target normalization, or
  row/tube masks, clearly improves stability or frozen-probe readiness.
- The combination fails in a reproducible and informative way while the ViT
  baseline succeeds.

The paper may be framed as VideoMamba + V-JEPA. The careful version of that
framing is "first-pass implementation, feasibility study, and bounded empirical
evaluation," not "state-of-the-art VideoMamba-JEPA pretraining."
