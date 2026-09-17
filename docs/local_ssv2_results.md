# Local SSv2 Subset Sanity Results

These results are from a small real-video Something-Something V2 subset. They
are implementation and stability evidence only. They are not accuracy results
and should not be interpreted as representation quality.

## Data Fetch

The official Something-Something V2 dataset is research-use data. For this local
sanity run, the annotations, research license, download instructions, and video
archive part `00` were fetched from the Hugging Face mirror:

- https://huggingface.co/datasets/morpheushoc/something-something-v2

Only archive part `00` was downloaded locally. It is a 1 GB split gzip-tar part
from the full 19.5 GB dataset mirror. The helper extracts early train-labelled
`.webm` members without downloading the full archive.

Commands:

```bash
python -m diagnostics.fetch_ssv2_hf_part --parts 00
python -m diagnostics.extract_ssv2_hf_subset --num-clips 128
python -m diagnostics.make_subset --input diagnostic_runs/data/ssv2_subset/ssv2_train_128.csv --output diagnostic_runs/data/ssv2_subset/ssv2_train_32.csv --num-clips 32 --seed 234 --require-existing
```

Local data created:

- raw annotations/license/archive part:
  `diagnostic_runs/data/ssv2_hf_raw`
- extracted videos:
  `diagnostic_runs/data/ssv2_subset/videos`
- 128-clip manifest:
  `diagnostic_runs/data/ssv2_subset/ssv2_train_128.csv`
- fixed 32-clip sanity manifest:
  `diagnostic_runs/data/ssv2_subset/ssv2_train_32.csv`

## Sanity Configs

These configs use 32 real SSv2 clips, batch size 2, 4 frames, 64x64 crops, 4
iterations, 1 epoch, and `cuda:0`.

```bash
python -m app.main --fname configs/diagnostics/local_ssv2_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_videomamba.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_videomamba_attention_rowtube.yaml --devices cuda:0
```

## Current Result Summary

The table reports the final logged row from each 4-iteration run.

| Run | Loss | Target Var | Pred Var | Enc Grad | Pred Grad | Mem MB | Clips/s | Wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 0.83497 | 0.50183 | 0.00455 | 0.01246 | 0.45387 | 150.03 | 11.57 | 172 |
| VideoMamba-Tiny, Mamba predictor, multi-block masks | 0.85765 | 0.43825 | 0.00309 | 0.00020 | 0.47839 | 194.04 | 2.34 | 856 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.84220 | 0.44273 | 0.00432 | 0.01307 | 0.47432 | 178.67 | 2.19 | 914 |

## Interpretation

All three real-video sanity runs completed. This confirms that the fetched SSv2
subset works with the real video decoder, manifest path, transforms, masks,
trainer loop, diagnostic logging, and checkpoints.

Target variance is nonzero in all runs, and prediction variance remains low but
nonzero after four iterations. This is expected at initialization and should be
tracked over longer real-video runs as the collapse signal.

The Mamba-predictor run again shows a much smaller encoder gradient norm than
the ViT baseline and the attention-predictor row/tube variant. This is not yet a
scientific conclusion, because the run is intentionally tiny, but it is now a
real-video signal that was tested in the longer 32-clip and 128-clip phases
below.

The local VideoMamba timing is slower than ViT. This remains a local fallback
environment result, not a final efficiency claim.

## 32-Clip Overfit/Sanity Run

The next local step ran three epochs over the fixed 32-clip real-video subset.
Each run used batch size 2, 16 iterations per epoch, 48 total optimizer steps,
4 frames, 64x64 crops, and `cuda:0`.

Commands:

```bash
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba_raw.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba_mamba_rowtube.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba_attention_multiblock.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba_attention_rowtube.yaml --devices cuda:0
```

### Core Three-Run Summary

Final logged rows for the initial core comparison:

| Run | Loss | Target Var | Pred Var | Enc Grad | Pred Grad | Mem MB | Clips/s | Device Hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 0.75656 | 0.54972 | 0.00387 | 0.00508 | 0.39358 | 150.04 | 10.66 | 0.00286 |
| VideoMamba-Tiny, Mamba predictor, multi-block masks | 0.75706 | 0.61284 | 0.00445 | 0.00030 | 0.35357 | 194.96 | 2.01 | 0.01324 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.73873 | 0.53437 | 0.00435 | 0.00207 | 0.40579 | 178.67 | 2.22 | 0.01175 |

Loss trend:

| Run | First Loss | Last Loss | Min Loss | Epoch 1 Avg | Epoch 2 Avg | Epoch 3 Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 0.82152 | 0.75656 | 0.73006 | 0.79536 | 0.76451 | 0.75326 |
| VideoMamba-Tiny, Mamba predictor, multi-block masks | 0.82768 | 0.75706 | 0.75100 | 0.80997 | 0.77763 | 0.76700 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.81998 | 0.73873 | 0.73137 | 0.79826 | 0.75613 | 0.74290 |

Gradient and throughput trend:

| Run | First Enc Grad | Last Enc Grad | Median Enc Grad | Median Pred Grad | Mean Clips/s Excluding First Step |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 0.00431 | 0.00508 | 0.00496 | 0.37345 | 10.71 |
| VideoMamba-Tiny, Mamba predictor, multi-block masks | 0.00014 | 0.00030 | 0.00028 | 0.42802 | 2.05 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.00316 | 0.00207 | 0.00412 | 0.41631 | 2.32 |

Interpretation:

- All three configurations reduce loss over 48 real-video optimizer steps.
- None of the three runs shows NaNs or immediate prediction-variance collapse.
- The attention-predictor row/tube VideoMamba run has the best final loss in
  this tiny setting.
- The Mamba-predictor VideoMamba run learns in terms of loss, but its encoder
  gradient norm remains much smaller than the ViT baseline and the
  attention-predictor variant. This strengthens the hypothesis that the Mamba
  predictor path may bottleneck signal into the encoder, or that its gradient
  scale needs separate normalization/tuning.
- Throughput remains much lower for VideoMamba than ViT in this local fallback
  environment. This should not be treated as a final efficiency result until
  the optimized Mamba kernel stack is tested.

### Controlled Ablations

The same 32-clip, 48-step setup was then extended to isolate three factors:

- target transform: layer-normalized target tokens vs raw target tokens
- predictor type: Mamba predictor vs lightweight attention predictor
- mask type: V-JEPA-style multi-block masks vs row/tube masks

Summary:

| Run | First Loss | Last Loss | Min Loss | Epoch 1 Avg | Epoch 2 Avg | Epoch 3 Avg | Median Enc Grad | Last Pred Var | Mean Clips/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT, multi-block, layer-norm | 0.82152 | 0.75656 | 0.73006 | 0.79537 | 0.76451 | 0.75326 | 0.00496 | 0.00387 | 10.71 |
| VideoMamba, Mamba predictor, multi-block, layer-norm | 0.82768 | 0.75706 | 0.75100 | 0.80997 | 0.77763 | 0.76700 | 0.00027 | 0.00445 | 2.05 |
| VideoMamba, Mamba predictor, multi-block, raw | 0.82303 | 0.76255 | 0.73877 | 0.81118 | 0.77528 | 0.76418 | 0.00027 | 0.00445 | 2.16 |
| VideoMamba, Mamba predictor, row/tube, layer-norm | 0.82430 | 0.76994 | 0.74210 | 0.81405 | 0.77679 | 0.76661 | 0.00021 | 0.00310 | 2.07 |
| VideoMamba, attention predictor, multi-block, layer-norm | 0.82021 | 0.74248 | 0.72973 | 0.79475 | 0.75691 | 0.74189 | 0.00433 | 0.00319 | 2.29 |
| VideoMamba, attention predictor, row/tube, layer-norm | 0.81998 | 0.73873 | 0.73137 | 0.79826 | 0.75613 | 0.74290 | 0.00412 | 0.00435 | 2.32 |

Ablation interpretation:

- Target normalization does not appear to be the main issue in this tiny setup.
  Raw targets slightly worsen final loss versus layer-normalized targets for the
  Mamba-predictor multiblock run, and the encoder-gradient scale remains almost
  unchanged.
- Predictor type is the strongest signal so far. With the same multi-block
  masks, the attention predictor gives lower final loss and a much larger
  median encoder-gradient norm than the Mamba predictor.
- Row/tube masks do not fix the Mamba-predictor gradient issue. With the Mamba
  predictor, row/tube masks have slightly worse final loss and similarly tiny
  encoder gradients.
- With the attention predictor, row/tube and multi-block masks are close. The
  row/tube variant has the best final loss, but the difference is too small and
  the run too tiny to claim superiority.
- The best immediate next candidate for a larger 128-clip subset run is
  VideoMamba-Tiny with the attention predictor, testing both multi-block and
  row/tube masks.

## 128-Clip Subset Sanity Run

The next local step used the 128 extracted SSv2 clips. Each run used batch size
2, 4 frames, 64x64 crops, 64 iterations per epoch, 2 epochs, seed 234, and
`cuda:0`.

Commands:

```bash
python -m app.main --fname configs/diagnostics/local_ssv2_128_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_128_videomamba_attention_multiblock.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_128_videomamba_attention_rowtube.yaml --devices cuda:0
python -m diagnostics.summarize_selected --csv diagnostic_runs/local_ssv2_128/vit_multiblock/local_ssv2_128_vit_multiblock_r0.csv diagnostic_runs/local_ssv2_128/videomamba_attention_multiblock/local_ssv2_128_videomamba_attention_multiblock_r0.csv diagnostic_runs/local_ssv2_128/videomamba_attention_rowtube/local_ssv2_128_videomamba_attention_rowtube_r0.csv --output diagnostic_runs/local_ssv2_128/summary_current.json
```

Summary from the persisted CSV logs:

| Run | Steps | First Loss | Last Loss | Min Loss | Epoch 1 Avg | Epoch 2 Avg | Last Target Var | Last Pred Var | Median Enc Grad | Median Pred Grad | Mean Clips/s | Max Mem MB | Device Hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 128 | 0.81706 | 0.71926 | 0.67479 | 0.75508 | 0.71596 | 0.43263 | 0.00275 | 0.00452 | 0.34262 | 10.33 | 150.04 | 0.00710 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 128 | 0.81709 | 0.67468 | 0.65823 | 0.74364 | 0.68772 | 0.33866 | 0.00218 | 0.00309 | 0.36836 | 2.26 | 206.30 | 0.03177 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 127 | 0.81188 | 0.68413 | 0.64509 | 0.74311 | 0.68609 | 0.34748 | 0.00281 | 0.00234 | 0.34692 | 2.06 | 178.67 | 0.03587 |

The row/tube run completed both epochs and checkpointed, but the CSV is missing
one logging row from epoch 2. The table reports the persisted CSV evidence
rather than the console output.

Interpretation:

- All three 128-clip runs reduce loss over two epochs on real SSv2 clips.
- No NaNs were observed in the persisted logs.
- Target variance remains nonzero and prediction variance remains low but
  nonzero, so these runs do not show immediate representation collapse.
- VideoMamba-Tiny with the lightweight attention predictor has lower final and
  epoch-average loss than the ViT-Tiny baseline in this small local subset.
  This is useful feasibility evidence, not a representation-quality claim.
- The two VideoMamba attention-predictor mask variants are close. Multi-block
  has the better final logged loss, while row/tube has the better minimum loss
  and similar epoch-average loss. This is not enough to claim that either mask
  strategy is superior.
- Local VideoMamba throughput remains about 4-5x slower than ViT in this
  environment. This should be treated as a local implementation/kernel warning,
  not a final efficiency result.
- The next evidence gap is a proper held-out frozen probe on a larger subset,
  not more generated-video or synthetic checks.

## Frozen Linear-Probe Overfit Diagnostic

A small frozen-probe diagnostic was added after the 128-clip runs. It loads the
`target_encoder` from each checkpoint, freezes it, extracts mean-pooled features
for the local SSv2 subset, and trains only a linear head.

The current 128-clip subset is label-sparse: 128 clips contain 84 labels, and
56 labels occur only once. To avoid pure singleton memorization, the diagnostic
filters to labels that occur at least twice. This leaves 72 clips across 28
classes. The probe is still an overfit/plumbing check, not a held-out accuracy
result.

Commands:

```bash
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_128_vit.yaml --checkpoint diagnostic_runs/local_ssv2_128/vit_multiblock/local_ssv2_128_vit_multiblock-latest.pth.tar --output diagnostic_runs/local_ssv2_128/probes/vit_multiblock_probe.json --batch-size 8 --probe-epochs 40 --min-label-count 2 --device cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_128_videomamba_attention_multiblock.yaml --checkpoint diagnostic_runs/local_ssv2_128/videomamba_attention_multiblock/local_ssv2_128_videomamba_attention_multiblock-latest.pth.tar --output diagnostic_runs/local_ssv2_128/probes/videomamba_attention_multiblock_probe.json --batch-size 8 --probe-epochs 40 --min-label-count 2 --device cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_128_videomamba_attention_rowtube.yaml --checkpoint diagnostic_runs/local_ssv2_128/videomamba_attention_rowtube/local_ssv2_128_videomamba_attention_rowtube-latest.pth.tar --output diagnostic_runs/local_ssv2_128/probes/videomamba_attention_rowtube_probe.json --batch-size 8 --probe-epochs 40 --min-label-count 2 --device cuda:0
```

Results:

| Run | Probe Clips | Classes | Feature Var | Final Train Loss | Final Train Acc | Best Train Acc | Feature Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 72 | 28 | 0.25256 | 0.59026 | 0.9167 | 0.9167 | 1.85 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 72 | 28 | 0.20995 | 1.13809 | 0.6944 | 0.7639 | 4.04 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 72 | 28 | 0.21039 | 1.13901 | 0.6944 | 0.7639 | 4.00 |

Interpretation:

- The probe path works for both ViT and VideoMamba checkpoints: checkpoint load,
  frozen feature extraction, mean pooling, linear-head training, and JSON/CSV
  logging all complete locally.
- The ViT checkpoint is easier for the tiny linear head to fit than either
  VideoMamba checkpoint in this overfit diagnostic.
- The two VideoMamba attention-predictor mask variants are essentially tied by
  this diagnostic.
- Because this is train-on-subset overfit behavior on 72 clips, it should not
  be described as validation accuracy. The next publishable probe step needs a
  larger subset with a real held-out split and enough repeated labels per class.

## 512-Clip Subset And Held-Out Probe Diagnostic

The next local run extracted 512 SSv2 clips from the same archive part and ran a
larger three-epoch comparison. The balanced probe split was built from the
512-clip manifest with 32 classes, 3 train clips per class, and 1 held-out clip
per class.

Commands:

```bash
python -m diagnostics.extract_ssv2_hf_subset --num-clips 512 --manifest diagnostic_runs/data/ssv2_subset/ssv2_train_512.csv
python -m diagnostics.make_balanced_probe_split --input diagnostic_runs/data/ssv2_subset/ssv2_train_512.csv --train-output diagnostic_runs/data/ssv2_subset/ssv2_probe_train_balanced.csv --eval-output diagnostic_runs/data/ssv2_subset/ssv2_probe_eval_balanced.csv --train-per-class 3 --eval-per-class 1 --max-classes 32 --seed 234
python -m app.main --fname configs/diagnostics/local_ssv2_512_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_512_videomamba_attention_multiblock.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_512_videomamba_attention_rowtube.yaml --devices cuda:0
```

Training summary:

| Run | Steps | First Loss | Last Loss | Min Loss | Epoch 1 Avg | Epoch 2 Avg | Epoch 3 Avg | Last Target Var | Last Pred Var | Median Enc Grad | Mean Clips/s | Max Mem MB | Device Hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 768 | 0.82369 | 0.64137 | 0.42909 | 0.69925 | 0.62642 | 0.59857 | 0.72352 | 0.04680 | 0.07028 | 9.33 | 150.04 | 0.04628 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 768 | 0.82047 | 0.71896 | 0.41646 | 0.66616 | 0.64917 | 0.66162 | 0.72574 | 0.00662 | 0.00924 | 2.20 | 206.30 | 0.19795 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 768 | 0.81131 | 0.70946 | 0.48851 | 0.66888 | 0.65264 | 0.66654 | 0.75085 | 0.00654 | 0.00744 | 2.36 | 178.67 | 0.18174 |

Held-out probe commands:

```bash
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_512_vit.yaml --checkpoint diagnostic_runs/local_ssv2_512/vit_multiblock/local_ssv2_512_vit_multiblock-latest.pth.tar --manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_train_balanced.csv --eval-manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_eval_balanced.csv --output diagnostic_runs/local_ssv2_512/probes/vit_multiblock_probe.json --batch-size 8 --probe-epochs 80 --min-label-count 1 --device cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_512_videomamba_attention_multiblock.yaml --checkpoint diagnostic_runs/local_ssv2_512/videomamba_attention_multiblock/local_ssv2_512_videomamba_attention_multiblock-latest.pth.tar --manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_train_balanced.csv --eval-manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_eval_balanced.csv --output diagnostic_runs/local_ssv2_512/probes/videomamba_attention_multiblock_probe.json --batch-size 8 --probe-epochs 80 --min-label-count 1 --device cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_512_videomamba_attention_rowtube.yaml --checkpoint diagnostic_runs/local_ssv2_512/videomamba_attention_rowtube/local_ssv2_512_videomamba_attention_rowtube-latest.pth.tar --manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_train_balanced.csv --eval-manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_eval_balanced.csv --output diagnostic_runs/local_ssv2_512/probes/videomamba_attention_rowtube_probe.json --batch-size 8 --probe-epochs 80 --min-label-count 1 --device cuda:0
```

Probe summary:

| Run | Train Clips | Eval Clips | Classes | Final Train Acc | Best Train Acc | Eval Acc | Eval Loss | Feature Var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 96 | 32 | 32 | 0.8229 | 0.9271 | 0.0312 | 7.5585 | 0.38754 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 96 | 32 | 32 | 0.8021 | 0.8438 | 0.0000 | 6.4880 | 0.27995 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 96 | 32 | 32 | 0.7917 | 0.8333 | 0.0000 | 6.3463 | 0.27954 |

Interpretation:

- The 512-clip local run confirms that this laptop can run a larger bounded
  subset experiment end to end.
- ViT-Tiny has the best epoch-3 loss and much stronger median encoder-gradient
  scale.
- VideoMamba attention variants reduce loss but plateau higher by epoch 3.
  Their final prediction variance is much lower than ViT's, and encoder-gradient
  scale remains lower. This is now a consistent diagnostic concern.
- Multi-block and row/tube VideoMamba runs are close. Row/tube is slightly
  faster and uses less memory, but it does not improve the probe outcome.
- The held-out probe infrastructure works, but 3 train clips per class is not
  enough for a meaningful accuracy claim. Held-out accuracy is at chance for all
  configurations.
- The next paper-relevant probe should use fewer classes with more clips per
  class, or a larger extracted subset, so each class has enough examples for
  both training and evaluation.

### Denser 8-Class Probe Split

To check whether the chance-level 32-class probe was mainly caused by too few
examples per class, a second split was built from the same 512 clips with 8
classes, 6 train clips per class, and 2 held-out clips per class.

Command:

```bash
python -m diagnostics.make_balanced_probe_split --input diagnostic_runs/data/ssv2_subset/ssv2_train_512.csv --train-output diagnostic_runs/data/ssv2_subset/ssv2_probe_train_balanced_8c.csv --eval-output diagnostic_runs/data/ssv2_subset/ssv2_probe_eval_balanced_8c.csv --train-per-class 6 --eval-per-class 2 --max-classes 8 --seed 234
```

Probe summary:

| Run | Train Clips | Eval Clips | Classes | Final Train Acc | Best Train Acc | Eval Acc | Eval Loss | Feature Var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 48 | 16 | 8 | 0.8333 | 0.8333 | 0.0625 | 4.2282 | 0.34793 |
| VideoMamba-Tiny, attention predictor, multi-block masks | 48 | 16 | 8 | 0.8333 | 0.8750 | 0.0625 | 3.8172 | 0.24201 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 48 | 16 | 8 | 0.8125 | 0.8750 | 0.0625 | 3.7297 | 0.23904 |

Interpretation:

- The denser split did not produce meaningful held-out accuracy either. Each
  model gets 1 of 16 held-out clips correct.
- The linear heads fit the tiny train split moderately well, so the probe code
  and label mapping are working, but the representations after this short
  pretraining run do not generalize on this split.
- This does not invalidate the VideoMamba + V-JEPA paper direction. It means
  the current local setting should be reported as feasibility and diagnostic
  evidence, not as representation-performance evidence.
- A paper-level probe likely needs either longer pretraining, a larger/better
  balanced subset, stronger augment/eval protocol, or all three.

## 512-Clip 10-Epoch Follow-Up

To test whether the chance-level held-out probes were mainly caused by too few
pretraining epochs, a longer 512-clip run was added for the ViT-Tiny baseline
and the faster VideoMamba attention-predictor row/tube configuration. The
multi-block VideoMamba variant was not repeated in this follow-up because the
3-epoch row/tube and multi-block runs were close, while row/tube was faster and
used less memory.

Commands:

```bash
python -m app.main --fname configs/diagnostics/local_ssv2_512_long_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_512_long_videomamba_attention_rowtube.yaml --devices cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_512_long_vit.yaml --checkpoint diagnostic_runs/local_ssv2_512_long/vit_multiblock_10ep/local_ssv2_512_vit_multiblock_10ep-latest.pth.tar --manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_train_balanced.csv --eval-manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_eval_balanced.csv --output diagnostic_runs/local_ssv2_512_long/probes/vit_10ep_32c_probe.json --batch-size 8 --probe-epochs 80 --min-label-count 1 --device cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_512_long_videomamba_attention_rowtube.yaml --checkpoint diagnostic_runs/local_ssv2_512_long/videomamba_attention_rowtube_10ep/local_ssv2_512_videomamba_attention_rowtube_10ep-latest.pth.tar --manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_train_balanced.csv --eval-manifest diagnostic_runs/data/ssv2_subset/ssv2_probe_eval_balanced.csv --output diagnostic_runs/local_ssv2_512_long/probes/videomamba_rowtube_10ep_32c_probe.json --batch-size 8 --probe-epochs 80 --min-label-count 1 --device cuda:0
```

Training summary:

| Run | Steps | First Loss | Last Loss | Min Loss | Epoch 10 Avg | Epoch 10 Target Var Avg | Epoch 10 Pred Var Avg | Last Target Var | Last Pred Var | Median Enc Grad | Mean Clips/s | Max Mem MB | Device Hours | Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks, 10 epochs | 2560 | 0.82369 | 0.53267 | 0.16541 | 0.38251 | 0.28780 | 0.08338 | 0.44866 | 0.00384 | 0.20322 | 9.50 | 150.04 | 0.15204 | 546.1 |
| VideoMamba-Tiny, attention predictor, row/tube masks, 10 epochs | 2558 | 0.81131 | 0.53668 | 0.38167 | 0.71806 | 0.69205 | 0.03197 | 0.37733 | 0.02651 | 0.06670 | 2.08 | 178.67 | 0.68946 | 2480.8 |

The VideoMamba CSV is missing two logged microsteps but the run completed
10 epochs and wrote the epoch-10 checkpoint.

Held-out probe summary:

| Run | Train Clips | Eval Clips | Classes | Final Train Acc | Best Train Acc | Eval Acc | Eval Loss | Feature Var | Total Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, 10 epochs, 32-class split | 96 | 32 | 32 | 0.4792 | 0.4896 | 0.0625 | 5.7011 | 0.33056 | 5.70 |
| VideoMamba-Tiny row/tube, 10 epochs, 32-class split | 96 | 32 | 32 | 0.7083 | 0.7396 | 0.0625 | 6.2431 | 0.37483 | 10.63 |
| ViT-Tiny, 10 epochs, 8-class split | 48 | 16 | 8 | 0.6250 | 0.6250 | 0.1250 | 2.8715 | 0.30758 | 2.50 |
| VideoMamba-Tiny row/tube, 10 epochs, 8-class split | 48 | 16 | 8 | 0.7083 | 0.7292 | 0.0000 | 3.5656 | 0.31827 | 5.81 |

Interpretation:

- Longer ViT-Tiny training clearly improves the V-JEPA loss curve on the
  512-clip subset. Its average loss falls from 0.69886 in epoch 1 to 0.38251 in
  epoch 10.
- Longer VideoMamba row/tube training remains numerically stable, but it
  plateaus: average loss is 0.67012 in epoch 1 and 0.71806 in epoch 10, with
  noisy intermediate epochs.
- VideoMamba prediction variance improves compared with the 3-epoch checkpoint,
  but it remains lower than the ViT epoch-10 average. Median encoder-gradient
  scale is also still lower.
- The longer checkpoints do not solve held-out probe performance. On the
  32-class split both models get 2 of 32 held-out clips correct. On the 8-class
  split, ViT is exactly at chance and VideoMamba gets 0 of 16.
- This is useful negative evidence: the immediate bottleneck is not just the
  number of epochs. The next experiments should improve the data/probe protocol
  or training objective details before spending much more compute on the same
  512-clip setting.

### Probe Sensitivity Check

The same 10-epoch checkpoints were also evaluated with a stronger linear-probe
setting: `--probe-lr 0.1` and `--probe-epochs 300`. This checks whether the
standard 80-epoch, `lr=0.01` probe was simply too weak.

Probe summary:

| Run | Train Clips | Eval Clips | Classes | Final Train Acc | Best Train Acc | Eval Acc | Eval Loss | Total Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, 10 epochs, 32-class split, strong probe | 96 | 32 | 32 | 0.6354 | 0.7188 | 0.0312 | 38.0932 | 9.50 |
| VideoMamba-Tiny row/tube, 10 epochs, 32-class split, strong probe | 96 | 32 | 32 | 0.8125 | 0.8854 | 0.0938 | 36.8779 | 13.87 |
| ViT-Tiny, 10 epochs, 8-class split, strong probe | 48 | 16 | 8 | 0.8958 | 0.9375 | 0.1250 | 9.3160 | 6.65 |
| VideoMamba-Tiny row/tube, 10 epochs, 8-class split, strong probe | 48 | 16 | 8 | 1.0000 | 1.0000 | 0.0625 | 10.9503 | 10.43 |

Interpretation:

- The stronger probe fits the tiny train splits much better, especially the
  VideoMamba 8-class split, which reaches 100% train accuracy.
- Held-out accuracy still does not become meaningful. The stronger probe mostly
  amplifies train-set memorization and produces very large held-out losses.
- This argues against treating the earlier chance-level probe as just a weak
  optimizer artifact. The evidence source still needs a better split, more
  data, stronger representation training, or all three.

## 2048-Clip Subset And Balanced Probe Follow-Up

The next local step extracted 2048 train-labelled SSv2 clips from archive part
`00` into a separate subset folder. This was intended to answer whether the
512-clip probe failures were mostly caused by too little class support.

Commands:

```bash
python -m diagnostics.extract_ssv2_hf_subset --num-clips 2048 --output-dir diagnostic_runs/data/ssv2_subset_2048/videos --manifest diagnostic_runs/data/ssv2_subset_2048/ssv2_train_2048.csv
python -m diagnostics.make_balanced_probe_split --input diagnostic_runs/data/ssv2_subset_2048/ssv2_train_2048.csv --train-output diagnostic_runs/data/ssv2_subset_2048/ssv2_probe_train_16c_15x5.csv --eval-output diagnostic_runs/data/ssv2_subset_2048/ssv2_probe_eval_16c_15x5.csv --train-per-class 15 --eval-per-class 5 --max-classes 16 --seed 234
python -m diagnostics.make_balanced_probe_split --input diagnostic_runs/data/ssv2_subset_2048/ssv2_train_2048.csv --train-output diagnostic_runs/data/ssv2_subset_2048/ssv2_probe_train_32c_10x5.csv --eval-output diagnostic_runs/data/ssv2_subset_2048/ssv2_probe_eval_32c_10x5.csv --train-per-class 10 --eval-per-class 5 --max-classes 32 --seed 234
python -m app.main --fname configs/diagnostics/local_ssv2_2048_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_2048_videomamba_attention_rowtube.yaml --devices cuda:0
```

Subset label distribution:

- 2048 clips.
- 174 classes.
- 55 classes with at least 15 clips.
- 27 classes with at least 20 clips.
- The 16-class split has 240 train clips and 80 held-out clips.
- The 32-class split has 320 train clips and 160 held-out clips.

Engineering note:

- A first ViT run was intentionally launched with a short foreground timeout to
  expose errors and was interrupted.
- A later ViT run hit a transient Windows/OneDrive `PermissionError` while
  appending to the CSV log during epoch 3.
- `src/utils/logger.py` now retries CSV append opens to handle these transient
  local file locks.
- The completed ViT summary below uses the final full CSV segment after the
  last CSV header. The earlier interrupted partial segment is ignored.

Training summary:

| Run | Steps | First Loss | Last Loss | Min Loss | Epoch 1 Avg | Epoch 2 Avg | Epoch 3 Avg | Last Target Var | Last Pred Var | Median Enc Grad | Mean Clips/s | Max Mem MB | Device Hours | Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks, 2048 clips | 3072 | 0.82577 | 0.49298 | 0.19180 | 0.60522 | 0.53289 | 0.53257 | 0.54735 | 0.21356 | 0.17511 | 6.80 | 150.04 | 0.25927 | 931.8 |
| VideoMamba-Tiny, attention predictor, row/tube masks, 2048 clips | 3065 | 0.83706 | 0.75317 | 0.35599 | 0.67723 | 0.72255 | 0.72566 | 0.83054 | 0.01102 | 0.07481 | 1.62 | 178.67 | 1.06112 | 3818.5 |

The VideoMamba CSV is missing a small number of microstep rows, but the run
completed all three epochs and wrote the epoch-3 checkpoint.

Before running the 2048-trained checkpoints, the older 512-clip 10-epoch
checkpoints were evaluated on the improved 2048-derived probe splits. This did
not rescue held-out accuracy:

| Checkpoint | Split | Train Clips | Eval Clips | Classes | Best Train Acc | Eval Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ViT 512-clip 10-epoch | 16c 15x5 | 240 | 80 | 16 | 0.2833 | 0.1000 |
| VideoMamba 512-clip 10-epoch | 16c 15x5 | 240 | 80 | 16 | 0.4208 | 0.0875 |
| ViT 512-clip 10-epoch | 32c 10x5 | 320 | 160 | 32 | 0.2219 | 0.0188 |
| VideoMamba 512-clip 10-epoch | 32c 10x5 | 320 | 160 | 32 | 0.3688 | 0.0437 |

The new 2048-trained checkpoints were then probed on the same balanced splits:

| Run | Split | Train Clips | Eval Clips | Classes | Final Train Acc | Best Train Acc | Eval Acc | Eval Loss | Feature Var |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny 2048 | 16c 15x5 | 240 | 80 | 16 | 0.2625 | 0.3625 | 0.0750 | 3.5532 | 0.34619 |
| VideoMamba-Tiny row/tube 2048 | 16c 15x5 | 240 | 80 | 16 | 0.3500 | 0.4292 | 0.0875 | 3.4141 | 0.36883 |
| ViT-Tiny 2048 | 32c 10x5 | 320 | 160 | 32 | 0.2906 | 0.2906 | 0.0125 | 4.9195 | 0.34040 |
| VideoMamba-Tiny row/tube 2048 | 32c 10x5 | 320 | 160 | 32 | 0.3563 | 0.3781 | 0.0375 | 4.7442 | 0.36813 |

Interpretation:

- The 2048 subset improves the probe substrate, but it does not produce useful
  held-out frozen-probe accuracy.
- ViT still trains more cleanly: its loss decreases to an epoch-3 average of
  0.53257 and final prediction variance is much higher than VideoMamba's.
- VideoMamba is stable, but the loss plateaus after epoch 1 and prediction
  variance remains very low. This reinforces the earlier concern that the
  masked-latent objective is not driving VideoMamba targets/predictions in the
  same way it drives ViT.
- VideoMamba fits the small probe train splits slightly better than ViT in some
  cases, but held-out accuracy remains near chance.
- This is a local transition point. Locally, the project now has evidence for a
  bounded feasibility/diagnostic paper. A positive performance result likely
  needs either a targeted objective modification or selective LUMI-scale runs.
