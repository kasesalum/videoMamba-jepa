# Local Generated-Video Sanity Results

These results are from a generated-video fixture. They are not evidence about
SSv2 accuracy or representation quality. Their purpose is to verify that the
local repository can run the real video reader, transforms, mask collators,
model forward passes, optimizer step, checkpointing, and diagnostic logging.

For the revised paper framing, these results are implementation evidence for
the VideoMamba + V-JEPA-style training path. They show that the combined model,
masking, predictor, optimizer, checkpointing, and logging path runs end to end
locally. They do not validate performance or representation quality; the next
scientific evidence must come from small real-video SSv2 subset runs.

The fixture used 8 generated MP4 videos, 24 frames per video, 96x96 source
resolution, batch size 2, 4 sampled frames, 64x64 crop, 2 iterations, and 1
epoch on `cuda:0`.

## Current Fixture Summary

| Run | Loss | Target Var | Pred Var | Enc Grad | Pred Grad | Mem MB | Clips/s | Last Wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Tiny, multi-block masks | 0.81138 | 0.64777 | 0.00365 | 0.00436 | 0.38372 | 150.03 | 12.41 | 161 |
| VideoMamba-Tiny, Mamba predictor, multi-block masks | 0.81694 | 0.56486 | 0.00337 | 0.00000 | 0.00000 | 188.30 | 2.36 | 848 |
| VideoMamba-Tiny, attention predictor, row/tube masks | 0.80607 | 0.65284 | 0.00517 | 0.00755 | 0.38367 | 178.67 | 2.21 | 904 |

## Interpretation

All three fixture runs completed. That confirms the local path is operational:
video decoding, augmentation, masks, model initialization, forward/backward,
checkpointing, and enriched CSV logging work.

The generated-video fixture also confirms that the diagnostic metrics are
informative enough for the next phase. Target variance is nonzero in all cases,
while prediction variance is much lower at initialization. This is expected for
short untrained runs and is exactly the kind of collapse/stability signal the
study should track on real video subsets.

The speed gap is a warning, not a conclusion. The local VideoMamba runs use
fallback paths because optimized fused Mamba kernels are not available in this
environment. Real timing claims require a target CUDA or ROCm environment with
the intended kernel stack.

## Commands

```bash
python -m diagnostics.make_synthetic_videos --num-videos 8 --frames 24 --size 96 --manifest diagnostic_runs/data/generated_videos.csv --output-dir diagnostic_runs/data/generated_videos
python -m app.main --fname configs/diagnostics/local_generated_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_generated_videomamba.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_generated_videomamba_attention_rowtube.yaml --devices cuda:0
```
