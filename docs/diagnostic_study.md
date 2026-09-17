# Compute-Bounded VideoMamba + V-JEPA Implementation and Evaluation Study

This repo now supports a phase-1 implementation and evaluation study instead of
a full 300-epoch pretraining attempt. The headline contribution is VideoMamba
trained with a V-JEPA-style masked latent objective under realistic compute
constraints. The diagnostic harness is the support structure: it makes the
combination measurable, reproducible, and cheap enough to evaluate before
cluster-scale pretraining.

Main question:

> Can a pure VideoMamba video backbone be pretrained with a V-JEPA-style masked
> latent prediction objective under realistic compute constraints?

Secondary questions:

- Does the combination run stably on small real-video subsets?
- Does it avoid collapse under standard V-JEPA masks?
- Is a Mamba predictor or lightweight attention predictor more stable?
- Are row/tube masks better aligned with VideoMamba than V-JEPA multi-block
  masks?
- Does target normalization materially affect stability?

## Workflow

1. Install the minimal local diagnostic dependencies.

```bash
python -m pip install -r requirements-diagnostics.txt
```

The inherited `requirements.txt` is broader and may be appropriate for a full
cluster environment, but the diagnostics file is enough for the local smoke and
generated-video fixture path.

2. Run smoke tests before using paid compute.

```bash
python -m diagnostics.smoke --model vit_tiny --mask-type multiblock3d
python -m diagnostics.smoke --model videomamba_tiny --predictor mamba --mask-type multiblock3d
python -m diagnostics.smoke --model videomamba_tiny --predictor attention --mask-type row_tube
```

If a smoke result reports `ModuleNotFoundError`, install the missing dependency
or the full environment from `requirements.txt` before interpreting any model
behavior. The smoke test writes structured JSON under `diagnostic_runs/` even
when setup fails.

If the VideoMamba smoke test fails on LUMI/ROCm, do not run the matrix there.
Use the failure to decide whether to port kernels, use a CUDA environment, or
drop LUMI for the pilot.

3. Optional local generated-video fixture.

This is not a research experiment. It only verifies the real video reader,
transforms, masks, trainer loop, checkpointing, and CSV metrics before an SSv2
manifest is available.

```bash
python -m diagnostics.make_synthetic_videos --num-videos 8 --frames 24 --size 96 --manifest diagnostic_runs/data/generated_videos.csv --output-dir diagnostic_runs/data/generated_videos
python -m app.main --fname configs/diagnostics/local_generated_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_generated_videomamba.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_generated_videomamba_attention_rowtube.yaml --devices cuda:0
python -m diagnostics.summarize_selected --csv diagnostic_runs/local_generated/vit_multiblock/local_generated_vit_multiblock_r0.csv diagnostic_runs/local_generated/videomamba_mamba/local_generated_videomamba_mamba_r0.csv diagnostic_runs/local_generated/videomamba_attention_rowtube/local_generated_videomamba_attention_rowtube_r0.csv --output diagnostic_runs/local_generated/summary_current.json
```

The manifest parser in the original repo splits on spaces, so prefer relative
paths or absolute paths without spaces.

The current fixture results are committed in
`docs/local_generated_results.md`.

4. Create a small SSv2 subset CSV.

The official SSv2 dataset is research-use data. For local sanity checks, the
repo includes helpers to fetch the annotations, research license, download
instructions, and video archive part `00` from the Hugging Face mirror
`morpheushoc/something-something-v2`, then extract a small train-labelled subset
from that archive part:

```bash
python -m diagnostics.fetch_ssv2_hf_part --parts 00
python -m diagnostics.extract_ssv2_hf_subset --num-clips 128
python -m diagnostics.make_subset --input diagnostic_runs/data/ssv2_subset/ssv2_train_128.csv --output diagnostic_runs/data/ssv2_subset/ssv2_train_32.csv --num-clips 32 --seed 234 --require-existing
```

Quick local SSv2 sanity configs:

```bash
python -m app.main --fname configs/diagnostics/local_ssv2_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_videomamba.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_videomamba_attention_rowtube.yaml --devices cuda:0
```

Longer 32-clip overfit/sanity configs:

```bash
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba_raw.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba_mamba_rowtube.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba_attention_multiblock.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_overfit_videomamba_attention_rowtube.yaml --devices cuda:0
```

Current 128-clip subset configs:

```bash
python -m app.main --fname configs/diagnostics/local_ssv2_128_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_128_videomamba_attention_multiblock.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_128_videomamba_attention_rowtube.yaml --devices cuda:0
python -m diagnostics.summarize_selected --csv diagnostic_runs/local_ssv2_128/vit_multiblock/local_ssv2_128_vit_multiblock_r0.csv diagnostic_runs/local_ssv2_128/videomamba_attention_multiblock/local_ssv2_128_videomamba_attention_multiblock_r0.csv diagnostic_runs/local_ssv2_128/videomamba_attention_rowtube/local_ssv2_128_videomamba_attention_rowtube_r0.csv --output diagnostic_runs/local_ssv2_128/summary_current.json
```

Tiny frozen linear-probe overfit diagnostic:

```bash
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_128_vit.yaml --checkpoint diagnostic_runs/local_ssv2_128/vit_multiblock/local_ssv2_128_vit_multiblock-latest.pth.tar --output diagnostic_runs/local_ssv2_128/probes/vit_multiblock_probe.json --batch-size 8 --probe-epochs 40 --min-label-count 2 --device cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_128_videomamba_attention_multiblock.yaml --checkpoint diagnostic_runs/local_ssv2_128/videomamba_attention_multiblock/local_ssv2_128_videomamba_attention_multiblock-latest.pth.tar --output diagnostic_runs/local_ssv2_128/probes/videomamba_attention_multiblock_probe.json --batch-size 8 --probe-epochs 40 --min-label-count 2 --device cuda:0
python -m diagnostics.frozen_probe --config configs/diagnostics/local_ssv2_128_videomamba_attention_rowtube.yaml --checkpoint diagnostic_runs/local_ssv2_128/videomamba_attention_rowtube/local_ssv2_128_videomamba_attention_rowtube-latest.pth.tar --output diagnostic_runs/local_ssv2_128/probes/videomamba_attention_rowtube_probe.json --batch-size 8 --probe-epochs 40 --min-label-count 2 --device cuda:0
```

This probe is an overfit/plumbing diagnostic on repeated-label clips from the
tiny subset. It should not be reported as held-out probe accuracy.

Next 512-clip held-out-probe path:

```bash
python -m diagnostics.extract_ssv2_hf_subset --num-clips 512 --manifest diagnostic_runs/data/ssv2_subset/ssv2_train_512.csv
python -m diagnostics.make_balanced_probe_split --input diagnostic_runs/data/ssv2_subset/ssv2_train_512.csv --train-output diagnostic_runs/data/ssv2_subset/ssv2_probe_train_balanced.csv --eval-output diagnostic_runs/data/ssv2_subset/ssv2_probe_eval_balanced.csv --train-per-class 3 --eval-per-class 1 --max-classes 32 --seed 234
python -m app.main --fname configs/diagnostics/local_ssv2_512_vit.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_512_videomamba_attention_multiblock.yaml --devices cuda:0
python -m app.main --fname configs/diagnostics/local_ssv2_512_videomamba_attention_rowtube.yaml --devices cuda:0
```

For held-out probes, pass the balanced probe train manifest through
`--manifest` and the held-out split through `--eval-manifest`.

Current local SSv2 subset results are committed in
`docs/local_ssv2_results.md`.

Chronological local run timings and local compute accounting are committed in
`docs/experiment_timeline.md`.

Use the same two-column format as the main README:

```text
/absolute/path/to/video_0001.webm 0
/absolute/path/to/video_0002.webm 0
```

Start with 32-128 clips for overfit/sanity runs. If you already have a full
manifest, create a fixed-seed subset with:

```bash
python -m diagnostics.make_subset --input /absolute/path/to/SSv2_train.csv --output diagnostic_runs/data/ssv2_128.csv --num-clips 128 --require-existing
```

Then update `configs/diagnostics/base_short_pretrain.yaml` to point to that
subset.

5. Generate the phase-1 matrix and cost estimate.

```bash
python -m diagnostics.run_matrix --write-configs
```

This writes generated configs under `diagnostic_runs/configs/phase1/` and a
plan JSON at `diagnostic_runs/phase1_plan.json`. It does not launch jobs.

6. Replace the default budget estimate with measured smoke timing.

```bash
python -m diagnostics.run_matrix --write-configs --measured-ms 2500
```

Use the measured wall-time per microstep from smoke or a short benchmark. The
budget gate fails if the training matrix exceeds the post-reserve cap.

7. Run only after the budget gate passes.

```bash
python -m diagnostics.run_matrix --run --measured-ms 2500
```

For Slurm, change `run.launch` in `configs/diagnostics/phase1_matrix.yaml` to
`slurm` and set the partition/folder fields.

8. Summarize results.

```bash
python -m diagnostics.summarize --root diagnostic_runs/phase1
```

The enriched training CSVs include loss, JEPA loss, target/prediction variance,
activation magnitude, gradient norms, memory, clips/sec, and per-step device
hours.

## Budget Rule

The matrix assumes:

- EUR 1000 total cap.
- EUR 0.35 per LUMI GPU-hour.
- 10% reserved for setup/smoke benchmarks.
- 20% reserved for reruns and final verification.

Only 70% of the cap is available for the first training matrix. For LUMI-G
standard full-node jobs, the estimator charges 4 GPU-hours per node-hour.

## Phase-1 Comparisons

The default matrix compares:

- ViT-Tiny V-JEPA baseline.
- VideoMamba-Tiny with a Mamba predictor.
- VideoMamba-Tiny with a lightweight attention predictor.
- Raw target tokens vs layer-normalized target tokens.
- V-JEPA multi-block masks vs row-tube masks.

No Kinetics-400 or ImageNet experiments are part of phase 1.

## Stop Conditions

Stop before spending more compute when any of these happen:

- The smoke benchmark fails on the target hardware.
- The measured microstep time makes the matrix exceed the budget gate.
- Loss becomes NaN or prediction variance collapses to near zero in repeated
  sanity runs.
- VideoMamba cannot overfit 32-128 SSv2 clips while the ViT baseline can under
  comparable settings.

## Publication Decision

Frame the paper as a compute-bounded implementation and empirical evaluation of
VideoMamba trained with a V-JEPA-style masked latent objective. Keep diagnostics
as supporting evidence, not the main identity of the paper.

Proceed toward a workshop/short paper if phase 1 gives one of:

- VideoMamba + V-JEPA trains stably enough to report bounded results.
- A small modification, such as attention predictor, target normalization, or
  row/tube masks, clearly improves stability or frozen-probe readiness.
- The combination fails in a reproducible and informative way while the ViT
  baseline succeeds.

Minimum acceptable experiment set:

- ViT-Tiny V-JEPA baseline.
- VideoMamba-Tiny with a V-JEPA-style objective and Mamba predictor.
- VideoMamba-Tiny with a V-JEPA-style objective and attention predictor.
- Multi-block masks vs row/tube masks.
- Raw target tokens vs layer-normalized target tokens.
- 32-128 clip sanity/overfit runs, followed by a short SSv2 subset run if
  stable.
- Frozen-probe result when feasible.

Each reported configuration must include:

- fixed seed
- exact config path
- GPU or device used
- wall time and estimated GPU-hours
- final loss curve
- target variance and prediction variance
- gradient norms
- frozen-probe result if available

Do not claim a state-of-the-art VideoMamba-JEPA result unless later experiments
actually support it. Also avoid claiming "first ever" unless a final literature
check before submission still supports that wording.
