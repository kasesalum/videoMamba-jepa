# Research Handoff

This document is a compact continuation guide for the VideoMamba + V-JEPA
study. It summarizes what exists, what has been learned, and what should happen
next.

## Project Identity

The project is a compute-bounded implementation and empirical evaluation of a
pure VideoMamba video backbone trained with a V-JEPA-style masked latent
prediction objective. The paper should not claim state-of-the-art pretraining.
The defensible target is a modest workshop or short-paper contribution.

The main question is:

> Can a pure VideoMamba video backbone be pretrained with a V-JEPA-style masked
> latent prediction objective under realistic compute constraints?

Diagnostics are the support structure for that question. They are not a
separate project identity.

## What Was Implemented

- End-to-end local training paths for ViT-Tiny V-JEPA and VideoMamba-Tiny with a
  V-JEPA-style masked latent objective.
- Smoke tests for model import, masked prediction, optimizer step, checkpoint
  round trip, and frozen feature extraction.
- Generated-video fixtures to test the real video loader without SSv2.
- SSv2 subset fetch/extract helpers.
- Balanced probe split helper.
- Frozen linear-probe diagnostic.
- CSV logging for loss, target/prediction variance, activation magnitude,
  gradient norms, memory, throughput, wall time, GPU time, and device-hours.
- Configs for ViT baseline, VideoMamba with Mamba predictor, VideoMamba with
  attention predictor, multiblock masks, row/tube masks, raw targets, and
  normalized targets.
- Retry handling around CSV logging to reduce Windows/OneDrive append-lock
  failures.
- Documentation for rationale, local results, timeline, LUMI transition, and
  HPC usage.

## What Was Already Run

Local experiments used an NVIDIA GeForce RTX 3070 Laptop GPU. The local path
covered:

- synthetic smoke tests;
- generated-video sanity runs;
- SSv2 32-clip sanity/overfit runs;
- SSv2 128-clip runs;
- SSv2 512-clip runs;
- SSv2 512-clip 10-epoch follow-up;
- frozen-probe and stronger probe-sensitivity checks;
- SSv2 2048-clip runs with more balanced held-out probe splits.

The detailed timing report is in `docs/experiment_timeline.md`. The local SSv2
interpretation is in `docs/local_ssv2_results.md`.

## What The Results Mean

The implementation works. VideoMamba + V-JEPA can run end to end through the
loader, masks, model, predictor, target encoder, optimizer, logging, checkpoint
path, and frozen-probe extraction.

The current evidence is not a positive performance result. ViT trains more
cleanly. VideoMamba is stable enough to continue testing, but it is slower in
the local fallback environment, has weak prediction-variance behavior, and has
not produced useful held-out frozen-probe accuracy on the local subset regime.

The honest current claim is:

> We implemented VideoMamba with a V-JEPA-style objective, showed it runs under
> bounded compute, and identified stability/generalization issues compared with
> a ViT V-JEPA baseline.

This is still potentially publishable for a modest venue if the evidence is
presented as bounded implementation and empirical evaluation. It is not yet a
"VideoMamba-JEPA works well" result.

## Article Strategy

Track A: positive bounded implementation paper.

- Show that VideoMamba + V-JEPA trains stably on a larger real-video subset.
- Report matched ViT baseline, throughput, memory, GPU-hours, loss curves,
  target/prediction variance, gradient norms, and frozen-probe accuracy.
- If a modification helps, frame it as a practical adjustment needed for the
  combination rather than as a SOTA recipe.

Track B: exploratory/diagnostic negative paper.

- Show that the exact combination runs but underperforms or fails to transfer
  under matched compute while ViT behaves more cleanly.
- Make the contribution reproducibility-oriented: compute budget, exact
  manifests, exact configs, variance/collapse diagnostics, and probe results.
- This is appropriate only if the failure is controlled and informative, not if
  the experiment is merely underpowered or the probe protocol is broken.

Do not claim "first ever" without a final literature check. The safer phrasing
is "we did not find a formal venue publication for this exact VideoMamba +
V-JEPA-style masked latent prediction combination as of the final check."

## Assumptions

- Target venue is a workshop, short paper, or modest conference target.
- No Kinetics-400, ImageNet, or full 300-epoch pretraining is required for the
  first paper.
- SSv2 remains the phase-1 real-video dataset.
- Dataset archives, extracted videos, checkpoints, and raw run logs are not
  committed to git.
- Every reported result must include seed, config path, data manifest rule,
  GPU/device, wall time, estimated GPU-hours, loss curve, target/prediction
  variance, gradient norms, and frozen-probe result when feasible.

## Next Experiment Plan

1. Validate the cluster environment.
   - Run the Rocket or LUMI smoke script from `hpc/`.
   - Stop if VideoMamba fails on ROCm.
   - Record exact software versions and measured smoke timing.

2. Create a better SSv2 stage-1 subset.
   - Use 8192 clips if storage allows.
   - Build balanced probe splits with enough train and held-out clips per class.
   - Keep the manifest and split-generation command fixed.

3. Run the minimum matrix.
   - ViT-Tiny V-JEPA baseline.
   - VideoMamba-Tiny + attention predictor + row/tube masks + layer-normalized
     targets.
   - VideoMamba-Tiny + attention predictor + row/tube masks + L2-normalized
     targets.
   - Three seeds if compute allows.

4. Run probes.
   - Use the frozen-probe script with fixed train/eval manifests.
   - Treat train-only memorization as a plumbing check, not paper evidence.

5. Decide the paper direction.
   - If VideoMamba is stable and probe results improve, write Track A.
   - If ViT succeeds and VideoMamba reproducibly fails or collapses, write
     Track B.
   - If both fail or the probe remains noisy, improve the data/probe protocol
     before spending more compute.

6. Scale selectively.
   - Increase only one axis at a time: subset size, resolution, frame count, or
     epochs.
   - Keep the best 1-2 VideoMamba configs plus ViT.
   - Preserve at least 20% of the compute budget for reruns and final checks.

## Where To Look

- `README.md`: top-level map.
- `docs/research_rationale_memo.md`: research reasoning and publication
  positioning.
- `docs/current_status_brief.md`: short status summary.
- `docs/experiment_timeline.md`: chronological local compute report.
- `docs/local_ssv2_results.md`: local SSv2 result interpretation.
- `docs/hpc_lumi_usage.md`: access process and compute-cost assessment.
- `docs/lumi_transition_plan.md`: decision rule for moving beyond local runs.
- `hpc/README.md`: concrete Rocket/LUMI commands and Slurm templates.
