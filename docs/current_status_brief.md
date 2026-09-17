# Current Status Brief

This note is a short, shareable explanation of the current project state and
the immediate research logic.

For a continuation-oriented handoff, see `docs/research_handoff.md`. For Rocket,
LUMI, access steps, and conference compute estimates, see
`docs/hpc_lumi_usage.md`.

Yes. In short, we now have a working research scaffold, not just an idea.

What we did:

- Reframed the project as **VideoMamba trained with a V-JEPA-style masked latent objective**, under bounded compute.
- Checked publication positioning: I did not find a formal venue paper for the exact VideoMamba + V-JEPA combination, so the novelty claim is plausible if stated cautiously.
- Built a diagnostic experiment path around SSv2 subsets.
- Added configs, frozen-probe tools, balanced split tools, logging, and documentation.
- Ran local experiments on the RTX 3070 Laptop GPU:
  - synthetic smoke runs;
  - 32/128 clip sanity runs;
  - 512-clip 3-epoch runs;
  - 512-clip 10-epoch follow-up;
  - frozen probes and stronger probe-sensitivity checks.

What we learned:

- The code runs locally end to end.
- VideoMamba + V-JEPA is feasible as an implementation.
- Local compute is enough for meaningful pilot diagnostics, not enough for full-scale pretraining.
- ViT baseline behaves more cleanly: loss improves steadily.
- VideoMamba runs stably, but is slower and its loss/variance behavior is weaker.
- Frozen probes currently do **not** show useful held-out representation quality. (After pretraining, a simple linear classifier on frozen encoder features does not classify held-out SSv2 videos better than chance in a useful way.)
- Stronger probes can memorize tiny train splits, but still fail held-out evaluation, so the issue is not just probe optimizer weakness. Probe optimizer is the optimizer that trains the probe head.  

My current interpretation:

This is still publishable as a modest workshop/short-paper direction, but not yet as "VideoMamba-JEPA works well." The honest current story is:

> We implemented VideoMamba with a V-JEPA-style objective, showed it runs under bounded compute, and identified concrete stability/generalization issues compared with a ViT V-JEPA baseline.

Potential next steps:

1. **Improve the evaluation substrate**
   - Build a larger, more label-balanced SSv2 subset. (Compared to diagnostic_runs/data/ssv2_subset)
   - Avoid 1-2 held-out clips per class; that is too noisy.
   - Maybe use fewer classes with many clips per class first.

2. **Validate the probe protocol**
   - Add random-feature and supervised sanity baselines.
   - Try kNN or logistic regression alternatives.
   - Confirm the labels/splits are not too sparse or semantically awkward.

3. **Test focused VideoMamba-JEPA fixes**
   - Target projection instead of raw/layer-norm targets only.
   - Different predictor capacity.
   - Different mask ratios.
   - Row/tube masks vs multi-block masks.
   - EMA and learning-rate tweaks.

4. **Then scale selectively**
   - Do not run a huge full matrix yet.
   - Pick 2-3 best candidates and run them on a larger subset.
   - Use CUDA if possible; LUMI/ROCm remains risky for Mamba kernels.

So the next best move is probably not "more epochs on the same 512 clips." It is: **construct a better subset/probe setup, then rerun ViT vs VideoMamba under that cleaner evaluation.**

## Local-To-LUMI Decision Rule

Continue locally while the experiments are answering code, data, probe, and
small-configuration questions. Move to LUMI or another larger GPU environment
when:

- the local runs are stable and reproducible;
- the probe/data protocol has been validated enough that more compute would
  answer a research question rather than debug the pipeline;
- the remaining bottleneck is subset size, sequence length, number of epochs,
  or number of seeds;
- the planned run has a fixed config, fixed manifest construction rule, and a
  clear stop condition.

If local experiments become strong enough to justify a humble workshop paper,
the project can stop before LUMI and write around bounded implementation plus
diagnostic evidence. If local evidence remains too weak but the failure mode is
well understood, transition to LUMI with only the best 2-3 configurations rather
than a broad matrix.

## Update After 2048-Clip Local Runs

The next local step extracted a 2048-clip SSv2 subset and created more balanced
probe splits: 16 classes with 15 train and 5 held-out clips per class, and 32
classes with 10 train and 5 held-out clips per class.

This improved the evaluation substrate but did not produce useful held-out
linear-probe accuracy. ViT trained more cleanly than VideoMamba, while
VideoMamba remained stable but showed weak prediction variance and poor
held-out probe transfer. This is now enough local evidence for a bounded
feasibility/diagnostic story. A positive "VideoMamba-JEPA works well" result
requires either targeted objective changes or selective scaling on LUMI.

## Current Operational Recommendation

Continue locally only for cheap diagnostics and probe/config checks. Start
Rocket/LUMI access work now. Prefer Rocket/CUDA for the first cluster pilot if
available, because it is closer to the local NVIDIA environment. Use LUMI only
after the smoke benchmark in `hpc/lumi_smoke.sbatch` confirms that the ROCm
environment can run the VideoMamba path.
