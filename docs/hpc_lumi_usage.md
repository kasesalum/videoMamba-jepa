# HPC, Rocket, and LUMI Usage Notes

This document records the practical access path, cost assumptions, and compute
plan for moving the VideoMamba + V-JEPA study from local diagnostics to a
cluster run. It is written as an operational handoff, not as an official
accounting document. Confirm prices, allocation rules, and billing before
spending real compute.

Last checked: 2026-06-20.

## Short Recommendation

Continue local work only for cheap diagnostics: probe sanity checks, config
validation, and small objective changes. Start UT HPC / ETAIS access work now.
If Rocket GPU access is available, use Rocket first because it is CUDA/NVIDIA
and therefore closer to the local environment. Use LUMI only after a ROCm smoke
benchmark confirms that PyTorch, Mamba dependencies, model imports, one masked
batch, one optimizer step, checkpoint save/load, and frozen-probe extraction all
work.

## Useful Official Links

- [UT HPC Center](https://hpc.ut.ee/)
- [UT HPC public documentation](https://docs.hpc.ut.ee/public/)
- [ETAIS self-service overview](https://docs.hpc.ut.ee/public/ETAIS/overview/)
- [ETAIS / UT / LUMI prices](https://etais.ee/en/prices/)
- [Rocket GPU computing docs](https://docs.hpc.ut.ee/public/cluster/Running_jobs/gpu_computing/)
- [Rocket job submission docs](https://docs.hpc.ut.ee/public/cluster/Running_jobs/submit_jobs/)
- [LUMI users in Estonia](https://lumi-supercomputer.eu/get-started-2021/users-in-estonia/)
- [LUMI billing policy](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/)
- [LUMI-G hardware](https://docs.lumi-supercomputer.eu/hardware/lumig/)
- [LUMI Slurm partitions](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/)

## Access Path

1. Identify the resource path.
   - For UT Rocket, request cluster access through UT HPC.
   - For LUMI, Estonian academic users should start through ETAIS / UT HPC. The
     LUMI Estonia page says academic access requires a project PI affiliated
     with an Estonian academic research institution and points to ETAIS at
     `etais@etais.ee`.

2. Open or verify ETAIS self-service access.
   - Use [minu.etais.ee](https://minu.etais.ee/).
   - Authentication is handled through MyAccessID.
   - Work is organized by organization, project, resource, and allocation.
   - The allocation name is needed for Slurm jobs when using ETAIS-managed
     resources.

3. Request the service.
   - UT HPC documentation says cluster computing requests for Rocket or LUMI
     should go through the HPC request route.
   - Use `support@hpc.ut.ee` for UT HPC support questions.
   - Use `etais@etais.ee` for price/allocation questions, explicitly saying
     whether the request is for University of Tartu, Rocket, or LUMI.

4. Confirm allocation and billing.
   - In `minu.etais.ee`, find the project and the allocation name shown for the
     relevant resource.
   - Rocket ETAIS users normally need a Slurm `--account` such as
     `ealloc_...`.
   - LUMI projects normally use `--account=project_...`.
   - Before long jobs, ask support which project/account will be charged and
     whether any annual internal allowance applies.

5. Set SSH access.
   - Add the public SSH key in the ETAIS/MyAccessID flow as documented by UT
     HPC.
   - For Rocket, log in to the Rocket frontend after the account is active.
   - For LUMI, follow the LUMI SSH and project-account instructions provided
     with the allocation.

6. Create project storage.
   - Keep repository code, small configs, scripts, and result summaries in git.
   - Keep SSv2 archives, extracted videos, checkpoints, TensorBoard logs, and
     large CSVs outside git, preferably in project storage.
   - Record the manifest construction command in the paper notes.

7. Run only a smoke benchmark first.
   - The smoke job must finish before any training matrix is submitted.
   - Record wall time, GPU type, PyTorch version, ROCm/CUDA version, and whether
     optimized Mamba kernels were used or a fallback path was used.

## Billing Notes

ETAIS pricing currently lists these relevant rates:

- UT HPC GPU compute: `0.5 EUR/GPU/h`.
- LUMI GPU compute: `0.35 EUR/GPU/h`.
- LUMI CPU compute: `0.008 EUR/core h`.
- LUMI project persistent/scratch storage: `0.0106 EUR/TB/h`.

The LUMI billing page defines GPU compute in GPU-hours and says one GPU-hour
corresponds to a full MI250X module for one hour. LUMI-G nodes have four AMD
MI250X GPU modules. Each MI250X has two GCDs, and Slurm/HIP see those GCDs as
separate GPUs. Consequences:

- `standard-g` allocates full LUMI-G nodes and bills 4 LUMI GPU-hours per
  node-hour.
- `small-g` and `dev-g` can allocate individual GCDs and bill 0.5 LUMI
  GPU-hours per allocated GCD-hour before CPU/memory thresholds.
- For single-GPU pilot jobs, prefer `dev-g` for short smoke tests and `small-g`
  for real single-GCD training.
- Avoid `standard-g` unless the job uses the full node effectively.

These rates and accounting details can change. Reconfirm them before final
budget approval.

## EUR 1000 Budget Conversion

Using only the published GPU-hour rates:

| Resource | Rate | Gross GPU-hours from EUR 1000 | Training GPU-hours after 10% setup and 20% reserve |
| --- | ---: | ---: | ---: |
| Rocket GPU | 0.50 EUR/GPU-hour | 2000 | 1400 |
| LUMI GPU | 0.35 EUR/GPU-hour | 2857 | 2000 |

The reserve is intentional. Failed environment builds, queue interruptions,
bad hyperparameters, and final verification runs are normal in this project.
Do not spend the full EUR 1000 on the first matrix.

## Compute Needed For A Modest Conference Result

The current local evidence is enough for an implementation/diagnostic story,
but not enough for a strong positive performance claim. A modest workshop or
short-paper result should aim for one of:

- stable VideoMamba + V-JEPA training on a larger real-video subset;
- a small modification that clearly improves stability or frozen-probe
  readiness;
- a controlled negative result where ViT succeeds and VideoMamba fails in a
  reproducible, informative way.

Current local 2048-clip / 3-epoch small-geometry reference:

| Configuration | Local device-hours | Use in estimate |
| --- | ---: | --- |
| ViT-Tiny, multiblock, layer-norm target | about 0.26 | ViT baseline |
| VideoMamba-Tiny, attention predictor, row/tube, layer-norm target | about 1.06 | main VideoMamba path |

Scaling that shape to 8192 clips and 20 epochs multiplies runtime by roughly
`4 * (20 / 3) = 26.7`. For three configurations and three seeds:

- ViT baseline: `0.26 * 26.7 * 3 = about 21 GPU-hours`.
- Two VideoMamba configurations: `1.06 * 26.7 * 2 * 3 = about 170 GPU-hours`.
- Total: about 190 single-GPU-equivalent hours.

At the simple published rates, that is about EUR 67 on LUMI or EUR 95 on
Rocket. On LUMI `small-g`, one GCD-hour may account as 0.5 LUMI GPU-hour, so
the actual LUMI accounting could be lower if CPU/memory requests stay under the
thresholds. Use this as a planning estimate only.

A more meaningful conference-grade run should be budgeted much higher because
the local reference used only 4 frames at 64px. Increasing frames, resolution,
subset size, seeds, or adding final verification can easily dominate the small
subset estimate. A practical LUMI training budget is:

- 50-100 accounted GPU-hours for environment setup, smoke tests, and failed
  first attempts.
- 200-400 accounted GPU-hours for stage-1 8192-clip small-geometry runs.
- 800-1200 accounted GPU-hours for one stronger subset/resolution/epoch
  follow-up on the best 2-3 configurations.
- 300-500 accounted GPU-hours reserved for reruns, probes, and final checks.

Total recommended request for a modest conference attempt: 1500-1800 accounted
LUMI GPU-hours, plus storage. This remains within the EUR 1000 planning cap
under the current EUR 0.35/GPU-hour price, but only if the ROCm smoke benchmark
does not reveal a severe slowdown or dependency failure.

Plain money estimate:

| Experiment target | LUMI estimate | Rocket / UT HPC estimate | Interpretation |
| --- | ---: | ---: | --- |
| Minimal first cluster matrix | about EUR 45-67 | about EUR 95 | Useful first publishable-direction matrix, but probably not enough alone for the full paper. |
| Reasonable modest conference attempt | about EUR 525-630 | about EUR 750-900 | Smoke tests, stage-1 matrix, one stronger follow-up, probes, failed jobs/reruns, and final verification. |
| Full EUR 1000 cap | about 2857 gross LUMI GPU-hours | about 2000 gross Rocket GPU-hours | Budget ceiling; do not plan to spend all of it before reserves and final checks. |

The practical interpretation is that EUR 1000 should be enough for a modest
workshop or short-paper attempt, but not for SOTA-scale pretraining. The main
risk is not the nominal price; it is whether LUMI/ROCm runs the VideoMamba path
efficiently enough after the smoke benchmark.

Full SSv2/Kinetics/ImageNet-style SOTA pretraining is not in scope for this
budget.

## Cluster Execution Strategy

Use three stages:

1. Smoke.
   - Run `hpc/rocket_smoke.sbatch` or `hpc/lumi_smoke.sbatch`.
   - Confirm model import, masked batch, optimizer step, checkpoint round trip,
     feature extraction, and memory report.
   - Stop if VideoMamba smoke fails on ROCm.

2. Stage-1 subset.
   - Use `configs/hpc/stage1_matrix.yaml`.
   - Start with 8192 clips, 20 epochs, 4 frames, 64px crop, fixed seed set.
   - Run ViT baseline, VideoMamba attention row/tube layer-norm, and
     VideoMamba attention row/tube L2-normalized target.
   - Run probes on fixed balanced splits.

3. Final selective scale-up.
   - Pick only the 1-2 best VideoMamba configurations plus the ViT baseline.
   - Increase exactly one of: subset size, resolution, frame count, or epochs.
   - Keep seeds and manifests fixed.
   - Stop early if prediction variance collapses or probe results remain
     uninformative while the ViT baseline is healthy.

## Contact Email Draft

Subject: Request for Rocket/LUMI GPU allocation guidance for bounded video SSL study

Hello,

I am preparing a bounded video self-supervised learning study at the University
of Tartu and would like guidance on the appropriate Rocket or LUMI access path,
allocation/accounting setup, and expected billing.

The workload is PyTorch-based video representation learning. The initial
request is for smoke tests plus a modest pilot of roughly 1500-1800 accounted
LUMI GPU-hours, with SSv2 data stored in project storage. The first technical
risk is whether Mamba/SSM dependencies run correctly on LUMI ROCm; if not, a
CUDA Rocket pilot may be more appropriate.

Could you confirm:

- the correct request route and project/account setup;
- whether any UT annual internal compute allowance applies;
- the current Rocket and LUMI GPU-hour prices;
- whether a small-g LUMI pilot is appropriate for single-GCD PyTorch jobs;
- recommended storage location and quota for SSv2 videos and checkpoints?

Best regards
