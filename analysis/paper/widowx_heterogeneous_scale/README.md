# Matched general-MSE and Flow scale probe on WidowX

This replaces the incorrect Q1/Q3/Q5 specialist figure. No specialist data,
action-magnitude filtering, or rollout-action labels enter the current figure.

Regenerate:

```bash
python scripts/plot_widowx_heterogeneous_scale.py
```

## Measurements

- Frozen general policies: `groot/ft_wxmse/checkpoint-20000` and
  `groot/ft_wxflow/checkpoint-20000` on the cluster.
- Dataset: `groot/bridge_orig_lerobot`. Select the three most frequent nonempty
  lowercase task instructions among episodes with at least 24 frames, before
  examining residuals: sweep into pile, open the drawer, close the drawer.
- Sample 48 episodes per task, then 12 evenly spaced valid action-chunk starts
  per episode. No action-magnitude selection. All 1,728 observations are shared
  by the two policies; the plotter asserts identical episode/frame IDs and
  normalized action labels. Six continuous channels over eight steps; no gripper.
- These are diagnostics on the policies' demonstration dataset, not estimates
  on a policy-held-out dataset. Scale-fitting holdout is described separately.

Panel a uses the general MSE residual RMS and demonstration-action RMS.
Panel b preserves task-specific progress curves. A point is the square root
of mean residual energy in a progress decile, with equal episode weighting
within the decile. Shading bootstraps episodes. At least eight episodes are
required per displayed decile. Full action chunks exclude the last seven
frames; no zero padding is used to manufacture an end-of-episode decline.
Panel c resamples Flow 16 times on exactly the same observations and uses the
same label scale as a, not an executed or generated action. Spread is the
coordinate RMS deviation from the sample mean. Task colors are consistent
through a/b/c. Both scatter axes are divided by within-task geometric means;
reported correlations standardize ranks within each task.

Panel d fixes the general MSE action prediction and cross-fits Gaussian
variance over five disjoint episode folds. The baseline fits one constant
variance per task. The alternative predicts variance from proprioception,
descriptors of the frozen MSE prediction, and a 20-component PCA of its frozen
observation features. PCA and the variance regressor use fitting episodes only.
The regressor is fixed in advance: ExtraTrees, 300 trees, leaf size 32, direct
regression of residual energy relative to the task variance. No demonstrated
action, action-label magnitude, query residual, or oracle progress enters the
test variance predictor. Two explicitly labeled points show average Gaussian
log likelihood, with 95% episode-bootstrap confidence bars. The paired gain's
confidence interval and all per-task results are saved in the summary JSON.

## Reproducibility and review

Measured results: within-task residual/label Spearman rho = 0.773 (95% episode
bootstrap CI [0.740, 0.798]); matched Flow spread/label rho = 0.653 ([0.610,
0.689]). Fixed-scale and input-dependent Gaussian test log likelihoods are
0.149 and 0.207 nat/dim. The paired gain is +0.0575, 95% episode-bootstrap CI
[0.0482, 0.0667]; the per-task gains are +0.0380, +0.0713, and +0.0632.

The preview compiles with the unmodified official ICLR 2027 style downloaded
from https://raw.githubusercontent.com/ICLR/Master-Template/master/iclr2027/iclr2027_conference.sty.
The standalone plot is designed at 5.5-inch width.

Probe: `scripts/probe_widowx_general_scale.py`.
Cluster job: `scripts/cluster/widowx_general_scale.yaml`.
Raw data: `raw/{mse,flow}_rank{0,1,2,3}.npz` and `raw/hg_rank0.npz`; provenance, fields and sha256 checksums in `raw/MANIFEST.md` / `raw/manifest.json`.
Cluster source: `/mnt/pfs/yuchen/widowx_general_scale_20260906/raw`.

The figure tests association and comparative likelihood fit. A residual RMS
measures prediction error, not a direct repeated-label estimate of irreducible
noise. Higher likelihood for input-dependent variance motivates a scale model;
the policy-performance ablations establish its control benefit.

Reviewer checks: the policy is general MSE (pass); a/c observations and labels
match (asserted); b retains progress (pass); d has self-contained method labels,
metric direction, and uncertainty notation (pass); original-policy holdout is
not claimed (pass).

## v2 layout (2026-09-06)

`fig_widowx_heterogeneous_scale_v2.{png,pdf}` from `scripts/plot_widowx_heterogeneous_scale_v2.py`: same data and
numbers. Panels a and c replace the 1,728-point scatter with the pooled point density (grey hexbin, log-log) and, per
task, the geometric-mean residual (a) / Flow spread (c) within label-RMS octiles with 95% episode-bootstrap bars; the
dashed diagonal is proportional scaling. Legend carries the task episode counts from
`bridge_orig_lerobot/meta/episodes.jsonl` (exact lowercase instruction; 53,192 episodes in total). Panels b and d
read the stored summary JSON. The v1 figure and `figure.tex` are untouched; point `figure.tex` at the v2 PDF to adopt it.
v5 (`fig_widowx_heterogeneous_scale_v5.{png,pdf}`, current script output): panels a and c show a seeded random subset of
120 states per task as small solid dots plus a per-task log-log least-squares line fitted on all 576 states of that
task; rho is still the within-task rank correlation over all 1,728 states. v2-v4 were intermediate layouts.
