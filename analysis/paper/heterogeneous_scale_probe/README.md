# Heterogeneous-scale motivation figure

## Paper claim

The figure supports one narrow claim: the continuous-action residual
distribution has input-dependent scale. It does not identify every residual
as irreducible demonstration noise, and it does not motivate heavy tails; the
Student-t evidence belongs in the separate tail figure.

## Panels

- **(a) Residual follows label scale.** Across 2,880 held-out states, the
  converged MSE policy's residual RMS is positively associated with the
  continuous action-label RMS (within-task Spearman rho 0.341; within-task
  permutation p < 1e-4).
- **(b) Scale changes within tasks.** Each line is estimated separately from
  the 120 held-out states of one task after normalizing scale within that task.
  Eleven of 24 task-specific progress tests have uncorrected p < 0.05, and
  five remain significant after Benjamini--Hochberg correction (q < 0.05);
  these five are colored while the remaining tasks are retained in gray. The
  median task-specific max/min radius is 1.962x. This avoids averaging
  semantically unaligned progress stages across different tasks.
- **(c) Flow learns the scale pattern.** At each of 240 matched held-out
  states, 32 Flow
  action chunks are sampled. Flow spread is the square root of the mean
  coordinate-wise sample variance, equivalently the RMS deviation around the
  sample mean with the unbiased finite-sample correction. Its label-scale
  correlation is rho 0.194 (within-task permutation p 0.0042), and its
  ten-stage profile aligns with
  the MSE residual profile (Pearson r 0.922, p 1.5e-4). The latter is stronger
  and more honest than claiming that Flow spread alone has a significant
  progress effect: its standalone within-task progress permutation is p 0.13.
- **(d) Input-dependent Gaussian fits better.** The larger 2,880-state frozen
  MSE audit is divided into two disjoint episode halves. The action mean stays
  fixed. On each fitting half, the baseline estimates one variance per task;
  the heterogeneous model additionally predicts relative log variance from
  proprioception and episode progress using 500 extremely randomized trees
  (minimum leaf size 20), followed by one scalar likelihood calibration.
  Evaluation occurs only on the other episode half, with the folds swapped.
  Mean held-out Gaussian log likelihood rises from 1.134 to 1.175 nat per
  action dimension: a paired gain of 0.041, with task-bootstrap 95% CI
  [0.017, 0.061]. Twenty of 24 tasks have positive gain.

## Additional Flow check requested during figure design

The RMS difference between the Flow sample mean and the demonstrated label is
also structured: its within-task label-scale correlation is rho 0.284
(p 7.7e-6), and its progress-stage range is 2.281x (within-task permutation
p 0.0012). This supports the residual analysis but is omitted from the main
panel to keep the 1x4 layout readable.

## LIBERO replication (not shown in the main 1x4 figure)

The matched pi0.5/LIBERO probe contains 400 held-out states from 40 tasks and
excludes the binary gripper channel. It reproduces the two useful patterns:
MSE residual scale correlates with label scale (within-task rho 0.163,
p 0.0011) and varies across episode stages by 1.495x (permutation p < 5e-5).
Flow spread correlates with label scale even more strongly (rho 0.269,
p 4.8e-8), while its stage profile aligns with the MSE profile (r 0.850,
p 0.00185). As on GR1, the standalone Flow-spread progress test is weaker
(p 0.11); the paper should claim profile alignment, not an independently
significant progress effect for Flow. Flow mean-to-label RMS is structured in
both label scale (rho 0.364, p 5.7e-14) and progress (1.790x, p < 5e-5).

## Probe scope

- Model/dataset: GR00T N1.7 / RoboCasa-GR1, all 24 tasks.
- MSE audit: 120 policy-held-out states per task (2,880 states total), used in
  panels (a--b) and for the likelihood comparison in panel (d).
- Matched Flow probe: one policy-held-out state per task and progress decile
  (240 states), with exactly the same states evaluated by MSE and Flow.
- Action channels: all 29 continuous GR1 joint channels, including hand joints.
- Action chunk: 8 steps; all reported RMS quantities cover the full chunk.
- Flow: 32 independently seeded inference draws per state.
- Error bars in (a--c): resample whole tasks. Error bars in (d): task bootstrap
  over cross-fitted task-level log likelihoods.

The phrase “one shared scale” is used instead of “uniform,” because the
statistical baseline is a homoscedastic Gaussian, not a uniform probability
distribution. Panel (d) actually gives that baseline one scale per task, which
is more conservative than forcing a single scale across all 24 tasks.

## Files and reproduction

- `fig_heterogeneous_scale.pdf`: vector paper figure.
- `fig_heterogeneous_scale.png`: high-resolution preview.
- `fig_heterogeneous_scale_summary.json`: exact statistics and protocol.
- `raw/gr1_k32_rank*.npz`: four matched-probe shards.
- `figure.tex`: ICLR-ready inclusion and caption.

From the repository root:

```bash
python scripts/plot_heterogeneous_scale_evidence.py
```

The matched inference code is
`scripts/probe_mse_flow_heteroscedasticity.py`; the K=32 cluster launch spec is
`scripts/probe_gr1_flow_k32.yaml`.
