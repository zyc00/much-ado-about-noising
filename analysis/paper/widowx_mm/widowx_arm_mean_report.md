# WidowX continuous-arm mean executability probe

## Bottom line

The estimated conditional mean of the six continuous arm channels remains an
in-family executable motion.  Across 40 exact on-policy simulator states, it
never falls outside the empirical 95% envelope of sampled reference paths.
Its executed path is substantially closer to the sampled outcome centroid
than even the sampled arm medoid.  This directly weakens the standard
"MSE averages incompatible strategies into an invalid action" explanation.

The gripper is never averaged in this experiment.  Each comparison uses the
same sampled gripper sequence for the sampled-arm, mean-arm, and medoid-arm
branches, matching the paper's separate Bernoulli treatment of contact state.

## Protocol

- Policy: GR00T WidowX Flow checkpoint `checkpoint-20000`, at the exact
  four-step evaluation-time execution horizon.
- States: 40 on-policy states, 20 drawer and 20 eggplant-to-basket, drawn from
  multiple independently seeded episodes and multiple stages of each rollout.
- Mean estimate: 64 same-observation Flow draws at each state; average only the
  six continuous arm channels.
- Reference outcome cloud: eight additional, fresh action chunks.
- Intervention: restore the complete simulator and controller state for every
  branch.  For each reference gripper sequence, execute (1) its sampled arm,
  (2) the estimated mean arm, and (3) the sampled arm medoid nearest the mean.
- Metric: RMS distance of the candidate's executed pose path from the reference
  path centroid, divided by radial within-reference RMS. A ratio below one is
  inside one typical radius of the sampled outcome cloud.
- Restore audit: repeated execution of the same chunk returns identical pose
  paths (maximum absolute discrepancy 0.0).

## Results

| executed path | conditional mean, median (IQR) | sampled medoid, median (IQR) | mean closer / 40 | paired p |
|---|---:|---:|---:|---:|
| position / within RMS | **0.415 (0.266--0.540)** | 0.770 (0.521--0.949) | 32 | 6.5e-8 |
| orientation / within RMS | **0.382 (0.282--0.516)** | 0.575 (0.334--0.723) | 29 | 2.0e-5 |

The mean position path lies within one reference RMS at 39/40 states and the
mean orientation path at 40/40.  More stringently, no state places the mean
outside the empirical 95% reference-path envelope.  The result is consistent
within each task: median position ratios are 0.384 for drawer and 0.419 for
basket, versus medoid ratios of 0.765 and 0.788.

## Interpretation and boundary

This is causal evidence that the local continuous mean is executable, not a
claim that the full joint distribution is exactly Gaussian.  The experiment
forces only the four actions used before replanning, so it establishes local
mean validity rather than full-episode performance of a deterministic mean
policy.  Together with the held-out one- versus two-Gaussian audit and the
closed-loop mode forks, it supports the following transition:

> Once discrete contact state is modeled separately, the continuous action
> target is predominantly one-basin and its mean is a valid motion. Therefore
> MSE's weakness cannot be attributed primarily to averaging incompatible
> strategies; we must instead examine the shape and scale of residual noise.

## Artifacts

- Probe: `scripts/probe_widowx_arm_mean.py`
- Cluster specification: `scripts/cluster/widowx_arm_mean_pod.yaml`
- Raw output: `widowx_arm_mean_v1.json` / `.npz`
- Analysis: `analyze_widowx_arm_mean.py`
- Summary: `widowx_arm_mean_summary.json`
- Diagnostic figure: `widowx_arm_mean.png` / `.pdf`
