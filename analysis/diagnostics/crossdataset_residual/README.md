# General-MSE residual structure across all available tasks

## Main result

All tasks are retained; this is a training-demonstration fit audit, not policy-held-out validation.
Action-label magnitude is not the same quantity as predicted movement magnitude used in some older probes.

| Setting | Tasks | Mean task rho | Positive rho tasks | Median other-half progress ratio | Both progress folds >1 |
|---|---:|---:|---:|---:|---:|
| GR00T / GR1 | 24 | 0.341 | 24/24 | 1.20 | 14/24 |
| pi0.5 / LIBERO | 40 | 0.145 | 37/40 | 1.12 | 33/40 |
| GR00T / WidowX | 3 | 0.773 | 3/3 | 1.58 | 3/3 |
| U-Net / Tool-Hang | 1 | -0.101 | 0/1 | 1.18 | 1/1 |
| U-Net / Transport | 1 | 0.049 | 1/1 | 1.06 | 1/1 |

Each dot in all_tasks.png is one task. Panel a uses the RMS of the demonstrated label and prediction-minus-label residual in identical normalized continuous channels. Binary grippers are excluded; GR1 hand joints are retained.
Panel b ranks 10 progress bins by energy in one episode half, then evaluates the top/bottom thirds on the other half and swaps folds. Values above 1 support repeated stage-scale ordering, not a common increasing/decreasing trajectory shape. Half splits are for diagnostic replication, not policy training.
CSV contains all task names, episode-bootstrap pointwise CIs, both progress fold ratios, and all progress curves. Counts of CIs excluding zero are not multiplicity-adjusted discoveries. No significance claim is made from a ratio merely exceeding 1.

## Five-dimension self-review

- Contribution: describes cross-task fit structure; does not establish the cause of HT performance gains.
- Clarity: separates demonstration-label magnitude, model-predicted movement, and nonmonotonic progress dependence.
- Experimental strength: every available task retained, including negative and weak associations.
- Completeness: 24 GR1 + 40 LIBERO + 3 WidowX + 2 RoboMimic tasks; no comparable Fractal general-MSE archive included.
- Soundness: identical label/residual channels, complete executed windows, episode-disjoint progress replication. Conditional noise and contact causality not identified.

## Claim-evidence map

- Action magnitude is associated with fit residuals across settings | All per-task rho values in CSV | Use measured coverage; do not assert every task is strongly positive.
- Scale varies with progress in a task-dependent way | Other-episode-half stage ranking | Report effect sizes and both-fold consistency, not a universal monotonic trend.
- These effects generalize to unseen policy inputs | No policy-held-out data here | Not established.
