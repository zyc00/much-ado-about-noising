# Selected progress examples: interpretation and review

## Figure outline

- Left: unchanged empirical action/residual correlation values for every qualifying task/group.
- Right (trajectories): three example episodes from one selected task per dataset;
  dashed line is the 12-demonstration task profile.
- Right (replicated): the two six-demonstration halves for the same selected task,
  with all qualifying full-task profiles in the background.

## What changed and why

The previous plot used five uniformly sampled tasks and task-averaged stage profiles,
not single-episode traces. The new plot explicitly selects examples with stage-scale
contrasts that recur across two episode halves. Selection uses BOTH halves and is
exploratory, not held-out validation. The old plot, its selection, and its source JSON
are unchanged. The left panel is recomputed from exactly the same empirical values.

Each task has 12 probed demonstrations. A fixed seed splits these 6/6. Progress bins
are ranked by residual RMS in one half; the other-half RMS of the highest three bins
is divided by that of the lowest three. Swap halves and use the smaller ratio as
the selection score. Select the highest-scoring task per dataset, among tasks with
at least eight common bins. Every candidate participates; no hand-edited curves.

Selected tasks and bidirectional descriptive ratios:

| Dataset | Selected task | Ratios |
| --- | --- | --- |
| RoboCasa-GR1 | Tray to tiered shelf | 2.29, 1.82 |
| LIBERO | Turn on stove and place moka pot | 1.32, 1.47 |
| Bridge | Upright hot-sauce bottle, cardboard fence | 1.81, 1.96 |
| Fractal | Pick green can from top shelf of fridge | 2.16, 2.23 |

For the trajectory version, the three episode shapes closest to the full-task
profile are shown, not the three highest-amplitude episodes. This is an additional
explicit illustrative selection, not a random sample. No smoothing, temporal
warping, missing-bin imputation, or phase annotation was introduced. Each episode
is shown at the ten sampled progress-bin centers; this is NOT a dense rollout.
Curves use a common full-task normalization, not individual episode rescaling.
The GR1 example includes a comparatively flat episode: not every demonstration
exhibits the same strength of variation.

The strongest discovery-half variation alone did not replicate for the initially
ranked GR1 task (other-half high/low ratio 0.994) and was weak for the initially
ranked LIBERO task (1.078). This motivated the explicitly exploratory two-half
selection; those exploratory failures must not be represented as independent
validation successes. The new selection cannot estimate prevalence.

## Sources and reproducibility

- `selected_progress_examples_manifest.json`: exact selection, selected task names,
  episode IDs, all plotted example values, normalization and source hashes.
- GR1/LIBERO: existing `analysis/paper/mse_scale/{gr1,pi05}.npz`.
- Bridge/Fractal: compact `*_episode_energy.npz` exports from the existing cluster
  probe `/mnt/pfs/yuchen/crossdataset_general_mse_20260907/`.
- Export only computes the same squared residual energy and aggregates within
  episode/progress bin; no new inference or training was performed.
- Every reconstructed task curve is checked against `expanded_summary.json`.
- Original source hash is checked after figure generation.

## Five-dimension self-review

1. Contribution: illustrates residual stage variation; not a new causal result.
2. Clarity: individual episodes, task means and episode-half curves are distinct.
3. Strength: selected examples show repeatable contrasts, with a weaker LIBERO effect retained.
4. Completeness: four datasets and all original task correlations retained; original
   random-example plot remains the general-coverage view.
5. Soundness: selection disclosed; two halves are NOT independent validation after
   selecting with both. Training-data residuals do not identify irreducible noise.

Claim: selected tasks show nonmonotonic stage-dependent fitting errors.
Evidence: measured episode and episode-half curves. Status: descriptive support.

Claim: averaging hides asynchronous contact peaks.
Evidence: no event-aligned/contact-labeled trajectories in this audit. Status: untested hypothesis.

Claim: every task has strong progress-dependent noise.
Evidence: not supported by original population profiles. Status: do not claim.
