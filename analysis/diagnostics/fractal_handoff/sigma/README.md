# HT videos aligned with the checkpoint's predicted sigma

Scenes: 1235 (bottom drawer), 1236 (middle), 1240 (top).
Checkpoint: `/mnt/pfs/yuchen/groot/ft_fr_nu224/checkpoint-20000`.

## Measurement and alignment

We reset each original scene seed and execute the recorded HT actions.
Before every action t, we compare the physical state, exposed controller state,
sticky-gripper state and exact input-image hash with the original trajectory.
All 900 states pass; physical/controller differences are zero. We also query
the original policy on that observation: all 900 action predictions match the
original saved predictions exactly (maximum absolute difference 0).

The native inference diagnostic additionally evaluates the trained sigma
decoder on the same hidden representation. The plotted scalar is
`mean(softplus(s_raw + ht_sbias)) + 0.001`, over 8 chunk steps and all 7 valid
action coordinates, including gripper; `ht_sbias = -0.5093` (rounded).
It is the **shared scale in checkpoint-normalized action space**, not a
physical positioning tolerance, not measured residual, and not sigma inferred
from action magnitude. There is no demonstration label for these rollout states.
No model weights, training jobs or action readout were changed.

`sigma[t]` aligns with the observation **before** action t. The blue curve is
native drawer-joint qpos in cm before the same action. Success is qpos <= 5 cm.
All three static figures and annotated videos share the same sigma and opening
axis ranges, start at zero, and show unsmoothed measurements. Gray curves in
the videos show the full trajectory; colored curves/cursors show progress.

Videos use the existing observation-camera recordings, sampled every 3 env
steps, played at 10 fps (~10 s for 300 steps); this is not wall-clock speed.
Static frames are at steps 0, 30, 60, 120, 210 and 297, explicitly labeled;
red markers identify those times on the sigma curve. Displayed frames are
decoded from the original compressed video; exact uncompressed frame snapshots
are retained in the cluster raw NPZs.

## Files

- `ht_<seed>_sigma.png` / `.pdf`: six frames + sigma + drawer opening.
- `ht_<seed>_sigma.mp4`: video with synchronized sigma and opening curves.
- `<seed>_curves.npz`: lossless numeric probe arrays (300 steps), including sigma,
  decoder component averages, drawer qpos, action translation norm and actual
  EEF displacement. Component averages are not independently trained scales.
- `<seed>.json`: checkpoint, definition and replay verification.
- `1235.npz.incomplete-download`: **invalid partial download; do not use**.

Complete raw files and the 900-call native sigma log are on the cluster:
`/mnt/pfs/yuchen/fractal_rollout_sigma_20260908/`.
The original action/state trajectories are under:
`/mnt/pfs/yuchen/fractal_handoff_20260908/collect/`.

Scripts: `probe_fractal_rollout_sigma.py`, `export_fractal_sigma_curves.py`,
`plot_fractal_rollout_sigma.py` under repository `scripts/`.

## Descriptive readings

| Scene | Sigma min | Median | Max | Last-60 median |
|---|---:|---:|---:|---:|
| 1235 | 0.1920 | 0.2258 | 0.4010 | 0.2100 |
| 1236 | 0.1723 | 0.3505 | 0.4521 | 0.3734 |
| 1240 | 0.1669 | 0.3811 | 0.4791 | 0.3825 |

These selected failures do not establish a causal mechanism or a general
relationship between contact, action magnitude and sigma.

## Figure self-review

- Contribution: diagnostic of these three trajectories only, not a new method claim.
- Clarity: explicit timestamps, shared axes, sigma units and gripper inclusion.
- Evidence: sigma is measured from the actual checkpoint; all replay checks pass.
- Completeness: all user-requested scenes and all 300 steps are retained.
- Soundness: no causal claim about suppressed training gradients is made from
  rollout sigma without demonstrated targets/residuals.
