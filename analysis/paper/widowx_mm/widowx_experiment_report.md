# WidowX Flow same-state multimodality probe

## Bottom line

The WidowX result supports a narrower and more defensible version of the
Section 1 claim. The Flow head is **not mathematically unimodal everywhere**:
in an unbiased sample of 100 on-policy states, 22 have a confirmed gripper
mode and 6 have a confirmed arm-axis non-Gaussian split. After a geometric
audit requiring two resolved peaks and negligible gripper contribution, 3 of
the 100 states retain clear arm profiles. Their raw chunks are an
execute-versus-defer wrist motion or two time-shifted versions of the same
translation profile. A post-hoc reanalysis of the four predicted actions that
evaluation actually executes finds 10 arm positives and 6 resolved arm
profiles, so the multimodality is not confined to predictions discarded at
the next replan. A direct same-state closed-loop fork then rejects a
persistent-route interpretation for the tested modes: mode identity is
decodable during the forced action prefix (65/72 branches) but not in the
middle (40/72) or late (31/72) rollout, and late mode-centroid separation is
smaller than within-mode rollout spread in every motion-producing case.

The paper should therefore claim that the stochastic head is used mainly for
**local event/phase uncertainty**, not that its output is literally unimodal.

## Model and state sources

- Checkpoint: `ft_wxflow/checkpoint-20000`, the evaluated GR00T WidowX Flow
  checkpoint (4 sampling steps, 8 action steps, 7 action dimensions).
- Training-state screen: 3,000 Bridge states.
- On-policy screen: 1,000 observations drawn from the checkpoint's own saved
  rollouts, balanced before random subsampling across `close the drawer` and
  `put the eggplant in the yellow basket`.
- Uniform confirmation sample: 100 of those on-policy states, selected without
  reference to their screen score (56 drawer, 44 eggplant).

## Test

For each state, the observation and all deterministic vision/language/state
features are held fixed. Only the action head's Gaussian seed changes, and the
sampler uses the exact evaluation setting (4 inference steps).

1. Screen each state with 8 sampled action chunks.
2. For a confirmation state, draw 32 new chunks and fit PC1 separately for the
   full chunk, arm, translation, rotation, and gripper subspaces.
3. Draw 64 additional, independent chunks and compare a one-Gaussian model to
   a two-Gaussian model on the fixed discovery axis.
4. Call a statistical split only when BH-corrected q <= 0.05, delta-BIC >= 10,
   both components have at least 20% mass, and their means are separated by at
   least 2 pooled standard deviations.
5. Because a two-Gaussian fit can approximate a skewed shoulder, separately
   audit whether the fitted density has two resolved peaks. A valley-depth
   threshold of 0.25 is used only as a descriptive geometric audit, not as an
   additional pre-registered hypothesis test.

The simulated single-candidate power at K=64 is deliberately conservative:
6% for a balanced 3-sigma mixture and 61% for a balanced 4-sigma mixture (2%
and 34%, respectively, when the smaller mode has 20% mass).

## Results

### Broad screen and hard-case confirmation

| source | screened | arm separation above matched-null q95 | gripper separation above q95 | hard cases deep-tested | confirmed full / arm / gripper |
|---|---:|---:|---:|---:|---:|
| Bridge training states | 3,000 | 9.3% | 25.9% | 55 | 39 / 16 / 41 |
| Flow rollout states | 1,000 | 8.2% | 18.4% | 48 | 25 / 10 / 23 |

The hard cases are intentionally selected from the largest screen scores, so
their confirmation fractions are evidence of existence, not prevalence. Most
training-state arm fits are correlated with the gripper split. On rollout
states, three especially clean candidates have two resolved arm peaks and at
most 5% of their between-cluster energy in the gripper.

### Unbiased on-policy prevalence estimate

| tested subspace | confirmed states / 100 | 95% Wilson interval |
|---|---:|---:|
| full action chunk | 27 | 19.3--36.4% |
| arm | 6 | 2.8--12.5% |
| translation | 10 | 5.5--17.4% |
| rotation | 5 | 2.2--11.2% |
| gripper | 22 | 15.0--31.1% |

All 27 full-chunk positives are accounted for by the gripper or arm tests: 22
also pass the gripper test and 5 pass the arm test. The sixth arm positive is
not visible in the full space because unrelated dimensions dilute its axis.
The gripper structure is task-specific in this sample: all 22 positives occur
among the 44 eggplant states, while none of the 56 drawer states passes the
gripper test.

Among the six arm positives:

- two drawer states have two resolved peaks, zero gripper energy, and a clear
  wrist-motion-versus-no-wrist-motion split;
- one eggplant state has a resolved, time-shifted lateral translation profile
  (90% arm and 10% gripper energy);
- one eggplant split is 79% gripper-correlated despite being found on the arm
  axis; and
- two drawer fits are shallow shoulders rather than clearly separated density
  peaks in this confirmation batch.

Thus the strict statistical rate is 6/100, while the visually and geometrically
clear arm-profile rate is 3/100 (95% Wilson interval 1.0--8.5%). Examples reach
12.4 degrees in wrist yaw or 1.5 cm in lateral displacement, so these are not
floating-point jitter.

The mean translation curves also favor a phase interpretation: their
same-step separation is 1.2--1.4 cm, but the symmetric nearest-point distance
between the two mean curves falls to 0.5--0.8 cm after allowing progress along
the eight-step path to differ.

### Executed-prefix reanalysis

The policy predicts eight actions but evaluation executes four before
replanning. Repeating the discovery/confirmation analysis on this four-step
prefix of the already saved, uniformly sampled batches gives:

| tested subspace | statistical positives / 100 | two resolved peaks / 100 |
|---|---:|---:|
| full executed prefix | 21 | 17 |
| arm | 10 | 6 |
| translation | 11 | 7 |
| rotation | 7 | 2 |
| gripper | 12 | 12 |

Of the six resolved arm-prefix cases, three are drawer states with zero
gripper contribution, two are arm-dominated basket states, and one basket
state is 83% gripper-correlated. This analysis was added after inspecting the
full-chunk result, so it is labeled post hoc; it nevertheless uses the same
fresh discovery and independent confirmation samples and applies BH correction
across all 100 states. The result rules out the convenient explanation that
the observed modes live only in the unexecuted half of the predicted chunk.

### Adjacent-state confirmation

Six candidates were re-sampled with an independently fitted center-state axis
at rollout offsets -4, -2, -1, 0, +1, +2, and +4 (64 fresh chunks at every
offset; BH correction across all 42 tests).

- Drawer state `267:1`: the rotation split passes at -2, 0, and +1. At -1 the
  secondary component falls to 9% mass, and by +4 the split disappears.
- Drawer state `385:0`: the arm split passes only at -1 and 0.
- Eggplant state `73:3`: the arm split passes at 0 and +2; the +1 frame narrowly
  misses the delta-BIC threshold (9.4 versus 10).
- Drawer state `11:2` does not reproduce under the full strict rule.
- Drawer state `440:4` meets the numerical mixture rule only at the center but
  its fitted density has one peak, so it is not treated as a resolved mode.
- The gripper control is balanced only at the center state; on neighboring
  frames one event-time component rapidly becomes a small minority.

The modes therefore occupy short temporal windows around local maneuvers. The
raw action profiles show “rotate now versus not within this chunk” and the same
translation pulse shifted within the 8-step horizon. This supports a phase or
event-boundary interpretation rather than persistent competing routes.

### Direct same-state closed-loop fork

We test route identity causally rather than infer it from marginal action
plots. The scan fits modes only on the four predicted actions that evaluation
actually executes. It screened 165 on-policy states and triggered 66 deep
tests. A candidate is accepted only with a fresh 32-sample discovery axis and
an independent 64-sample confirmation batch, raw permutation p <= 0.01,
delta-BIC >= 10, separation >= 2 pooled SD, at least 20% component mass, two
density peaks, and valley depth >= 0.25. Four cases pass (two independently
seeded episodes per task); all have Bonferroni p = 0.026 after correcting over
both arm groups and all 66 adaptive deep tests.

For each case, we snapshot the complete simulator and controller state, select
12 typical confirmation chunks from each component, restore the identical
state for all 24 branches, force the assigned four-action prefix, and then let
the stochastic policy replan through 48 total simulator steps. A fixed-action
replay check restores end-effector pose to within 1.6e-6. Three cases produce a
measurable initial pose split. In the fourth, a statistically strong
translation-space split produces no end-effector displacement (maximum 6e-6
cm over 48 steps); it is retained as an action-space result but excluded from
the route-persistence aggregate.

| state | task / mode group | delta-BIC / sep. | forced pose gap (cm / deg) | late spatial between/within | late-path p | endpoint-metric p | successes A / B |
|---|---|---:|---:|---:|---:|---:|---:|
| `ep0:m10` | drawer / rotation | 36.4 / 3.65 | 4.96 / 15.2 | 0.33 | 0.648 | 0.100 | 0 / 0 |
| `ep1:m30` | drawer / translation | 11.5 / 3.21 | 3.85 / 5.1 | 0.24 | 0.843 | 0.256 | 1 / 1 |
| `ep3:m2` | basket / rotation | 12.3 / 4.06 | 0.64 / 7.1 | 0.35 | 0.532 | 0.669 | 4 / 2 |
| `ep4:m40` | basket / translation | 53.8 / 2.81 | 0 / 0 | n/a (no arm actuation) | 1.000 | 1.000 | 0 / 0 |

The intervention has a strong positive control. Leave-one-branch-out decoding
of the assigned mode from full end-effector pose is 65/72 during the forced
chunk (90.3%, 95% Wilson interval 81.3--95.2%, permutation p = 0.0002). It
falls to 40/72 in the middle (55.6%, p = 0.253) and 31/72 late (43.1%, p =
0.808). The pattern holds in each state: forced-prefix accuracy is
83.3--95.8%, whereas late accuracy is 29.2--50.0%. Across the three
motion-producing cases, the late Euclidean mode-centroid distance is only
0.24--0.35 times the radial within-mode RMS;
none has a significant late path, endpoint task metric, or success split (all
p >= 0.53, 0.10, and 0.64, respectively). Thus the confirmed action modes
cause different immediate maneuvers but do not remain identifiable as
closed-loop geometric routes in these states.

Normalization note: the immutable run JSON records the initial implementation's
per-coordinate within-mode RMS. The table, figure, and audit JSON recompute the
ratio from the raw trajectories using radial 3-D RMS, which is the denominator
consistent with Euclidean centroid distance; the probe source is corrected for
future runs. Decoding and all permutation tests are unchanged.

## Claim boundary for the paper

### Continuous mean executability

A separate exact-state intervention tests the MSE target after excluding the
gripper. At 40 on-policy states (20 per task), the six-channel arm mean is
estimated from 64 same-observation draws and compared with eight fresh sampled
chunks and the nearest sampled medoid. Every matched branch uses the identical
sampled gripper sequence. The executed mean lies outside the empirical 95%
reference-path envelope in 0/40 states; its median path distance from the
reference centroid is 0.415 within-sample RMS for position and 0.382 for
orientation, versus 0.770 and 0.575 for the medoid. Full details and raw
artifacts are in `widowx_arm_mean_report.md`.

Supported:

> Across three large-policy stacks, stochastic action heads rarely express
> alternative high-level routes. Their structured sample variation is
> dominated by the timing of discrete gripper/hand events; WidowX additionally
> exhibits sparse arm modes that cause different immediate maneuvers but lose
> their identity under closed-loop replanning in the tested same-state forks.

Not supported:

> Current large robot policies are unimodal, or never sample two different arm
> chunks.

## Artifacts

- Main probe: `scripts/probe_widowx_multimodality.py`
- Adjacent-state probe: `scripts/probe_widowx_mode_offsets.py`
- Closed-loop fork probe: `scripts/probe_widowx_branching.py`
- Raw training screen: `train_v1.json`, `train_v1.npz`
- Raw rollout hard-case screen: `rollout_v1.json`, `rollout_v1.npz`
- Raw uniform rollout sample: `rollout_random_v1.json`, `rollout_random_v1.npz`
- Raw adjacent-state audit: `offsets_v1.json`, `offsets_v1.npz`
- Executed-prefix reanalysis: `rollout_random_prefix4_v1.json`
- Raw closed-loop forks: `widowx_branching_prefix_v1.json`,
  `widowx_branching_prefix_v1.npz`
- Figure script: `make_widowx_mm_audit.py`
- Closed-loop figure script: `make_widowx_branching_audit.py`
- Closed-loop fork figure and statistics: `widowx_branching_audit.png`,
  `widowx_branching_audit.pdf`, `widowx_branching_audit.json`
- Paper-ready representative figure: `widowx_main_modes.png` / `.pdf`
- Hard-case figure: `widowx_rollout_audit.png` / `.pdf`
- Uniform-sample arm figure: `widowx_random_arm_audit.png` / `.pdf`
