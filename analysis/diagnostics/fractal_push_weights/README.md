# Check: are demonstrated drawer pushes downweighted by HT?

## Answer

A local instance exists, including a clear top-drawer example resembling the
user-highlighted first push in rollout 1240. It is **not a universal property**
of drawer pushing. Frozen-checkpoint re-evaluation does not establish which
historical training updates caused the rollout failure.

## Measurement

- Frozen `/mnt/pfs/yuchen/groot/ft_fr_nu224/checkpoint-20000`, nu=224,
  shared sigma over 8 x 7 coordinates (gripper included), beta=0, no scale clamp.
- Uniform random 20 episodes for each close-bottom/middle/top instruction;
  all 2,006 valid chunk starts from 60 episodes, seed 20260909.
- Training demonstrations with actual labels, eval mode without dropout or
  augmentation. This is a loss-level diagnostic, not recorded training history.
- Reconstructed NLL agrees with the model's actual loss to max 2.38e-7.
- S = sum of squared normalized residuals, d=56:
  `gate = (nu+d)/(nu+S/sigma^2)`;
  `W = gate/sigma^2`;
  `d(loss per dimension)/d(prediction) = W*(prediction-label)/d`.
  W is a coefficient per unit output residual, not the parameter-gradient norm.
  Shared-backbone gradients through the sigma branch are not included in this
  mean-output diagnostic. A common minibatch averaging factor is omitted.

## Concrete visual example: training episode 19034, close top drawer

Approximate visible inward-motion window: steps 9 through 18, identified from
the image sequence (row 4 of `blind_sequences_task2.jpg`). Baseline: preceding
steps 0 through 8. Each number below is the median over its interval.

| Quantity | Before push | Visible push | Push / before |
|---|---:|---:|---:|
| Predicted sigma | 0.1675 | 0.3366 | 2.009 |
| Label-residual RMS | 0.2341 | 0.2330 | 0.995 |
| Student-t gate | 0.8624 | 1.1152 | 1.293 |
| Mean-output weight W | 30.455 | 9.883 | 0.325 |
| Mean-output gradient norm | 0.9547 | 0.3086 | 0.323 |

Thus, at this checkpoint, comparable residual magnitude gets about 68% less
mean-output gradient during this push. Sigma rises while residual RMS does
not; the Student-t gate increases rather than adding suppression. This is
local sigma-driven downweighting, **not proof that it caused under-learning**.
The same state's training target need not be equally predictable from the
available observations, and these are only a few temporally correlated samples.

## Generality checks

1. Removed the previous late-episode (last 40%) restriction entirely.
   Action-defined forward proxy: mean raw positive x dominates lateral motion
   over the chunk. Among 856 forward samples, sigma median=.219 versus .240
   for 1,150 others; W=20.52 versus 16.81. Within-episode W ratio median=1.218,
   bootstrap 95% CI [1.021, 1.504] across 59 qualifying episodes.
2. Strong forward actions (also top quartile of within-episode mean-translation
   magnitude): 126 samples, sigma=.232, gate=1.011, W=17.73; all others
   sigma=.231, gate=1.009, W=18.43. Within-episode W ratio=1.181,
   CI [.924, 1.301] across 21 qualifying episodes. No general strong suppression.
3. Manually inspected the first four randomly sampled episodes per instruction
   in image-only sequence panels. Nine had approximate visible inward-motion
   intervals; three were too ambiguous and excluded with reasons recorded in
   `visual_push_intervals.json`. Of these nine, three had lower median W during
   pushing than beforehand. Episode 19034 has the strongest drop (0.325x);
   episodes 4097 and 11697 have milder drops (0.868x and 0.858x). Median W
   ratio across the nine is 1.023. These small manual windows are a diagnostic,
   not ground-truth contact labels or an independent benchmark.

## Implication

The rollout observation and training demonstration **can** show analogous
sigma increases during a real push. But "all large pushes are suppressed" is
not supported. This justifies investigating local sigma/gradient calibration,
not claiming that reducing nu or globally removing heteroscedastic weighting
will repair close-drawer. Changing nu does not directly remove 1/sigma^2.

Old `analysis/paper/fractal_debug.md` statements that demonstrated policies are
identical, that failures are proven outside training support, or that all
loss-level explanations are closed should not be treated as established facts.
Aggregate fitting agreement is weaker than those claims.

## Artifacts

Local: `analysis.json`, `visual_push_analysis.json`, `protocol.json`, `done.json`,
image sequence sheets and manual interval JSON in this directory.
Raw cluster: `/mnt/pfs/yuchen/fractal_push_weights_20260908/` contains
`probe.npz` (labels, predictions, sigma, S, gate, W, identities), `frames.npz`,
`derived.npz`, and the exact probe/analysis scripts under `code/`.

The GPU probe completed successfully. Its initial shell attempted analysis
before that script finished copying; analysis was then run successfully on
the utility pod. No checkpoint or training job was modified.

## Follow-up mechanism audit

See `mechanism_audit.json`, generated by `scripts/audit_fractal_push_mechanism.py`.

For episode 19034, push window t9..18:

- First forward prediction is below the normalized training target at 10/10
  states; the sum of decoded predictions is 83.4% of the sum of decoded
  training targets. First-x normalized mean bias is -0.1344. The comparison
  uses clipped training labels: 5/10 raw x labels exceed the percentile cap.
  Raw first-x label mean=.17735, decoded clipped target mean=.16337, HT
  prediction mean=.13630. Clipping is shared by the baseline recipe, not
  uniquely an HT mechanism.
- Gripper contributes 59.8% of full-chunk residual energy; the currently
  executed first gripper action contributes only 1.0% of total energy;
  gripper error in the last four predicted steps contributes 38.5%.
  The first action's gripper target is always 0 (normalized -1), but a 0->1
  change enters the end of the chunk at state t14 and moves earlier in the
  horizon in subsequent states. The model predicts a smooth transition.
- The same shared sigma/weight is applied to every action coordinate at every
  chunk step. Thus future gripper errors can influence the scale used for
  current arm supervision. The numbers do not prove that those gripper errors
  caused this sigma prediction. If sigma is held fixed and gripper is removed
  only from S and d, the median weight increase is merely 3.6%; a separately
  trained scale could behave differently. No Bernoulli-model improvement has
  been established here.
- Moving both interval boundaries independently by up to +/-2 steps gives
  25 variants. All retain lower push-stage weight; push/pre ratio ranges
  .281..724. This local finding is not an exact-boundary artifact.
- Frozen-parameter nu counterfactual: nu=14 increases the per-state push
  weight by median 1.529x versus nu224, while before-push weights fall to
  .708x. Nu7 gives 1.660x and .678x. Smaller nu need not suppress all samples:
  W=(nu+d)/(nu*sigma^2+S); when S/sigma^2<d it increases the weight as nu falls.
  These calculations freeze both prediction and sigma, so they do not predict
  the outcome of fine-tuning with a new nu.

Matched Flow follow-up: pod `yuchen-fractal-push-flow-0908` has been submitted
and is pending a GPU as of this audit. It will evaluate all states in the nine
visually inspected episodes using 16 samples per state and a separate zero-noise
initialization. It checks that processed labels exactly match the HT probe.
Results will be saved as `matched_flow.npz`, `matched_flow_done.json`, and
`matched_flow_analysis.json` at the cluster root. No Flow comparison result is
claimed before these files exist.
