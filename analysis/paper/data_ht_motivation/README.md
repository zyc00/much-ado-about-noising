# Data-side motivation for heteroscedastic Student-t regression

## Mini-outline

- Measure conditional action spread from demonstrations without using a trained policy.
- Verify that a state-local radius both varies and predicts unseen action spread.
- Remove both state-local and fixed coordinate scales before testing residual shape.
- Use the two observations to motivate separate scale adaptation and tail robustness.

## Cross-dataset scale protocol

The right panel of the primary scale figure covers 149 tasks and 3,688
demonstrations:

- RoboMimic: Lift, Can, Square, Tool-Hang, and Transport (1,000 demos);
- LIBERO: 40 tasks (480 demos);
- RoboCasa / GR1: 24 tasks (288 demos);
- Bridge / WidowX: 40 task-stratified tasks (960 demos);
- Fractal: 40 task-stratified tasks (960 demos).

For every task and episode-progress decile, an eight-step continuous-action
mean and radius are estimated from one fold of demonstrations. Stage bins are
ranked using only this reference-fold radius, and residual RMS is subsequently
measured on disjoint demonstrations. Action channels are normalized using
task-wide statistics from the same reference fold. Binary grippers are
excluded; the continuous GR1 hand joints are retained. The two LeRobot subsets
are sampled deterministically and balance 12 reference and 12 query episodes
per selected task.

The left panel uses converged MSE policies as operational conditional-mean
estimators. The 10th--90th-percentile span of per-input residual RMS is 4.62x
for GR1/RoboCasa, 2.14x for LIBERO, 1.90x for Tool-Hang, and 1.77x for
Transport. The corresponding full observed spans are 51x, 16x, 9x, and 8x.
These are training-fit diagnostics, so the independent data-only test in the
right panel is needed to establish that the scale structure is not just model
error.

In the cross-demonstration test, the held-out *variance* ratio between the
widest and narrowest reference-radius quintiles is 4.63x for RoboMimic, 2.01x
for LIBERO, 1.92x for RoboCasa/GR1, 2.88x for Bridge/WidowX, and 1.83x for
Fractal. All task-bootstrap 95% confidence intervals exclude one; within-task
stage permutations give `p < 2.5e-4`.

### Reconciliation with the learned GR1 scale

The learned HT scale and the former stage-radius plot used different units of
analysis. On the 2,400-state GR1 HT probe, predicted sigma spans 55.25x from
minimum to maximum and 9.35x from the 10th to 90th percentile. On the matched
frozen-MSE probe, per-input residual RMS spans 50.79x and 4.62x, respectively.
The former cross-dataset plot instead averaged demonstrations into ten coarse
stage cells before taking quantiles; its 1.73x GR1 span was therefore a
between-stage aggregate, not the state-level scale range. That number is no
longer used as the primary visual comparison.

## Tool-Hang state-local protocol

The source is the 200-demo human Tool-Hang dataset cached at the path recorded
in `summary.json`; the file SHA-256 is also recorded there. Each sample is an
8-step chunk of the six continuous controller channels. The binary gripper is
excluded because a Bernoulli-like command creates discontinuities that neither
a Gaussian nor a Student-t continuous likelihood should model.

Phases are extracted from sustained gripper events:

- frame transit: 10 frames after the first grasp to 45 frames before release;
- frame insertion: 40 to 8 frames before the first release;
- tool transit: 10 frames after the second grasp to 55 frames before release;
- tool hanging: 50 to 8 frames before the final release.

For each query, the state feature concatenates two consecutive observations
(object state, end-effector position/quaternion, and gripper position). The
query action is compared with 32 nearest states in the same phase. Neighbor and
query demonstrations belong to disjoint deterministic folds. At most 12
queries per demonstration and phase are retained so long episodes do not
dominate.

The local residual is a data-geometry proxy for conditional spread. It includes
state-matching and local-mean estimation error and should not be called an
identified sample of aleatoric noise.

## Main results

- The central 80% of independently estimated local radii spans 2.27x.
- Sorting query states by their reference-set radius produces a monotonic
  held-out residual-RMS sequence: 0.074, 0.084, 0.089, 0.103, and 0.143.
  The largest/smallest ratio is 1.93 (95% episode-bootstrap CI 1.80--2.08).
  Swapping the demonstration folds gives 1.91 (CI 1.78--2.07).
- Permuting radius-group assignments within each semantic phase gives
  `p < 2.5e-4`; the local relationship is not explained by the four phase
  labels alone.
- The held-out, coordinate-normalized residuals are centered near zero in both
  phase groups, but their fitted Gaussian radii differ: 1.14 for transit and
  0.82 for precision. The precision component is 28% narrower. Reversing the
  folds gives 1.14 versus 0.90, preserving the ordering.
- Frame insertion / transit: paired median RMS ratio 0.849 (95% episode-bootstrap
  CI 0.793--0.878); insertion is lower in 76% of demonstrations.
- Tool hanging / transit: paired median ratio 0.797 (CI 0.777--0.823); hanging
  is lower in 79% of demonstrations.
- After local and per-coordinate scale normalization, held-out
  `P(|z| > 3) = 0.00978`, versus `0.00270` for the fitted Gaussian (3.62x).
- The cross-fitted Student-t has `nu = 6.35` and improves held-out NLL by 0.0266
  nat per action dimension. Reversing the folds gives `nu = 6.11` and a 0.0219
  nat improvement.
- Excluding action elements whose target or local mean is within 0.001 of a
  controller bound retains 98.4% of the held-out elements and preserves the
  result (`nu = 6.53` and held-out NLL gain 0.0253). Thus, controller clipping
  does not explain the observed tail.
- The held-out high/low contrast is 1.84x, 1.93x, and 2.06x with 16, 32, and 64
  neighbors, respectively. Those sensitivity runs are stored in the sibling
  `data_ht_motivation_k16` and `data_ht_motivation_k64` directories.

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| Continuous actions have stage-dependent scale. | Across 149 tasks in five datasets, reference-fold radii predict 1.35--2.15x held-out RMS contrasts; all task-bootstrap CIs exclude one. | Supported across RoboMimic, LIBERO, RoboCasa/GR1, Bridge/WidowX, and Fractal. |
| Scale also varies within semantic phase. | The stricter Tool-Hang state-local probe predicts a 1.93x held-out RMS contrast after phase-preserving permutation. | Supported on human Tool-Hang. |
| Heavy tails are not solely caused by phase or coordinate scale mixing. | Held-out survival after local and per-coordinate normalization; 3-sigma events are 3.62x Gaussian. | Supported on human Tool-Hang. |
| Student-t is a better residual model than Gaussian. | Positive held-out NLL gain in both cross-fit directions. | Supported for this residual probe. |
| HT is universally optimal for robot learning. | These figures do not evaluate policy success. | Needs the main closed-loop experiments. |

## Reproduction

```bash
python scripts/probe_data_ht_motivation.py
python scripts/plot_data_ht_motivation.py
python scripts/probe_crossdataset_stage_scale.py
python scripts/plot_crossdataset_stage_scale.py
python -m unittest scripts/test_data_ht_motivation.py
```

The paper-ready assets are `fig_crossdataset_stage_scale.pdf`,
`fig_heavy_tailed_residuals.pdf`, and `figures.tex`. The earlier two-component
explanatory view is retained as `fig_phase_gaussian_mixture.pdf`, and the
state-local Tool-Hang validation is retained as `fig_phase_dependent_scale.pdf`.
The PNGs are inspection copies. `preview.tex` compiles the prose and both
primary figures under the official ICLR 2027 style.

## Five-dimension reviewer self-check

- **Contribution:** Pass for motivation; the figures isolate two empirical
  properties that map directly to the two HT components.
- **Writing clarity:** Pass; the first figure separately establishes scale
  variation and held-out predictivity; the second isolates the remaining tail
  mismatch.
- **Experimental strength:** Needs the paper's multi-benchmark policy results;
  this is a data analysis, not an effectiveness experiment.
- **Evaluation completeness:** The cross-fit result now spans five datasets and
  149 tasks; the state-local Tool-Hang and neighborhood-count controls remain as
  stricter within-phase checks.
- **Method soundness:** Pass as motivation. Claims are deliberately limited to
  local data spread and residual shape; no causal or aleatoric-noise
  identification is asserted.
