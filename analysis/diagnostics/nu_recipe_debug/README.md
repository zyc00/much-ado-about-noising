# Student-t ν: statistical fit versus policy-training recipe

Started 2026-09-08. This is an active debugging study, not a completed recipe
recommendation. No claim that smaller ν must improve control success.

Latest user-directed change: the three continuation branches and the old
evaluation pod were stopped/canceled. A fresh Fractal nu=14 run with
detached-mean Gaussian auxiliary (lambda=1) has been submitted instead.
See [NU14_GAUSSIAN_AUX.md](NU14_GAUSSIAN_AUX.md). Historical launch/evaluation
details below describe canceled jobs, not current commitments.

## Questions and current answers

1. **Which recipe works best?** Not yet established. The continuation study
   was canceled at user request; the active trial is fresh nu=14 plus a
   detached-mean Gaussian scale auxiliary.
2. **Is the small fitted ν just a coordinate-versus-vector mistake?** Not in
   the new probe: matching the actual joint likelihood still gives small ν.
3. **Why can large training ν nevertheless work better?** The current leading
   hypothesis is coupled mean/scale optimization and sample reweighting,
   rather than a mathematical rule converting marginal ν into d times ν.
   Statistical misspecification and mismatch between likelihood and control
   remain alternative explanations. They are not ruled out.

## Completed likelihood audit

Probe: `scripts/probe_nu_likelihood_gap.py`.
Raw data: `/mnt/pfs/yuchen/ht_sigma_context_20260908/{fractal,bridge,gr1}/probe.npz`.
Each source `protocol.json`, also embedded in the result, records checkpoint,
episode IDs, task selection, preprocessing, and agreement with inference.

- Frozen predictions and their **own learned scalar σ**, no new policy fit.
- Actual valid action chunk: 8×7 for Bridge/Fractal; 8×29 for GR1.
- Joint Student-t log likelihood, including all ν-dependent normalizers.
- Location remains the learned prediction; calibrate one multiplicative scale
  on 2/3 of episodes per task, evaluate density on the other 1/3.
- This is a **density-calibration holdout**, not data excluded from policy
  training. Each dataset has 144 episodes and roughly 2,880 states.
- Bridge/Fractal use their 24 most frequent nonempty instruction groups;
  GR1 uses all 24 tasks. In particular, this Fractal selection does not cover
  move-near, whose instructions are fragmented. It is not a six-task SIMPLER
  population estimate.
- Repeat with continuous channels only, fixed-coordinate scales, and a
  calibration-estimated full second-moment matrix with 5% shrinkage.
  The latter is not a full joint Student-t covariance EM fit.

Initial coarse-grid results, all training channels, learned σ plus one
calibration multiplier:

| Dataset | Policy training ν | Joint fitted ν | Fixed-σ joint fitted ν |
|---|---:|---:|---:|
| Fractal | 224 | 7 | 7 |
| Bridge | 224 | 10 | 10 |
| RoboCasa-GR1 | 928 | 5 | 5 |

Removing the gripper gives joint ν≈3 for Fractal and ≈5 for Bridge, rather
than explaining away the discrepancy. GR1 hand joints are retained.
Allowing a fixed full covariance gives ν≈7, 14, 5 respectively: an isotropic
covariance assumption alone does not remove the low-ν preference here.

`likelihood/` preserves the first coarse-grid run. A refinement in
`likelihood_v2/` solves the scale score by a bracketed root (rather than a
finite fixed-point iteration), and continuously refines ν using calibration
likelihood only. See its JSON rather than treating grid values as exact
estimates. No held-out selection of ν or scale is performed.

The refined joint ν estimates (all training channels, calibrated own σ) are
**6.53 Fractal, 9.82 Bridge, 5.66 GR1**. The corresponding fixed-σ estimates
are 5.97, 9.85, 5.11. These are fitted descriptive parameters, not theoretical
predictions of the success-optimal training hyperparameter.

Synthetic checks use d=56 and true marginal ν=7:

- A shared latent scale gives joint ν≈7 and marginal ν≈7.
- Independent coordinate-wise t residuals give joint ν≈56, marginal ν≈7.
- Seven independent t blocks give joint ν≈20, marginal ν≈7.

Thus marginal tails alone cannot identify a joint t, but this possible
explanation is **not sufficient for our measured residual vectors**.
The joint log-density implementation is numerically checked against
[SciPy's multivariate Student-t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html).

## Optimization mechanism to test

Let S=||r||², q=S/σ², and d be the number of valid chunk coordinates. For the
joint t, ignoring constants independent of μ and σ,

    L = (ν+d)/2 log(1 + S/(ν σ²)) + d log σ.
    g_μ = w r/σ²,             w = (ν+d)/(ν+q).
    g_logσ = d - w q.

If training averages L over coordinates, divide both gradients by d.
The same weight w therefore couples mean fitting and scale fitting. It is
not simply a switch that rejects independently identified label noise.
Large ν can still suppress examples when q is much larger than ν+d.
Small ν also **amplifies** low-q examples relative to Gaussian regression;
with d=56 and ν=7, the maximum w is 9 rather than approximately 1.

For nonzero residuals and a hypothetical independently adjustable σ for
each sample, the optimum is σ²=S/d. Substituting it back gives

    L_profile = (d/2) log S + constant(ν,d),
    ∂L_profile/∂μ = d r/S.

That mean gradient is independent of ν. This is a diagnostic limiting case,
not a claim that our finite shared network can freely optimize every σ.
It shows why frozen-residual density fit and end-to-end trainability are
different optimization questions.

At that scale optimum, the curvature in log σ, per coordinate, is

    ∂²(L/d)/∂(log σ)² = 2ν/(ν+d).

For d=56 it is 1.6 at ν=224 and 0.2222 at ν=7, **7.2× flatter**. Meanwhile
near r=0, mean curvature is proportional to (ν+d)/(ν σ²), which increases
as ν falls. A shared optimization recipe can therefore become differently
conditioned as ν changes. This does not prove a control-performance cause,
nor justify multiplying a learning rate by 7.2 without testing: Adam and
shared parameters complicate that intervention.

There is also an exact scale-score asymmetry:

    (∂L/∂log σ)/d = ν(d-q) / [d(ν+q)].

When scale is badly underestimated (q→∞), this tends to -ν/d; when q→0 it
tends to +1. Gradient descent's pressure to **increase** scale is therefore
capped at ν/d, while pressure to decrease scale can reach 1. At d=56, ν=7
the increase-side cap is 0.125, versus 4 at ν=224. This is a property of the
Student-t likelihood, not an implementation error. It can be appropriate
for genuine contamination, but a large residual caused by a still-wrong
mean is treated by the same rule. Distinguishing those cases is central to
the training-versus-density-fit gap.

### Why the ratio ν/d matters to optimization without redefining ν

With σ fixed, the Hessian along the residual direction has eigenvalue

    h_radial = (ν+d)(νσ²-S) / (νσ²+S)².

It is negative when q>ν. For a calibrated near-Gaussian residual vector,
q is typically of order d, so ν=4d leaves typical samples in the positive
radial-curvature region, whereas ν much smaller than d does not. A genuine
high-dimensional t with small ν can still be the statistically correct
model: this is an optimization distinction, NOT a new marginal-to-joint
degrees-of-freedom conversion. Per-example negative curvature also does
not imply that the expected loss or the full parameter Hessian is negative.

Measured with each current checkpoint's predictions and σ held fixed:

| Dataset | Current ν | Samples with q>current ν | Samples with q>7 |
|---|---:|---:|---:|
| Fractal | 224 | 1.71% | 99.97% |
| Bridge | 224 | 0.73% | 100% |
| GR1 | 928 | 1.91% | 100% |

These are counterfactual output-space curvature measurements, not completed
ν=7 training runs. Raw summary: `fixed_nu_gate_budget_v2.json`; reproducer:
`scripts/probe_fixed_nu_gate_budget.py`.

At the actual current ν, gate w<0.5 occurs in only 0.49%, 0.21%, and 1.04%
of these probe states respectively. Inverse-variance weighting and t gating
must therefore be distinguished when interpreting HT-versus-MSE gradient
plots. **Sample frequency is not gradient mass:** a small fraction of
outliers can still matter disproportionately. For example, the highest-q
1% of GR1 states contributes 2.30% of summed prediction-gradient norms under
Gaussian weighting with the same σ, versus 0.42% under t weighting. This
comparison does not measure actual parameter-gradient cancellation or a
separately trained HG policy.

The formula and gradients are checked by autograd in the math test. If
smooth continuation alone does not help, a more theory-aligned candidate
is a fixed-weight Gaussian majorization step, rather than another arbitrary
ν staircase. Concavity of log gives the tight upper bound at current q₀:

    L(θ) <= (w₀/2) q(θ) + d log σ(θ) + C₀,
    w₀ = (ν+d)/(ν+q₀).

This is the Student-t IRLS/MM surrogate. Its output-space mean penalty is
quadratic for fixed σ; its log-scale penalty has positive curvature for
fixed mean. Crucially, **one gradient step at the current parameters has
exactly the same gradient as the original loss**. Merely detaching w and
renaming the loss would NOT be an improvement. A meaningful test needs
multiple inner/block updates or a different curvature-aware optimizer,
with fixed data/dropout for the surrogate and a fair compute comparison.
This candidate is not yet implemented as a training run or claimed to work.

The derivatives are checked by autograd in `scripts/test_nu_debug_math.py`.
Related prior evidence that heteroscedastic likelihood training can have
optimization pitfalls: [Seitzer et al., ICLR 2022](https://arxiv.org/abs/2203.09168).
Their Gaussian results do not establish a remedy for this Student-t policy.

## Completed gradient audit

See `../nu_gradient_switch/RESULTS.md` and raw `global1024/` arrays.
At a 1,024-example diagnostic batch, same input/dropout/weights:

| Switch | Full parameter-gradient angle | Sigma-decoder angle | Sigma-gradient norm ratio |
|---|---:|---:|---:|
| 10k: 128→64 | 12.56° | 71.78° | 0.248 |
| 12k: 32→14 | 8.58° | 15.45° | 1.699 |

So the scale branch can change disproportionately at a staircase boundary,
but it is not always suppressed. These are **pre-Adam gradients**, not actual
optimizer update directions. Dropout itself produces substantial gradient
variation; angles alone do not explain a success-rate drop.

`analyze_trainmode_radial.py` reuses the raw 1,024-example arrays. The exact
radial likelihood in training mode also prefers ν=6.46 at 10k and ν=5.37 at
12k, so the discrepancy is not confined to eval-mode residuals. Its
`trainmode_radial.json` records the calibration/test split and likelihoods.

On those SAME predictions/scales, a hypothetical change from ν=224 to ν=7
changes the highest-q decile's share of summed **per-example prediction
gradient norms** from 13.4–14.1% to 6.0–6.4%. The lowest-q decile rises from
7.4–7.7% to 14.7–15.4%. This is evidence of redistribution, not a measurement
of how much those groups contribute to an Adam update; parameter Jacobians
and gradient cancellation are absent from this statistic. High-q examples
are not labeled as noise or as valuable control actions by this probe.
Grouping that same batch by instruction gives 230 move-near, 79 close-drawer,
and 21 pick-coke episodes. Their aggregate prediction-norm shares change
little across ν=224,14,7 (move-near stays about 23%, close-drawer about 8%).
Thus the measured redistribution is NOT evidence that an entire one of these
task families is being disproportionately rejected. A particular phase could
still be affected; that requires finer, independently labeled analysis.

## Bounded continuation experiment — submitted

All three arms start from the same full 7k checkpoint:
`/mnt/pfs/yuchen/groot/ft_fr_nu1024_hold7k_to14_20260908/checkpoint-7000`.
This includes the Adam shards, scheduler, model, trainer state, and RNG files.

| Arm | Schedule after 7k | Training endpoint |
|---|---|---:|
| `stair14` | Replay 512, 256, 128, 64, 32, 14 in successive 1k blocks | 13k |
| `smooth14` | Log-linear 1024→14 from 7k to 12k, then hold | 13k |
| `smooth7` | Log-linear 1024→7 from 7k to 12k, then hold | 13k |

This compares complete recipes; log-linear and staircase schedules do not
have identical cumulative ν exposure, so an improvement would not isolate
discontinuity as the sole cause. A later matched-exposure ablation may be
needed. These are single-seed screening runs, not a final multi-seed ranking.

Training remains 8 GPUs, global batch 1024, original normalization,
augmentation, state dropout, mvt reduction, and loss implementation. Keep
the original 18k cosine LR horizon and 1k warmup; stop by callback at 13k,
so a shorter experiment does not silently alter the LR curve.

The repository's Trainer resets its iterable data stream using base seed
plus restored step. Hence a restarted run is not an exact replay of the
uninterrupted historical run; the resumed `stair14` control is essential.
Record rank-0 data and frozen-backbone feature fingerprints to check matching
between the new arms. Fail before any update if restored step, Adam counts,
LR, batch size, or source code differ from the specified protocol.

Resource cap: three 8-GPU training pods, each 6k additional updates, then
evaluation; a separate 3-GPU paired baseline evaluation. No existing jobs
were stopped. The training pods initially queue for a free full node.
No second-stage recipe search is automatically launched.

Launchers in `scripts/cluster/`:
`nu_recipe_fork_launch.py`, `nu_recipe_fork_train.sh`,
`nu_recipe_fork_eval.py`, `nu_recipe_submit.py`.
Copies on cluster: `/mnt/pfs/yuchen/nu_recipe_debug_20260908/code/`.
Runs: `/mnt/pfs/yuchen/groot/ft_fr_nudebug_{arm}_20260908/`.

## Evaluation and next decision

- At 13k, each arm automatically evaluates all six Fractal tasks with 100
  episodes, five environments, NAS=1, max 300 steps.
- Preserve the user's **unseed** reference protocol. Separately evaluate
  fixed per-episode seeds for paired diagnostics; never combine score columns.
- Independent legacy-seeded audit: existing 18k ν=14 policy versus fixed-ν224
  20k on coke, move-near, close-drawer. This aligns the evaluation protocol,
  not individual scenes; see the collector correction below. Strict fixed-
  scene coke/move-near checks are additional diagnostics. The models' different
  training histories mean neither comparison is a schedule-only ablation.
- Save per-episode outcomes, task/protocol/seed/checkpoint identifiers.
  Repeated unseed initial scenes can make 100 rollouts fewer than 100
  independent scene draws; do not quote naive binomial significance.
- Compare six-task macro success and individual regressions. A close-drawer
  improvement bought by collapsing coke or move-near is not a general win.
- Cross-check σ, q/d, gate distributions, and resumed data fingerprints.
- If smooth low ν still fails, inspect mean/scale optimization separately
  and actual Adam update directions before expanding ν sweeps. Any proposed
  scale preconditioner must be tested against the same objective and control.
- A promising candidate needs Bridge confirmation and independent seeds / a
  new evaluation scene set before recommending it as the best recipe.

Status and evidence should be updated here as runs finish. Lower held-out
density NLL is not a substitute for improved closed-loop success.

### Evaluation results and collector audit

The existing 18k ν=14 policy scored **97/100 move-near** and **84/100 coke** with seed 1234 in
the new diagnostic evaluator, versus its previously recorded **80/100
and 95/100 unseeded** respectively. These are different initial-scene sets, NOT a treatment effect.
Close-drawer under the seeded reproduction protocol is still pending.
This already prevents treating the unseeded-versus-seeded difference as
evidence that annealing intrinsically caused a large move-near regression.

**Protocol audit correction:** the existing seeded collector seeds only the
first reset of five asynchronous environments. Subsequent auto-resets and
completion order need not produce the same 100 scenes for different policies.
Accordingly, existing `*-nudebug-1234` files initially labeled
`paired_diagnostic` must be interpreted as **same-seed protocol reproductions,
not verified scene pairs**. Do not pair their outcome arrays by index. A
separate fixed-scene collector has been added for the new training branches;
it records episode seeds and initial-observation hashes and preserves the
unseeded reference collector unchanged.

The fixed-scene collector is implemented in
`scripts/cluster/nu_fixed_scenes_rollout.py` and unit-tested with an auto-reset
mock whose episode lengths change by policy. Each five-scene batch receives
explicit seeds, all five initial episodes finish before the next batch,
and auto-reset episodes from already-finished slots are ignored. Results
are in predetermined seed order, not completion order. New continuation
runs use this diagnostic alongside the unchanged unseeded reference.
`*-nudebug-fixed1234/` is distinct from legacy `*-nudebug-1234/`.

Fresh legacy seeded baseline reproductions are now **94/100 coke and
94/100 move-near** for fixed ν=224. Thus under that shared evaluation
protocol the annealed 18k model has coke 84 versus 94, and move-near 97
versus 94. This is not evidence of a broad across-task collapse. Strict
fixed-scene checks of both models on these two tasks are running on the
two idle GPUs of the existing diagnostic pod; no additional GPUs reserved.

### Launch recovery / provenance

The initial smooth14 and stair14 launches failed **before any optimizer
updates** because Transformers' tokenizer initialization attempted an HTTP
model-info lookup despite offline mode. The process-local wrapper now
resolves the identical cached processor snapshot by filesystem path, as
the evaluation server already did. Model/trainer/loss source is unchanged.
No failed-output directory was overwritten or removed.

- `smooth7`: original pod/output, launched with the cached-processor fix.
- `smooth14`: replacement `yuchen-fr-nudebug-smooth14-0908-r2`, output
  `/mnt/pfs/yuchen/groot/ft_fr_nudebug_smooth14_20260908_r2`.
- `stair14`: replacement `yuchen-fr-nudebug-stair14-0908-r2`, output
  `/mnt/pfs/yuchen/groot/ft_fr_nudebug_stair14_20260908_r2`.

The first stair14 process had already imported its wrapper when the cache
fix was copied into its audit directory. Its actual pre-fix source is
therefore separately preserved as `nu_recipe_fork_launch.executed_before_cache_fix.py`.
Do not treat the later audit copy as the source executed by that failed attempt.
All completed-step counts must come from `resume_verified.json` and training
logs, not merely a Kubernetes Running status.

The cached-processor fix has now passed model loading and the smooth7
branch's `resume_verified.json` confirms step 7000, restored Adam state
(all six inspected state entries at 7000), and original LR
7.228691778882693e-5 in both groups. Actual updates have passed **step 7030**,
and the first forward diagnostic passed the exact native-loss reconstruction
assertion. Initial startup records are copied locally into `startup/`.
The two replacement branches remain queued for a free full node.

Strict fixed-scene checks are still incomplete; the first ten shared initial
observations matched byte-for-byte. This is a collector verification, not
a success-rate conclusion from an incomplete evaluation. The comparator
`scripts/compare_fixed_scene_evals.py` rejects incomplete evaluations and
observation mismatches before producing paired statistical summaries.

### Fixed-seed coke completion; pairing check not fully passed

Both coke evaluations with predetermined seeds 1234–1333 completed:
annealed ν=14 at 18k is **81/100**, fixed ν=224 at 20k is **91/100**.
The initial-observation hashes match in **91/100** cases. Mismatch seeds:
1249, 1260, 1275, 1282, 1293, 1308, 1312, 1314, 1327.
The comparator correctly refuses paired statistical inference. Need to
distinguish renderer/floating-point differences from actual scene-state
differences; do not silently discard these nine or assume identical inputs.
Move-near under this fixed-seed collector is still running. These results
also compare 18k versus 20k training histories, not an isolated ν treatment.
