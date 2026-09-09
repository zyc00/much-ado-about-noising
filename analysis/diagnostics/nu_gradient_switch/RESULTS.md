# Fractal nu-switch gradient results

Completed the six actual switch-point audits (7k–12k) and a 1024-sample
mixed-data follow-up at 10k and 12k. No optimizer steps were taken.

## Main result: 1024 samples

Each comparison uses the same checkpoint, processed inputs, and dropout
realization. Gradients cover all 1,658,432,768 trainable parameters, accumulated
in float32. The model backbone is frozen. Angles below concern raw gradients,
not Adam-preconditioned updates. The 1024-sample batch matches the training
global sample count, but is independently selected, not the original batch.

| Checkpoint | Nu change | All-parameter cosine | All-parameter angle | All-gradient norm ratio | Action-decoder angle | Sigma-decoder angle | Sigma-gradient norm ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| 10k | 128 → 64 | 0.97605 | 12.56° | 0.935 | 13.18° | 71.78° | 0.248 |
| 12k | 32 → 14 | 0.98880 | 8.58° | 1.021 | 12.19° | 15.45° | 1.699 |

The overall gradient does not reverse or become approximately orthogonal.
However, changing nu can substantially alter the sigma decoder's update:
at 10k it rotates and shrinks; at 12k its norm increases instead. This is
evidence of a coupled mean/scale reweighting effect, not evidence that every
nu reduction uniformly suppresses the sigma gradient.

Absolute sigma-gradient norms are 0.10044 → 0.02496 at 10k and
0.03026 → 0.05140 at 12k. A large angle or relative change in a small branch
must not be equated with a large perturbation of the entire model.

### Smaller nu changes

| Checkpoint | Nu change | All-gradient angle | Sigma-gradient angle | Sigma-gradient norm ratio |
|---|---|---:|---:|---:|
| 10k | 128 → 115.2 | 2.04° | 3.09° | 0.837 |
| 12k | 32 → 28.8 | 1.43° | 5.20° | 1.195 |

A 10% reduction produces much smaller instantaneous direction changes.
This supports testing continuous annealing as a controlled follow-up, but
does not establish that continuous annealing will improve rollout success
or that the final low-nu objective is appropriate.

### Same-nu dropout control

At 10k, changing only dropout gives an all-parameter angle of 54.72° and
a sigma-decoder angle of 8.70° (sigma norm ratio 1.134).
At 12k, the corresponding values are 60.78°, 23.02°, and 1.167.
Overall dropout variability is larger than the nu-switch angle on these
batches. Systematic objective changes are not equivalent to stochastic noise,
so this is context, not a proof that the switches are harmless.

## Target-family quick check: 16 samples per family

| Checkpoint | Nu change | Move-near all-gradient angle | Close-drawer all-gradient angle |
|---|---|---:|---:|
| 7k | 1024 → 512 | 1.56° | 1.41° |
| 8k | 512 → 256 | 2.92° | 2.38° |
| 9k | 256 → 128 | 3.77° | 2.72° |
| 10k | 128 → 64 | 3.24° | 5.24° |
| 11k | 64 → 32 | 6.67° | 4.94° |
| 12k | 32 → 14 | 5.52° | 4.38° |

These one-batch-per-family checks do not show violent direction changes,
but are not population-level estimates or simulator-success measurements.

## Why the sigma branch can behave differently

Write Q_i = ||r_i||² / sigma_i² and d = 56. For the joint Student-t loss,

    dL_i / d mean_i = (nu+d)/(nu+Q_i) * r_i/sigma_i²
    dL_i / d log(sigma_i) = nu/(nu+Q_i) * (d-Q_i)

The first expression always points along the sample's residual. The second
has opposite signs for under- and over-scaled samples. Changing nu changes
their relative weights. Consequently, even though each sample's log-scale
derivative decreases in magnitude as nu falls (at fixed residual and sigma),
the *summed parameter gradient* can increase when cancellation is reduced.
This explains why sigma-gradient norms need not decrease monotonically.

An optional schedule parameterization is lambda = d/(nu+d), for which
the mean-gradient gate is 1 / [1 + lambda * (Q_i/d - 1)]. The current
nu sequence corresponds to lambda approximately
0.052, 0.099, 0.179, 0.304, 0.467, 0.636, 0.800. Equal halvings of nu are
not equal increments in this robustness parameter. Smooth log-nu or lambda
annealing is a proposed follow-up, not an evaluated replacement recipe.

## Interpretation and next controlled test

The measurements do **not** establish a pathological training recipe or
causally explain move-near's success-rate difference. They identify a concrete
coupling to monitor: nu changes the balance of mean and sigma updates, with
particularly large sigma-branch changes at one measured transition.

To isolate switch abruptness, start paired branches from the same checkpoint
and compare staircase versus continuous annealing, holding the endpoint,
training budget, optimizer state, and learning-rate schedule fixed. Evaluate
on matched initial scenes. If smoothing alone fails, investigate the target
nu/long-term sample weighting separately instead of assuming smaller jumps
solve the problem. No new policy training has been launched by this audit.

## Reproduction and artifacts

- Protocols, complete groupwise comparisons and per-sample diagnostics:
  `small_batch/` and `global1024/`.
- Cluster originals: `/mnt/pfs/yuchen/fractal_nu_gradient_20260908/`.
- Source: `scripts/probe_nu_gradient_switch.py`.
- Summary command:
  `python analysis/diagnostics/summarize_nu_gradient_switch.py analysis/diagnostics/nu_gradient_switch/global1024`.
- Local transfer manifests contain SHA256 hashes for the cluster originals.
  Full parameter-gradient vectors are computed but not retained on disk.
