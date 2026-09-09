# Continuous-action Gaussian-component audit

## Bottom line

After contact-state channels are separated, the strongest available evidence
supports a **single continuous action component**, not multiple manipulation
strategies.  This does not mean that every scalar marginal is exactly normal.
It means that adding a second Gaussian generally hurts held-out prediction of
the continuous action cloud, including on states deliberately selected to look
maximally multimodal.

The GR1 channel ablation is especially diagnostic.  The arm-and-waist output
is predictively single-component, while the articulated hands and the full
29-channel output are not.  Thus the apparent full-action modes are localized
to open/close hand configurations rather than alternative body trajectories.

## Protocol

For a fixed observation, the stochastic policy is sampled repeatedly at its
exact evaluation-time inference setting.  We compare a one-Gaussian density,
a Student-t density, and a two-Gaussian mixture by held-out log likelihood.
The test is performed on fixed random projections and on the strongest PC1
direction learned from samples disjoint from all evaluated samples.  This
prevents the two-Gaussian model from winning merely because it has more
parameters or because the same samples selected and evaluated the axis.

- WidowX: 100 uniformly sampled on-policy states, 32 discovery and 64
  confirmation draws per state; first four executed actions; six arm channels.
- GR1: 30 adversarial states retained from a 3,000-state full-action screen,
  with 32 fresh draws per state.  The same states and draws are evaluated as
  arm+waist, articulated hands, and all joints.
- pi0.5: 36 adversarial states retained from a 3,000-state screen, with 32
  fresh draws per state; six arm channels, excluding gripper.

## Results

`2G > 1G` counts states whose state-averaged held-out log likelihood favors a
two-Gaussian mixture.  `delta LL` is two-Gaussian minus one-Gaussian held-out
log likelihood in nats per projected draw; negative favors one Gaussian.  The
last column reports BH-corrected Shapiro-Wilk rejection along the independently
discovered PC1.  It tests exact normality, which is stricter than asking whether
a second component is useful.

| policy / tested subspace | states | 2G > 1G, fixed axes | 2G > 1G, independent PC1 | mean delta LL, independent PC1 | exact-normality rejects |
|---|---:|---:|---:|---:|---:|
| WidowX arm | 100 uniform | 31/100 | 47/100 | +0.015 | 45/100 |
| GR1 arm + waist | 30 adversarial | **1/30** | **1/30** | **-0.478** | **1/30** |
| GR1 articulated hands | 30 adversarial | 22/30 | 16/30 | +33.568 | 26/30 |
| GR1 all 29 joints | 30 adversarial | 20/30 | 16/30 | +13.017 | 26/30 |
| pi0.5 arm | 36 adversarial | **0/36** | **4/36** | **-0.772** | **3/36** |

The extreme positive mean log-likelihood gains for the GR1 hand/full rows are
caused by very narrow, repeatedly sampled hand configurations.  In the most
separated case, one finger channel lies near -0.46 or +0.14 while the other
finger joints switch coherently.  Across the 15 confirmed hard states, 348 of
480 chunks hold one hand configuration for the full horizon, 118 transition
once, and only 14 flicker.  These are discrete-like contact-state events
encoded as multiple joint targets.

WidowX is not exactly Gaussian everywhere, but its departure is localized and
interpretable.  All 10 previously detected executed-prefix arm splits favor
2G on held-out samples (median delta LL +0.170); the remaining 90 states favor
1G on average (mean -0.004, median -0.018).  Only six of the 100 have two
resolved arm-density peaks, and exact-state closed-loop forks show that their
identity disappears after replanning.  Thus they are sparse maneuver-phase
splits rather than persistent route alternatives.

## Paper-safe claim

> After separately accounting for discrete contact-state outputs, the
> continuous body-action distributions of GR1 and pi0.5 are predictively
> single-component even on adversarially selected states.  WidowX contains
> sparse local arm-phase splits, but these do not persist as alternative
> closed-loop strategies.  The evidence therefore disfavors multimodal route
> averaging as the primary explanation for MSE's performance gap.

Do **not** claim that all action marginals are exactly Gaussian.  Exact
normality is rejected in many WidowX states because of skew and the sparse
phase cases.  "Single Gaussian component" or "single continuous basin" is the
supported wording.

## Artifacts

- Analysis: `analyze_action_basin.py`
- Machine-readable results: `action_basin_report.json`
- Diagnostic figure: `action_basin_heldout.png` / `.pdf`
- GR1 raw extreme-state samples: `gr1_seek3k.npz`
- pi0.5 raw extreme-state samples: `pi05_seek3k.npz`
