# Why MSE fails on human ToolHang data — comprehensive MSE-vs-MIP anatomy

All success rates from the official train-harness eval (mode=eval, 100 episodes, model_latest,
seed-matched) unless marked otherwise. Historical best-checkpoint pair in parentheses.
Rollout-level numbers from 20-episode instrumented dumps per model (standalone harness,
seeds 31000–31019; used for anatomy, not SR claims).

## 0. The headline numbers

| | human MSE | human MIP |
|---|---|---|
| SR, official protocol | 33 / 37 (two seeds) | 60 |
| SR, historical best-checkpoint | 52 / 60 | 82 / 80 |
| rollout dumps (20 eps, standalone) | 3/20 | 12/20 |

## 1. Training set: MSE fits the noisy labels BETTER — and that is the disease
> **RETRACTION (2026-07-28): the residual table in SS1 below does not reproduce on current checkpoints** (probe_recon, n=92,562: hMSE slot-0 RMS 0.007 vs hMIP 0.003 — hMSE 3x LOOSER, sign-opposite; prior 0.227 inconsistent with hbase2's own logged train loss 4.7e-5 by ~1000x). Likely pre-fix chunk-slot misalignment or an older checkpoint generation. See positive_validation_report.md PART CCCXCIX.


Teacher-forced first-slot residual (normalized 10-d action space), 47,833 pairs:

| | overall RMS | median | align1 window | residual/floor at align1 |
|---|---|---|---|---|
| hMSE | **0.227** | **0.056** | 0.316 | 0.71 |
| hMIP (step1 / 2-step) | 0.248 / 0.236 | 0.107 | 0.326 / 0.313 | 0.74 / 0.71 |
| kNN aleatoric floor | — | — | 0.443 | 1.00 |

- MSE tracks the training labels better than MIP everywhere (median residual 2x lower).
  The labels carry heavy-tailed corrective tremor (sigma~0.2, df~2): tracking them closely
  means partially FITTING THE NOISE. MIP's deployed function is smoother — it pays
  teacher-forced residual against noisy labels for a cleaner servo. Fit quality is
  anti-correlated with SR on human data (also: lower-capacity MLP-MSE scores 92 vs
  chiunet-MSE 57 — capacity itself acts as a noise filter).
- The align1 window (insert-align hover — where the failures happen) is objectively the
  least-learnable segment: residual-to-floor ratio 0.71–0.74 vs 0.31–0.47 in every other
  window, and the noisiest windows by the floor itself are settle/align (0.81–0.86).

## 2. On-support: the failure is an incoherent, attenuated servo — quantified against GT

Align-hover behavior metrics (median over segments; flip = fraction of consecutive commanded
xyz actions reversing direction; ac1 = lag-1 autocorrelation; mag = median |command|):

| segment | flip | ac1 | mag |
|---|---|---|---|
| GT demos, align1 hover | **0.000** | **0.909** | 0.279 |
| hMSE failed episodes (dither window) | **0.176** | **0.492** | 0.173 |
| hMIP failed episodes | 0.147 | 0.499 | 0.085 |
| hMSE / hMIP successful episodes | 0.10 / 0.03 | 0.87 / 0.87 | 0.14 / 0.17 |

- Human demonstrators never reverse direction in the hover (flip 0.000) — their fine
  corrections are coherent. Failed policy episodes reverse every ~6 steps with HALF the
  temporal coherence and commands attenuated ~40% below GT: an under-gained, sign-flipping
  servo that makes no progress until the clock expires (timeouts).
- Both objectives fail THIS WAY when they fail; MSE enters the mode 2x as often. The
  attenuation is the L2 signature: averaging heavy-tailed corrections shrinks commanded
  gains; the anchor absorbs the noise instead and preserves servo structure.

## 3. Off-support: MSE also has an escape tail on human data (a secondary mode)

Tube distance to a 40-demo anchor tree (z-scored obs), 20 eps/model:

| | maxd p50 | maxd p90 | frac steps d>=2 | d>=4 |
|---|---|---|---|---|
| hMSE | 5.0σ | **145σ** | 26.1% | **17.7%** |
| hMIP | 3.1σ | 6.8σ | 19.7% | 6.6% |

- 2–3 of 20 hMSE episodes are runaways (maxd 276–326σ; e.g., seed 31010 assembles the frame
  then escapes); hMIP's worst episode is 6.8σ. The majority failure mode remains the mild-
  excursion timeout (median episode maxd 5σ).
- The differential response field (Rdiff at 10–40mm kinematic offsets) is only WEAKLY
  restoring for BOTH models (frac>0 53–65%, mixed ordering) — the tube gap is not a simple
  field-strength story; it compounds from servo coherence + noise-fitting jitter.

## 4. Representation: no geometry disease — and the advantage still lives in the trunk

- No collapse: both encoders flat (kappa10 2.0–3.3 in every window; no height monoculture,
  column cosines 0.21–0.31 vs scripted MSE's 0.89). The scripted fold/starvation disease is
  ABSENT (human tremor supplies the anti-collapse label richness; dose curve: clean 1.90 ->
  wpmatch 4.12 -> human 5.6–5.9 settle covariance PR).
- No fusion: 9x9 cross-window NN confusion matrices are block-diagonal for both models —
  all label-similar cross-leg pairs (carry1<->carry2, align1<->align2, settle1<->settle2)
  at 0%. Executed dither actions align only weakly with ANY neighbor-label mean (own +0.41 /
  cross +0.34): failure is precision, not misassignment.
- Yet MIP's embedding is ~2x higher-dimensional (covariance PR 14 vs 7–8, both seeds) —
  the fingerprint of the anchor having fit fine structure the L2 averaged away.

## 5. The interventions decide the mechanism (the strongest evidence)

| arm | SR | what it tests |
|---|---|---|
| human MSE | 33 / 37 | baseline |
| hprogaux (progress-ramp aux) | 40 | geometry/richness channel: ~inert on human data |
| hheadmse (frozen hMIP trunk + fresh L2 head on noisy labels) | **57** | representation transfer: ~full MIP advantage |
| human MIP | 60 | both channels |
| hpds (PD/recovery prior), randaug ports | pending | field-structure / removable-richness channels |

- hheadmse is the decisive cell: a plain, noisily-supervised L2 head on the anchor-shaped
  trunk recovers 57 of MIP's 60 (vs 33/37 from-scratch). The anchor's noise absorption acts
  DURING TRUNK FORMATION; once the trunk encodes the fine align-servo structure, ordinary
  regression reads it out despite the label noise. (Scripted analog: headmse 96 ~ MIP 95.)
- hprogaux at 40 shows generic dense trunk-shaping does NOT build that structure — the
  content of what the anchor teaches (state-conditional fine corrections beneath the noise),
  not trunk supervision per se, is the ingredient.

## 6. The mechanism, stated once

Human demonstrations supervise the task through heavy-tailed corrective noise concentrated
in the slow alignment hovers (aleatoric floors 0.81–0.86 there vs 0.32–0.44 elsewhere).
Plain L2 must fit the conditional mean THROUGH that noise: the gradient signal for the fine
align servo has the worst SNR exactly where precision matters most, so the learned servo
comes out attenuated (~40% low) and locally incoherent off the demonstrated groove (ac1 0.49
vs GT 0.91) — the policy hovers, reverses, and times out; occasionally the jitter compounds
into a full escape (17.7% of steps beyond 4 tube-sigma, worst episodes 300+ sigma). Nothing
is wrong with WHERE states map (no collapse, no fusion, faithful spectra): what is missing
is the fine-scale CONTENT of the action field. MIP fixes precisely that channel: the anchor
input absorbs the label noise during training, so the trunk learns the state-predictable
fine structure beneath it (2x embedding dimension, smoother deployed function that fits the
noisy labels WORSE), and the same noisy-label L2 readout on top of that trunk recovers
nearly the entire gap (57 vs 60). Scripted and human failures are therefore two diseases of
one objective: on clean scripted data L2 starves REPRESENTATION geometry in slow windows
(fold -> wrong-phase actions); on noisy human data L2 corrupts ACTION-FIELD content in slow
windows (attenuated dither -> timeouts). The denoising anchor supplies the missing gradient
channel in both.

## 7. Videos (analysis/traj_vis/videos/, seeds match the instrumented dumps)

- hMSE seed 31003 (timeout): in-support dither — hovers at the insertion, reverses, never
  assembles; hMIP at the SAME seed completes the task.
- hMSE seed 31010 (escape): assembles the frame, then the servo jitter compounds into a
  runaway (maxd 326 sigma); hMIP same seed: success.
- hMSE seed 31002: the rare MSE success (control — when the servo stays coherent it finishes).
- hMIP seed 31004: MIP's own failure mode (also a dither timeout — same phenotype, rarer).
