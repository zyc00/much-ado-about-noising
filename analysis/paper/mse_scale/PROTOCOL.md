# Frozen-MSE residual-scale probe

## Question and planned visual

Does a well-trained direct MSE predictor have the same continuous-action
residual scale throughout a demonstration? A compact, 5.5-inch vector figure
will show measured scale by demonstration progress for GR00T/GR1, pi0.5/LIBERO,
and human RoboMimic. Progress bins are a stage proxy, not semantic annotations.
We do not assume discrete scale clusters or Gaussian residuals.

## Pre-specified measurement

- Frozen final MSE checkpoint and a nearby late checkpoint; deterministic
  inference on identical demonstration observations, with augmentation off.
- GR1: all 24 tasks, 12 randomly selected episodes per task, one random valid
  observation in each of ten equal episode-progress bins (seed 20260904).
- LIBERO: all available tasks, 12 randomly selected episodes per task, the
  same ten-bin sampling. RoboMimic: all human episodes in Tool-Hang and
  Transport-ph, two valid observations per bin.
- Original fixed training normalization. GR1 uses the checkpoint's saved
  statistics, never a subset-recomputed normalizer. RoboMimic reconstructs
  the original full-dataset normalizer.
- Exclude binary grippers (LIBERO index 6; each RoboMimic arm index 9 after
  rot6d conversion). Retain all 29 GR1 joint channels, including hand joints.
- Primary residual is over the executed action chunk: GR1 8 steps, pi0.5
  10 steps, RoboMimic 8 steps starting at the first executed index 1.
  Only observations with a complete executed chunk are eligible. Store
  native full-chunk predictions and masks for single-step/full-chunk checks.
- Scale is RMS of normalized residuals, not the standard deviation of RMS.
  Keep task identity: differences between tasks must not masquerade as
  within-task stage variation. Also report coordinate-centered residual
  spread and residual-mean contribution within each task/progress cell.
- Randomly partition sampled episodes within each task into two halves to
  test whether stage-scale patterns replicate. These halves are held out
  from one another for scale estimation, NOT held out from policy training.
- Bootstrap whole episodes, not individual coordinates or overlapping
  chunks. Report all sampled tasks; choose no examples by outcome.

## Scope

The inspected recipes trained on all demonstrations (no policy-held-out
split). This is a training-data fit diagnostic, not an estimate of
irreducible conditional noise. Unequal residual scales motivate an
input-dependent loss scale; they do not alone establish heavy tails,
Student-t optimality, or a causal explanation of control performance.

## Checkpoint audit

The Tool-Hang `t12_..._s1_l2` checkpoint exists but its log stops at 23k;
exclude it. Use `snap_hmse_chi` at 299999 and 287999 instead. GR1 uses 60k
and 58k, pi0.5 uses 30k and 25k, Transport-ph uses 300k and 280k.

## Exploratory follow-ups (not pre-specified)

After the coarse progress contrast was found to be modest on RoboMimic:

1. Predict residual energy from proprioceptive nearest neighbors in a
   disjoint episode half, within each task. Select k by the fixed rule
   `min(30, ceil(sqrt(n_calibration)))`, with a minimum of five. Standardize
   state coordinates using calibration episodes only. Group query states
   into three equally populated bins by their predicted scale; query
   residuals do not enter prediction or grouping. Repeat with halves swapped.
2. Following the user's movement-size intuition, group observations by
   predicted physical movement magnitude (five within-task quantiles).
   Undo the original action normalization first. Use translation commands
   for LIBERO/RoboMimic, and relative arm/hand joint offsets for GR1.
   Absolute waist commands are excluded from the GR1 magnitude metric only;
   all 29 joint channels remain in its primary residual metric.
3. Retain all positive, weak, null, and reversed contrasts. These exploratory
   checks are not substituted for the original progress-bin result.
