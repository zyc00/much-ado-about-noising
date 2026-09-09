# Fractal HT nu=224 with exponential scale

Launched 2026-09-09 as an isolated parameterization ablation against
`/mnt/pfs/yuchen/groot/ft_fr_nu224`.

- Pod: `yuchen-fr-nu224-exp-0909`
- Output: `/mnt/pfs/yuchen/groot/ft_fr_nu224_exp_20260909`
- Objective: original per-dimension joint Student-t NLL, fixed `nu=224`,
  effective chunk dimension `d=56`.
- Only intended training change: replace the native scale
  `mean(softplus(raw - 0.5093)) + 0.001` by
  `exp(mean(raw) + log(0.4715760109))`.
- No Gaussian auxiliary, nu schedule, learned nu, clamp, or gripper change.
- The original seeded sigma-decoder initialization is retained. No decoder
  weights are zeroed. At raw zero, both parameterizations give exactly
  `sigma=0.4715760109`. On the first real batch, exp/softplus sigma quantiles
  were `[1.00119, 1.00195, 1.00292]`, so the initial scales match within 0.3%.
- All other training settings match the original Fractal recipe: original
  pretrained GR00T, fresh optimizer, seed/data seed 42, 20k updates, AdamW,
  LR 1e-4, WD 1e-5, 1k warmup, global batch 1024, state dropout 0.5.
- Automatic endpoint evaluation matches the result reported for the original
  nu=224 arm: seed 1234, all six Google Robot tasks, 100 rollouts/task,
  5 vector environments, NAS=1, and max 300 environment steps.

Exact source snapshots and the reference configuration are stored under the
run's `audit/` directory. `initialization_verified.json` records the first
real-batch comparison; `training_diagnostics.jsonl` records scale and gradient
statistics every 100 updates. Deterministic inference uses only the mean and
does not require the custom scale wrapper.
