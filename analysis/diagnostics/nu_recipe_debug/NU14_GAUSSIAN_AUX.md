# Fresh Fractal: nu=14 plus detached-mean Gaussian scale loss

User requested cancellation of the previous experiment fleet and a fresh
nu=14 training run with a Gaussian auxiliary to balance scale learning.

## Stopped

Deleted the following pod resources (training/evaluation processes stopped;
PFS checkpoints, diagnostics and completed/partial evaluation files retained):

- yuchen-fr-nudebug-smooth7-0908
- yuchen-fr-nudebug-smooth14-0908-r2
- yuchen-fr-nudebug-stair14-0908-r2
- yuchen-fr-nudebug-paired-0908

Those continuation/evaluation plans are canceled, not still pending.

## New experiment

- Pod: `yuchen-fr-nu14-gaussaux1-0908` (8 GPUs).
- Output: `/mnt/pfs/yuchen/groot/ft_fr_nu14_gaussaux1_fresh_20260908`.
- Fresh fine-tuning from `/mnt/pfs/yuchen/groot/model`, original pretrained
  GR00T N1.7; no old HT checkpoint or optimizer state. NOT fully random
  initialization of the pretrained model.
- Full Fractal: 87,212 episodes, original Google Robot embodiment/action mask.
- Fixed joint Student-t nu=14 from the first update. All training channels
  including gripper; effective chunk dimension d=56.
- Objective: `L_HT(mu,sigma;14) + L_HG(stop_gradient(mu),sigma)`.
  Gaussian coefficient lambda=1 is an explicitly announced first-trial
  choice, not an empirically tuned optimum or online norm equalization.
- Both terms use the SAME predicted sigma, masked-mean softplus(raw+sbias)
  plus 0.001. Both are normalized by total valid action dimensions.
- Shared parameters still receive auxiliary gradients through the sigma
  path. Detaching mu does NOT freeze future mean predictions under shared
  parameter updates.
- Optimizer: existing AdamW protocol, LR 1e-4, WD 1e-5, global batch 1024,
  8 GPUs with per-device batch128 and accumulation1, state dropout0.5,
  sbias=-0.5093. Frozen pretrained visual/LLM backbone, existing action-head
  fine-tuning convention. Warmup1000, total20000, saveevery1000 (keep20).
- Automatic final evaluation: original unseeded protocol, six Fractal tasks,
  100rollouts/task, 5envs, action execution1, max300steps.
- Process-local forward loss hook only; shared source model/trainer untouched.
  Inference uses the ordinary deterministic mean, with no auxiliary computation.
  Future training resumption REQUIRES this auxiliary hook; vanilla training
  would silently omit the auxiliary despite inference-compatible checkpoints.

## Validation and observability

`scripts/cluster/test_nu14_gaussian_aux.py` passed:

- no auxiliary gradient to predicted mean or mean-only head parameters;
- nonzero gradient to sigma and shared features;
- correct scalar-scale gradient and masked dimensions;
- combined direct mean gradient equals HT-only direct mean gradient.

Startup verified in the real 8-GPU run: first 10 optimizer updates completed;
logged training loss 0.0208, raw gradient norm 0.6297, LR 9e-7 (warmup).
At rank0 step0, native HT loss=0.274247, Gaussian auxiliary=-0.215810,
combined loss=0.058436; native reconstruction assertion passed. These loss
values omit density constants and can be negative. Sigma-gradient mean
absolute magnitude was 0.03932 (HT only) and 0.25671 (combined) on that batch.

Runtime asserts fresh step0, fixednu14, native joint loss branch and normalization,
AdamW/batch/LR conventions. `fresh_start_verified.json` records startup.
`training_diagnostics.jsonl` logs HT/Gaussian/total losses separately, residual RMS,
sigma, standardized squared residual, and both scale-gradient components every
100steps (rank0 batch). No online reweighting or additional optimizer changes.

Launch source: `scripts/cluster/nu14_gaussian_aux.yaml` and companion
`nu14_gaussian_aux_train.sh`, `nu14_gaussian_aux_launch.py`. Exact source snapshots
and reference config are copied to the run's `audit/` directory at startup.
