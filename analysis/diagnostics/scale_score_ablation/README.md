# Fractal: score × scale-parameterization ablation (2026-09-08)

User request: queue (1) new score + exp, (2) new score + softplus,
(3) old Student-t NLL + exp + detached-mean Gaussian auxiliary.

## Recipes

| Arm | Objective | Scale | Pod |
|---|---|---|---|
| se_exp | Scaled energy, joint t nu=14, K=16 | exp(mean raw + log sigma0) | yuchen-fr-se-exp-0908 |
| se_softplus | Same scaled energy and MC RNG | mean softplus(raw - 0.5093) + 0.001 | yuchen-fr-se-softplus-0908 |
| ht_exp_gaussaux1 | Per-dim joint-t NLL + HG(stopgrad mu, sigma), lambda=1 | exp(mean raw + log sigma0) | yuchen-fr-ht-exp-gaussaux-0908 |

All means above use the original action mask, d=56 (8-step chunk, 7 channels).
Training includes the gripper, matching the existing training convention.
All three are fresh fine-tunes from original pretrained GR00T, not old HT
checkpoints and not randomly initialized entire VLAs. Same AdamW, LR=1e-4,
WD=1e-5, 1000 warmup steps, 20000 total steps, global batch=1024, 8 GPUs,
state dropout=0.5, backbone freezing/tuning and data sampling unchanged.
Save every 1000 steps, retain all 20. Automatically evaluate 20k using existing
unseeded SIMPLER protocol: six Fractal tasks, 100 rollouts/task, 5 envs,
NAS=1, max episode steps=300. No claim of paired initial scenes.

## Initialization

sigma0 = softplus(-0.5093) + 0.001 = 0.4715760109.
log(sigma0) = -0.7516749790.

Zero ONLY sigma_decoder.layer2.W and .b, on all ranks, in ALL THREE arms.
Keep sigma layer1 and the entire mean path unchanged. Thus raw sigma output
starts exactly zero for every input, not just on average. Exp has no additive
floor and no hard clamp, and averages log-scales BEFORE exponentiation.
All loss arithmetic is FP32. Nonfinite loss stops the run; no silent clipping.
A first-real-batch runtime assertion checks exact initialization and agreement
with native Student-t NLL. Sigma layer1/shared sigma-path gradient is zero
at the first update due to the zero final layer; the mean path remains live.

Old running softplus+aux reference is preserved:
yuchen-fr-nu14-gaussaux1-0908, output
/mnt/pfs/yuchen/groot/ft_fr_nu14_gaussaux1_fresh_20260908.
It did NOT zero the sigma final layer (initial median about 0.47735), so it
is a useful reference but not an exactly matched fourth initialization arm.

## Exact sampled score

U ~ joint Student-t(nu=14, I_56); z = (a-mu)/sigma.
L_SE = (2/kappa) mean_k norm(z-U_k) + log(sigma).
kappa = E norm(U-U') = 11.2531740612, evaluated analytically.
The constant log(kappa) is omitted because nu and d are fixed.
No second division by d: the normalized score already has log(sigma)
coefficient 1, matching the per-dimension NLL convention.
No Gaussian auxiliary on the two SE arms. Neither mu nor sigma detached.
K=16 independent draws per input, using one chi-square(14) denominator
shared by the 56 coordinates of each draw. Integer-nu chi-square is sampled
as the sum of 14 independent squared standard normals.
Dedicated rank-local torch.Generator seed 314159+rank; does not consume
network/data-dropout RNG. No sampled action/noise is fed into the network.
No beta-NLL multiplier or online gradient balancing in any arm.

## Isolation and checkpoints

Shared cluster model/trainer sources are not edited.
Process-local forward hook replaces native loss and action_loss diagnostics.
The native HT banner describes the unused native loss on SE/exp runs;
use SCALE_SCORE_DIAGNOSTICS and training_diagnostics.jsonl as authoritative.
Custom recipe is saved in model configs and scale_score_recipe.json.
Vanilla deterministic action inference is unchanged (mu only).
IMPORTANT: vanilla sigma-dump code still assumes softplus; exp checkpoint
sigma probes MUST use scale_from_raw with the saved recipe.
Training resume requires a compatible wrapper; this fresh-start entrypoint
deliberately rejects resume rather than silently resetting sigma/optimizer.

Outputs:
- /mnt/pfs/yuchen/groot/ft_fr_nu14_se_exp_20260908
- /mnt/pfs/yuchen/groot/ft_fr_nu14_se_softplus_20260908
- /mnt/pfs/yuchen/groot/ft_fr_nu14_ht_exp_gaussaux1_20260908

## Checks

python scripts/cluster/test_scale_score_ablation.py:
passed exact initialization, native Gaussian parity, proper masking,
auxiliary mean-detach, analytic log-scale gradients, bounded SE mean
gradient, scale equivariance, joint-t second moments and shared-tail
dependence, and isolation from the global torch RNG.
Each pod reruns tests before data staging/training.
At submission, no whole 8-GPU node was free; Pending is expected.
No real optimizer step for the new arms is claimed until startup verification.

## Submission status

All three Pods were successfully created and verified Pending on 2026-09-08.
Server-side Kubernetes validation passed for all manifests. The objective unit
tests also passed under the cluster's actual GR00T Python environment before
submission. Existing running softplus+aux experiment was not interrupted.
