# Kronecker-factored covariance for the HT (heteroscedastic Student-t) head

Review package. The code under review runs on the cluster inside NVIDIA's Isaac-GR00T tree at
`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`; the files here are copies so the math can be read and
checked locally.

| file | what it is |
|---|---|
| `kron_branch.py` | the branch under review, copied verbatim out of `gr00t_n1d7.py` |
| `ht_kron_patch3.py` | the CURRENT patch script that produces that branch (restores a clean backup, then inserts) |
| `ht_kron_patch2.py` | previous version, kept to document the round-1 fixes |
| `check_kron.py` | purpose-built verification: re-implements the branch and checks it against brute force |
| `loss_excerpt.py` | surrounding context from `gr00t_n1d7.py` (variable definitions the branch relies on) |
| `ht_kron_patch.py` | the FIRST version of the patch, kept only to show the initialisation bug that was fixed |

## What the head models

The policy predicts an action chunk of `T = 8` steps x `A = 7` dims, so the residual for one sample
is a matrix `R` of shape `(T, A)` and `d = T*A = 56`. The existing HT loss assumes an isotropic
scale, `Sigma = sigma^2 I` with one scalar `sigma` per sample. This branch replaces that with a
full `d x d` covariance in Kronecker form,

    Sigma = Sigma_T (x) Sigma_A          Sigma_T: T x T over chunk steps, Sigma_A: A x A over action dims

which is 36 + 28 = 64 parameters per sample instead of 1596 for an unstructured 56x56, and gives the
matrix-variate Student-t negative log-likelihood

    M        = tr(Sigma_T^-1 R Sigma_A^-1 R^T)
    log|Sig| = A*log|Sigma_T| + T*log|Sigma_A|
    loss     = 0.5*(nu + d)*log1p(M/nu) + 0.5*log|Sig|

`M` is computed with two batched triangular solves rather than forming `Sigma`.

## Per-sample requirement

The covariance must be **learned per sample** — every batch element gets its own `Sigma_T`,
`Sigma_A` predicted from that sample's features, not a shared/global matrix. The parameters come
from the existing sigma head (`sigma_decoder`, output width 132), pooled over that sample's valid
timesteps, taking the first 64 entries. No new parameters, so checkpoint keys and the DDP parameter
count are unchanged.

## Identifiability and initialisation

`(c*Sigma_T) (x) (Sigma_A/c)` gives the same `Sigma` for any `c`, so `Sigma_A` is renormalised to
unit determinant and the whole scale lives in `Sigma_T`. That constraint drives the initialisation:

The biases are **floor-aware**, so the `+1e-4` on the Cholesky diagonal is accounted for, and the
target is the isotropic branch's *effective* sigma `sigma_iso = softplus(ht_sbias) + 1e-3` (that
branch adds its own `+1e-3`), not the bare `sigma0`:

- `diag(L_T)` init = `softplus(softplus^-1(sigma_iso - 1e-4)) + 1e-4 = sigma_iso` -> `Sigma_T = sigma_iso^2 I`
- `diag(L_A)` init = `softplus(softplus^-1(1 - 1e-4)) + 1e-4 = 1` -> `Sigma_A = I`, normalisation is a no-op
- hence `Sigma = sigma_iso^2 I` at step 0 and the loss equals the isotropic branch's loss exactly

where `sigma0 = softplus(ht_sbias) = 0.4706` is the base model's probed zero-shot residual RMS and
`sigma_iso = 0.471576`.

**Two initialisation bugs were found and fixed.** (1) `ht_kron_patch.py` put `sqrt(sigma0)` on both
diagonals, reasoning `(sqrt(sigma0))^2 * (sqrt(sigma0))^2 = sigma0^2`; the unit-determinant
normalisation then rescaled `L_A` to 1 and stripped that factor, leaving `Sigma = sigma0 I` —
2.125x too large (effective sigma 0.686 vs 0.4706). A 3-hour training run was discarded.
(2) Codex round 1 (R1-1): even after that fix, the `+1e-4` diagonal floor and the isotropic branch's
own `+1e-3` meant the effective scales differed (0.470676 vs 0.471576) and the loss was 0.077 nats
off at init. The biases are now floor-aware and target `sigma_iso`, so the two losses agree to fp32
precision (5.7e-06).

Also fixed in round 1: the mask guard (R1-2) now requires every sample's mask to equal
`rows (x) cols` exactly, so a sample with an interior hole (rows and columns all still non-empty)
falls back to the isotropic branch instead of silently using `d = 56` with a masked element in `R`.

## Acceptance criteria

1. The covariance is learned **per sample**: distinct `Sigma_T`/`Sigma_A` per batch element, and one
   sample's head output affects only that sample's loss.
2. The Mahalanobis term equals `vec(R)^T (Sigma_T (x) Sigma_A)^-1 vec(R)` on the explicit 56x56
   covariance, and `log|Sigma|` equals its explicit log-determinant.
3. At initialisation `Sigma == sigma_iso^2 I` with `sigma_iso = softplus(ht_sbias) + 1e-3`, so the
   loss coincides EXACTLY with the production isotropic multivariate-t loss it must start from
   (floors included, not merely close).
4. The Kronecker non-identifiability is fixed (`|Sigma_A| = 1`).
5. `ht_cov_mode` defaults to `"iso"`, so every existing arm (flow, isotropic HT at nu=112/224/896)
   is bit-identical to before the patch.
6. Numerically sound in the training loop: fp32, strictly positive Cholesky diagonals, no explicit
   `d x d` inverse.

## Verification

`python3 check_kron.py` (needs only torch). It mirrors the deployed branch line for line, INCLUDING
the sigma-head pooling and the mask guard, and runs in fp32 like production:

```
1. per-sample: factors differ=True, only sample 3 moves=True, padded steps ignored=True
2. vs explicit 56x56 (float64 reference): Mahalanobis rel diff 1.116e-07, log|Sigma| rel diff 8.678e-08
3. init: Sigma vs sigma_iso^2 I (sigma_iso=0.471576) diff 0.000e+00; loss vs production isotropic diff 5.722e-06
4. log|Sigma_A| (must be 0): max abs 8.047e-07  (fp32 rounding; 1.2e-15 in float64)
5. guard rejects interior hole=True, rejects ragged mask=True; branch taken for (absent, 'iso', 'kron') = [False, False, True]
6. autocast bf16: loss dtype torch.float32, factors fp32=True, finite non-zero grads=True, positive diagonals=True

ALL CHECKS PASS
```

Criterion 5 is additionally verified on the deployed file: `diff` against the pre-patch backup shows
**0 lines removed and 55 added**, so every pre-existing branch is byte-identical.

On the cluster, a 20-step smoke run on the 19k-episode subset trained without error and printed the
banner confirming the shapes and the initialisation.
