# Heteroscedastic Student-t (HT) loss — implementation guide

How the HT head is implemented in the runs we report (OpenVLA-OFT, GR00T N1.7, pi0.5, Cosmos3,
MIP), written so that the same objective can be reproduced on another machine for a real-robot
policy. Everything below is what the code actually does; deviations that we tested and rejected
are listed at the end.

Sources of truth (cluster paths): `oft/openvla-oft/prismatic/models/action_heads.py`
(`HTRegressionActionHead.hetero_t_loss`), `groot/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
(hetero_t branch), `cosmos3/cosmos-framework/cosmos_framework/model/generator/omni_mot_model.py`
(`_ht_action_loss`), and `mip/losses.py` in this repo.

______________________________________________________________________

## 1. The objective

Per training sample the policy predicts one action chunk `a_hat` (T steps x A dims) in one forward
pass, plus a raw scale output `s_raw` of the same shape from a parallel head. Define

- `mask`: 1 for valid action elements (chunk padding / unused dims = 0);
- `d = sum(mask)`: number of valid elements of this sample (e.g. OFT 8 x 7 = 56, GR1 8 x 29 = 232,
  pi0.5 50 x 7 = 350, Cosmos3-LIBERO 16 x 10 = 160, Cosmos3-DROID 32 x 8 = 256);
- `S = sum(mask * (a - a_hat)^2)`: squared residual norm of the whole chunk;
- `sigma = mean_over_mask(softplus(s_raw + sbias)) + 1e-3`: ONE scalar per sample;
- `nu`: a fixed constant (see section 4).

Per-sample loss (multivariate Student-t negative log-likelihood, constants dropped):

```
L = 0.5 * (nu + d) * log(1 + S / (nu * sigma^2)) + d * log(sigma)
```

Batch loss = `sum_b L_b / sum_b d_b` (mean over valid elements). Nothing else: no loss scale,
no beta-NLL weighting, no extra regularizer.

What it does: the gradient with respect to `a_hat` is `w * (a_hat - a) / sigma^2` with
`w = (nu + d) / (nu + S / sigma^2)`. For a sample whose residual is in the bulk (`S << nu sigma^2`)
`w ~ (nu + d) / nu`, i.e. the loss is an ordinary Gaussian NLL; for a sample with
`S >> nu sigma^2` the weight falls like `1/S`, so its gradient contribution stops growing.
The knee is at `S = nu sigma^2`, i.e. a per-element rms residual of `sqrt(nu / d) * sigma`.
With `nu = c^2 * d` the knee sits at `c` sigma per element.

Two other forms you will see in our code, equivalent up to the meaning of `nu`:

- "legacy" form `0.5 * (nu' + 1) * log(1 + S / (nu' * sigma^2 * d)) * d + d * log(sigma)`
  (the MIP-harness estimator). It equals the form above with `nu = nu' * d`. So `nu' = 2` in the
  legacy form is `nu = 2d` in the multivariate form. Do not mix the two conventions.
- the heteroscedastic Gaussian limit `nu -> inf`: `0.5 * S / sigma^2 + d * log(sigma)`. We use it
  only as a control; it is clearly worse (OFT-long 73 vs 96 at 50k).

## 2. Reference implementation

```python
import torch, torch.nn as nn, torch.nn.functional as F

class HTHead(nn.Module):
    """Mean head + parallel sigma head. Any regression head works as `mean`;
    `sig` is a copy of it (OFT: same MLPResNet; Cosmos3: a single Linear on the
    detached prediction; GR00T: the same category-specific MLP as the action decoder)."""
    def __init__(self, mean_head: nn.Module, sigma_head: nn.Module, sbias: float, nu: float):
        super().__init__()
        self.mean, self.sig = mean_head, sigma_head
        # zero-init the LAST linear layer of the sigma head so that
        # sigma == softplus(sbias) for every sample at step 0
        last = [m for m in self.sig.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.sbias, self.nu = float(sbias), float(nu)

    def forward(self, feats):                     # feats: [B, ...] from the backbone
        return self.mean(feats), self.sig(feats)  # each [B, T, A]


def ht_loss(pred, s_raw, target, mask, nu, sbias):
    """pred, s_raw, target: [B, T, A]; mask: [B, T, A] with 1 = valid element."""
    with torch.autocast("cuda", enabled=False):               # fp32: log/divide chain is not bf16-safe
        pred, s_raw, target, mask = pred.float(), s_raw.float(), target.float(), mask.float()
        d = mask.sum(dim=(1, 2)).clamp_min(1.0)                                 # [B]
        sigma = (F.softplus(s_raw + sbias) * mask).sum(dim=(1, 2)) / d + 1e-3   # [B] scalar per sample
        S = (((target - pred) ** 2) * mask).sum(dim=(1, 2))                     # [B]
        per_sample = 0.5 * (nu + d) * torch.log1p(S / (nu * sigma ** 2)) + d * torch.log(sigma)
        return per_sample.sum() / mask.sum()
```

Details that matter:

- `sigma` is a per-sample scalar (masked mean of softplus over the chunk). Per-dimension or
  per-element sigma was tested and is worse (section 7).
- The sigma head may take the backbone features (OFT, GR00T) or the detached mean prediction
  (Cosmos3). Both work. The Cosmos3 variant (`Linear(T*A -> 1)` on `pred.detach()`, zero-init) is
  the smallest and prevents the mean path from influencing sigma through shared features; use it
  if you want the fewest moving parts.
- The mean head must receive gradient only through the NLL above; no auxiliary MSE.
- Compute the loss in fp32 even if the model runs in bf16.

## 3. Single-pass readout and inference

- Plain regression heads (OFT's L1 head): use as is; HT only replaces the loss and adds the
  sigma head.
- Flow / diffusion heads (GR00T, pi0.5, Cosmos3): keep the network, but train and run it as a
  single pass: feed zeros as the "noisy" action, timestep t = 0 (bucket 0), and take the network's
  output as the action chunk directly. The velocity target is not used. This reuses the pretrained
  weights: at (zeros, t = 0) a pretrained flow head is already a competent regressor (GR1 base model
  zero-shot residual rms 0.29 on normalized actions).
- Inference: one forward, execute `a_hat`. The sigma head is not used at deployment (drop its
  weights when loading, e.g. `strict=False` on `sigma_model.*`). There is no noise channel and no
  sampling; the policy is deterministic.

## 4. Hyperparameters

| quantity | value we use | how to set it |
|---|---|---|
| `nu` | `nu = c^2 * d` with `c = 2`, i.e. `nu = 4d` | Default prescription. Verified: GR1 `nu = 928` 51.5 vs 47.5 at `2d`, WidowX `nu = 224` (= sweep optimum), Cosmos3 `nu = 640` (= flow, 96), OFT-long `nu = 224` 95 @150k vs 98 for the swept best (`nu = 1024`, = 18d), both above the released L1 head (94.5). pi0.5 is the mild exception where `2d` beats `4d` (97.6 vs 96.6, budget-matched). Anything in `[2d, 20d]` is within a few points on every stack; `nu = 2d` was the accidental default that underperformed on OFT-long (82) and Cosmos3 (93). |
| `sbias` | `sbias = log(exp(rms_0) - 1)` so that `softplus(sbias) = rms_0` | `rms_0` = per-element rms of the residual at initialization on a few hundred training samples (run the initialized head once; for a fresh head this is the rms of the normalized action targets). Values used: OFT libero-long `-0.373`, GR00T GR1 `-1.088` (rms 0.29), pi0.5 `-0.396`, Cosmos3-LIBERO `0.589`, Cosmos3-DROID `0.354` (raw joint positions, rms 0.886). A 10x miscalibration either removes the gradient (sigma too large) or gates most samples from step 0 (sigma too small). |
| everything else | baseline recipe unchanged | Same lr, schedule, batch size, EMA, augmentation and checkpoint cadence as the flow / L1 baseline. The loss is normalized per element, so no loss-scale change. |
| `beta` (beta-NLL) | 0 | Tested; no gain. |

`d` is the number of valid elements per sample after masking; if chunks are padded to a fixed
length, use the mask so that `d` and `sigma` are computed over real elements only.

## 5. What to log

Every ~200 steps, on rank 0, straight from the loss site:

- the effective `nu`, `sbias`, `d` (print once at step 1 — see pitfall 6a);
- `sigma` median and q10/q90;
- residual rms median (`sqrt(S / d)`);
- fraction of samples past the knee, `mean(S / sigma^2 > nu)`.

Expected: with a calibrated sigma the gated fraction is a few percent (OFT `nu = 1024`: 3.6%;
pi0.5: 24% with a pooled sigma). GR1's per-sample sigma under-predicts the residual ~5x and 97% of
samples sit past the knee; it still trains to the best result, so a large fraction is not by
itself a failure, but it means the loss operates in a Cauchy-like regime and `nu` matters less.

## 6. Pitfalls we hit (checklist)

a. **Config plumbing.** GR00T's `from_pretrained` whitelist dropped `loss_type / sbias / nu`, so
   two "HT" runs trained the flow objective. Verify at step 1: the sigma head's parameters exist
   in the saved checkpoint, and a banner printed from inside the loss shows the effective values.
   Never read hyperparameters with `getattr(cfg, "x", default)` at the loss site — a missing field
   must fail loudly.
b. **Precision.** Compute the NLL in fp32 (`autocast(enabled=False)`); `lgamma` in bf16 crashes
   (relevant only if you ever make `nu` learnable — do not, see section 7).
c. **DDP.** The head's parameters must be gradient-synced. In OFT the training script called
   `action_head.module.*`, bypassing DDP, so head parameters drifted per rank and the saved head
   was rank 0's. Check that the head is inside the DDP wrapper (or that its gradients are
   all-reduced).
d. **Sigma head initialization.** Zero-init the last layer so sigma starts exactly at
   `softplus(sbias)`; otherwise the calibration in section 4 is meaningless.
e. **Masks.** Compute `d`, `sigma` and `S` over the same mask; padded chunk steps or unused action
   dims must be excluded from all three.
f. **Fresh vs pretrained head.** With a fresh action head the early phase is slower than with a
   pretrained one (the sigma head has nothing to calibrate against yet); keep the baseline's
   warm-up and do not judge the run before it has passed the warm-up.
g. **Budget matching.** Check the samples-seen counter (lerobot: `smpl:` in the train log) at a fixed step when
   comparing arms; a script cloned without `torchrun` trained pi0.5 on 8x fewer samples and produced a misleading
   96.1 vs 97.6 comparison before this was caught.
h. **Checkpoint rotation.** Trainers that keep only the last N checkpoints delete mid-run points;
   evaluate during the run if you want a curve.
i. **Deployment.** Load the mean weights only; run one forward; no sampling. If the deployment
   stack has a flow sampler, bypass it (GR00T: short-circuit the Euler loop to one call at t = 0).

## 7. Variants we tested and do not recommend

| variant | result | reason |
|---|---|---|
| learnable `nu` by maximum likelihood (per-sample or global) | 2-7 points below the fixed-`nu` sweep on every stack (`nu` lands at 2-15) | ML `nu` fits the over-dispersed residuals; the SR-optimal `nu` is a gate position, not a fit |
| per-dimension or per-element sigma | worse or collapses (WidowX in/perdim 0.429 vs 0.629) | the split-by-dimension gate is unreliable |
| median-pinned `nu = c^2 * median(S / sigma^2)` | ties or -4 vs fixed `nu = 4d` | no gain over the fixed rule |
| argmax-`nu` lookup table | 91-92 @50k vs 94-96 for fixed `nu` | same |
| heteroscedastic Gaussian (`nu = inf`) | OFT-long 73 vs 96 @50k; GR1 20.6 vs 47.5 | tail samples take 18% of the gradient (top 1%) instead of ~1% |
| beta-NLL weighting | no gain | — |
| simple Huber cap (`delta = 2.5 x EMA-median chunk-residual norm`, no sigma head) | OFT-long 97 @50k (= HT within noise) but GR1 37.4 @60k (= MSE 37.8, vs HT 47.5) | not a substitute: it matches HT only where the residual tail is mild (OFT) |
| heteroscedastic Huber (Huber rho on the sigma-normalized norm + `d log sigma`) | OFT-long 95 @50k; GR1 40.7 @60k (between MSE 37.8 and HT 47.5) | the per-sample sigma recovers about half of the GR1 gap; the redescending t-gate is needed for the rest |
| heteroscedastic L1 (`sum|r|/sigma + d log sigma`, per-sample sigma) | GR1 44.5 @60k (above hetero-Huber 40.7, level with flow 44.1, under HT 47.5 / 51.5); OFT-long 69 @20k (HT c=2 89, MSE 84) | the sigma head carries most of the GR1 gain with an L1 kernel, but the Student-t gate is still 3-7 points better there and 20 points better on OFT-long at 20k |

## 8. Reference results (single-pass HT vs iterative baseline, budget-matched)

| stack (d) | baseline | HT | nu |
|---|---|---|---|
| GR00T GR1 robocasa (232) | flow 44.1 @60k | 51.5 @60k (47.5 at nu = 464) | 928 (4d) |
| GR00T WidowX bridge (56) | flow 57.1 @20k | 67.4 @20k | 224 (4d) |
| OpenVLA-OFT libero-long (56) | released L1 94.5 | 98 @150k | 1024 (18d); 4d (224): 95 @150k |
| OpenVLA-OFT libero-goal / object | 97.9 / 98.4 | 98 / 97 @150k | 1024 |
| pi0.5 LIBERO 4-suite (350) | published flow ~96.9 | 97.6 @30k (96.6 at nu = 1400 = 4d) | 700 (2d) |
| Cosmos3 libero-10 (160) | flow 96 @2000 | 96 @2000 | 640 (4d) |

Full tables and the mechanism results (gradient shares, gate fractions, multimodality probes)
are in `analysis/paper/vla_results_table.md` and `analysis/paper/multimodality_table.md`.

## 9. Minimal recipe for a new real-robot stack

1. Normalize actions (the same normalization the baseline uses); note `T`, `A`, and the mask.
2. Add a sigma head (copy of the mean head, or `Linear(T*A -> 1)` on the detached prediction),
   zero-init its last layer.
3. Set `nu = 4 * T * A` (or `4 * d` with masking).
4. Measure `rms_0` of the initialized head's residual on ~300 training samples;
   set `sbias = log(exp(rms_0) - 1)`.
5. Replace the loss with `ht_loss` above (fp32), keep every other training hyperparameter.
6. Log `sigma`, residual rms and the gated fraction; confirm the banner at step 1.
7. Deploy the mean head only, one forward per chunk.

Do not replace the Student-t NLL with a Huber cap to save effort: the raw-norm cap reproduced HT on OFT
but not on GR1 (37.4 vs 47.5), and even the sigma-normalized Huber reached only 40.7 there.
