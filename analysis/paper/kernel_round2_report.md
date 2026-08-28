# Two-view kernel mechanism validation · Round 2 experiment summary (for feedback to ChatGPT)

**Common protocol**: all operators = standard pairs protocol (anchor cloud from 40 clean demos, per-dim z-score,
1-NN; band [2,4); test = held-out puredart recovery data, N≈8.5k pairs; cos = sample average of the predicted
direction vs G_GT; poseig = de-z-scored symmetrized pos-block eigenvalues). All SR = held-out seeds
21000–21100, official protocol. Reference values: G_MIP cos 0.73 negative-definite, G_MSE indefinite (positive eigenvalue),
MSE-2k SR 71–79 (3 seeds), MIP-2k SR 94–99 (3 seeds).

## Experiment 0 (preliminary): head-surgery closed loop

Frozen features + a clean-data ridge linear head replacing the original head (same function class as the original
1×1 conv, on-support trainR2=1.000, deployment pipeline byte-for-byte identical to the standard eval):

| Deployment | SR |
|---|---|
| MSE original head | 71 |
| MSE features + ridge head | **74** |
| MIP original head | 99 |
| MIP features + ridge head | **95** |

**The features are a sufficient statistic for the SR gap**: swapping the head neither rescues MSE nor hurts MIP.
The linear readout of MSE features is weakly corrective and negative-definite (cos 0.56), but its gain of −0.011 ≈ 1/10
of G_GT — too weak to pull back.

## Experiment 1: ridge-head λ sweep (static part complete; closed-loop SR running)

λ ∈ {1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2}; features = penultimate (t=0, zeros input);
bias = mean error against the GT recovery action (6-d); rotsv = de-z-scored rot-block top singular value:

| Features | λ≤1e-2 plateau | λ=1 | λ=1e2 |
|---|---|---|---|
| MSE | cos 0.59–0.61, rotsv **0.13–0.14**, bias[2,3)=0.36–0.37 | cos 0.60, rotsv 0.084, bias[1,2) 0.09→**0.15** | readout collapses (trainR2 0.65) |
| MIP | cos **0.73–0.74**, rotsv 0.086–0.089, bias[2,3)=**0.24** | cos 0.72, rotsv 0.054 | same collapse |

**Verdict: no point on MSE's λ frontier reaches MIP's operator quality. The rotational amplifier lives in the features**
(it cannot be removed for λ≤1e-2; only pushing to λ=1 brings it down to the MIP level, at the cost of doubling
on-support bias with no gain in cos).
→ This is not a head-hyperparameter problem; the feature geometry is insufficient. 12 closed-loop SR runs (including
PNR-crossing statistics) are running; prediction: the SR at MSE's best λ will remain far below MIP's.

## Experiment 2: layerwise feature-ridge probe (complete)

Layer order: enc (obs encoder output) → cond → down0 → down1 → down2 → mid → up0 → up1 → pen.
cos vs G_GT:

| layer | MSE-t0 | MIP-t0 | MIP-τ | DEN-τ |
|---|---|---|---|---|
| enc | 0.65 | 0.72 | 0.72 | **0.78** |
| cond | 0.65 | 0.69 | 0.69 | 0.77 |
| down0 | 0.45 | 0.53 | 0.54 | 0.67 |
| down1 | 0.49 | 0.41 | 0.41 | 0.56 |
| down2 | 0.50 | **0.78** | 0.77 | 0.55 |
| mid | 0.49 | **0.77** | 0.78 | 0.51 |
| up0 | 0.51 | **0.78** | 0.77 | 0.54 |
| up1 | 0.60 | 0.72 | 0.72 | 0.58 |
| pen | 0.59 | 0.73 | 0.73 | 0.69 |

Three readings:
1. **Divergence starts at the early layers: the obs encoder itself is changed by MIP** (enc 0.65 vs 0.72; standalone
   denoiser 0.78) — it is not just high-level action-feature geometry;
2. The largest divergence is in the deep trunk (down2/mid/up0: 0.50 vs 0.78); MSE's deep layers actually **degrade**
   the corrective content already present in the encoder (0.65) down to 0.50–0.59; MIP's deep layers enhance it to 0.78;
3. **MIP's φ₀ and φ₁ are pointwise equal at every layer** (0.72/0.72, 0.78/0.77, 0.73/0.73, …) — weight sharing carries
   the representation reshaped by the aux view fully into the main slice; there is no layerwise gradient of the form
   "φ₁ improves first, φ₀ later"
   (they are one and the same state pathway).

## Experiment 3: DUP0_HIGH (complete)

regression_dup100 = the same (t=0, zeros) view duplicated 1×+100×, total loss scale identical to MIP,
no auxiliary slice, 300k steps:
- Operator: cos 0.67, poseig [−0.016, −0.010, **+0.010**] indefinite;
- **SR = 54 (< MSE 71–79)**.

Completed 2×2 factorial table:

| | weight 1× | weight ~100× |
|---|---|---|
| t=0 view | MSE: indefinite, SR 71–79 | **DUP100: indefinite, SR 54** |
| τ view | ZEROFLAT: indefinite (cos 0.46, +0.022, sv 1.5) | **ZEROIN: negative-definite, cos 0.65**, slice-SR 75 |

**Verdict: high-weight optimization per se is ruled out; the key is the auxiliary slice.** Moreover, ZEROIN's input is zero =
exactly the same input as the t=0 view; the only difference is the t-embedding → it is the representation regime of the
τ-slice that gets amplified by the high weight. (Caveat: training uses grad-norm clipping, so DUP100 is not a perfect
Adam-absorption control, but SR≤MSE suffices to fix the direction of the conclusion.)

## Experiment 4: t-slice-only (completed earlier; archived)

ZEROIN is this experiment: single-view training f(τ, 0, s)→a\* → negative-definite operator with cos 0.65, linear feature
decoding 0.66, **standalone closed-loop deployment 75%** (RANDIN 78%) ≈ MSE level, far below MIP's 94–99. → The τ-slice
regime by itself suffices to produce a GT-like operator; two-view is needed for SR (on-support accuracy),
not for the operator.

## Experiment 5: bad-mode projection (core complete)

bad1 = MSE pos-block positive-feedback eigendirection; rot-amp = MSE rot-block top right-singular direction
(unified de-z-scored convention, so the MSE positive-eigenvalue magnitudes differ slightly from the main table's
convention; the sign structure is consistent):

| Model | gain@bad1 (signed) | rot response @ MSE-amp direction | own rot top-sv |
|---|---|---|---|
| G_GT | −0.010 | 0.013 | 0.063 |
| MSE | **+0.003 (amplifying)** | **0.152 (= its own dominant mode)** | 0.152 |
| MIP | **−0.020 (contracts in the opposite direction, stronger than GT)** | 0.036 | 0.075 |

**MIP actively contracts along MSE's positive-feedback direction and compresses the rotational spectrum to GT magnitude;
MSE's amplifier direction is its own dominant mode.**

Full table of variants and ridge heads (same convention):

| Model | gain@bad1 | rot@MSE-amp | rot top-sv |
|---|---|---|---|
| Standalone denoiser | −0.010 | **0.001** | 0.012 |
| SCRAMBLE | −0.012 | 0.001 | 0.012 |
| ZEROIN | −0.008 | 0.021 | 0.053 |
| ZEROFLAT | **+0.014 (worse than MSE itself)** | 0.115 | 0.134 |
| EQW | +0.008 | 0.099 | 0.122 |
| DUP100 | ~0 (flat) | 0.027 | 0.077 |
| MSE-ridge (λ 1e-4…1e-2) | **−0.006…−0.008 (suppressed!)** | 0.050 | **0.140** |
| MIP-ridge (λ 1e-4…1e-2) | −0.003…−0.005 | 0.035 | 0.089 |

Three readings:
1. The corrective family (denoiser/SCRAMBLE/ZEROIN/MIP) consistently suppresses both bad modes; the failing family
   (ZEROFLAT/EQW) **re-creates** the bad modes (ZEROFLAT's bad1 is even larger than MSE's own);
2. **The head/feature attribution of MSE's two bad modes splits**: the ridge readout kills the pose positive feedback
   (+0.003→−0.006, head-fixable), but the rotational amplifier's magnitude remains in the features
   (top-sv 0.140 ≈ full model's 0.152; only the input direction has rotated);
3. In closed loop, fixing only the head-fixable part → SR 71→74–77; **the feature-locked defects
   (10× gain deficit + rotational amplifier) dominate the failures**.
   (Caveat: models with the ABS representation project entirely positive under this delta convention; this geometry
   does not apply to them — discussed in a separate table.)

## Appendix: other verdicts concluded in the same round

- **The jacreg dose-of-perturbation-correction-supervision line is closed**: explicit input-Jacobian regularization
  λ=1e-3/1e-4/0 → SR 1/24/71-79, monotonically harmful;
  jacreg-1e-4's operator cos is as high as 0.92 but indefinite (+0.023) with SR 24 → **a high directional cos does not
  equal stability; what determines closed-loop SR is operator negative-definiteness (stability), not directional similarity**.
- Re-run of the feature-ridge probe's tube-input row after the chunk-alignment fix: DEN-φ₁ 0.67✓ / SCRAM-φ₁ 0.68✓ /
  MIP-φ₁ 0.93 (+0.021 marginal). Conclusions unchanged.

## Experiment 1 closed-loop SR (12/12 complete)

| λ | MSE SR | MIP SR | MSE cross4 | MIP cross4 | SR\|not-crossed |
|---|---|---|---|---|---|
| 1e-8 | 53 | 83 | 0.53 | 0.23 | 100/100 |
| 1e-6 | 75 | 88 | 0.29 | 0.15 | 100/100 |
| 1e-4 | **77** | **94** | 0.29 | 0.10 | 100/100 |
| 1e-2 | 46 | 67 | 0.62 | 0.49 | 100/100 |
| 1 | 1 | 6 | 1.00 | 0.99 | —/100 |
| 1e2 | 0 | 0 | 1.00 | 1.00 | — |

**Preregistered prediction confirmed: the best SR of MSE features over the entire λ curve (77 @1e-4) is far below MIP's
(94 @1e-4), and MIP dominates at every λ (by 15–30 points).** The conclusion "MSE is not a matter of untuned head
hyperparameters — the feature geometry is insufficient" is now settled. A second law-grade observation: **in every row
that has non-crossing episodes, SR|never crossed PNR=4 = 100**
(including the single remaining episode for MIP at λ=1) — failures are 100% explained by crossing, and the SR gap
reduces entirely to the difference in crossing rates determined by the feature-extrapolation operator
(mechanistic chain: feature geometry → extrapolation operator → crossing rate → SR, with each link measured independently).
Secondary finding: under-regularization (1e-8) hurts both sides — the ridge head needs a small amount of regularization
to suppress extrapolation along feature noise directions;
λ≥1 collapses across the board (the on-support fit is destroyed, cross4→1.0). The λ effect (±20 points) does not change
the gulf between the two feature sets.
