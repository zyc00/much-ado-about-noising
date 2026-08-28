# Mechanism validation · Rounds 3–4 summary (early-trajectory reversal + ET batch)

Protocol unchanged: operator = pairs protocol (anchor cloud from 40 clean demos, z-score 1-NN, band [2,4),
test = held-out puredart); SR = held-out seeds 21000–21100, official protocol;
badmode = unified de-z-scored convention (bad1 = MSE positive-feedback pose direction, rot-amp = MSE rotational
amplification direction);
cross4 = fraction of episodes whose maximum support distance over the rollout is ≥4 (=PNR).

## 0. Central conclusion (the two-factor law)

    SR = (1 − cross4) × 100 + cross4 × SR|crossed
    cross4      ≈ f(per-step absolute accuracy in the deployed state)        — determines "whether you drift out"
    SR|crossed  ≈ g(negative-definiteness × gain of the far-field operator)  — determines "whether you can come back once out"
    Across all 30+ closed-loop settings, without exception: SR|not-crossed = 100%.

- **Directional cosine does not predict SR** (three counterexamples: AUXDET cos 0.91 indefinite → SR 64; jacreg-1e4 cos
  0.92 indefinite → 24;
  early MSE cos 0.84 negative-definite but weak gain / poor accuracy → 51). What predicts SR|crossed is
  negative-definiteness + GT-magnitude gain.
- **MSE's tragedy is a timing mismatch**: the extrapolation geometry is destroyed before accuracy matures;
  **MIP's role = keeping the extrapolation geometry alive through the accuracy-maturation period** (not creating it).

## I. Timeline (ET1 / ET5 / trajectory probes)

### Full MSE trajectory (8 ckpts, closed-loop + static side-by-side)
| step | SR | cross4 | SR\|crossed | operator cos | rotsv | cleanErr p50 (train/held gap≈1.0) |
|---|---|---|---|---|---|---|
| 1k | 21 | 0.87 | 9 | 0.78 | 0.090 | 0.0292 |
| 5k | 46 | 0.70 | 23 | 0.83 | 0.082 | 0.0164 |
| 10k | 51 | 0.56 | 12 | **0.84 negative-definite** | 0.078 | 0.0109 |
| 25k | 41 | 0.65 | 9 | 0.83 / 0.66* | 0.012/0.063 | 0.0086 |
| 50k | 44 | 0.58 | 3 | **0.49** | **0.124** | 0.0062 |
| 100k | 63 | 0.39 | 5 | 0.52 | 0.129 | 0.0056 |
| 200k | 71 | 0.34 | 15 | 0.51 | 0.110 | 0.0049 |
| 300k | **79** | 0.22 | 5 | 0.56 | 0.104 | — |

(*Two independent runs at 25k: early run 0.83 / traj run already 0.66 at 27k — the destruction window is 25–50k,
shifting slightly across seeds.)

Readings:
1. **The corrective geometry emerges spontaneously under pure MSE within 10k steps (0.84 negative-definite, no rotational
   amplifier, negative bad1)**;
2. **The destruction is a fast phase transition at 25k–50k, after which the model remains stable in the destroyed state
   for 250k steps** (0.46–0.56 + rotsv 0.10–0.14);
   it coincides with the marginal clean-error improvement period 0.0086→0.0062; gap≈1.0, not classical overfitting;
3. **Yet SR rises monotonically (21→79)**: cross4 is driven exclusively by accuracy; the destroyed operator caps MSE at 79
   (0.22 crossing × ~95% death after crossing);
4. SR|crossed stays at 3–23 throughout: early on the direction is good but the gain is weak and drift goes too deep
   (median maxd 41–107); later, the direction is destroyed.
   **MSE never has a phase in which recovery / pull-back is possible.**

### MIP trajectory (maintenance)
| step | operator cos | SR (step1) | cross4 | SR\|crossed |
|---|---|---|---|---|
| 1k | 0.81 | — | — | — |
| 5k–25k | 0.73–0.74 | 82 (s25k) | 0.20 | 10 |
| 100k | — | **96** | **0.07** | **43** |
| 300k | 0.73 | 95 | 0.07 | 29–40 |

**MIP settles at 0.73 by step 5k and does not move for 300k steps** (no destruction anywhere along the trajectory);
its clean error is below MSE's at every time point
(100k: 0.0032 vs 0.0056) — **no accuracy–extrapolation tradeoff; both curves arrive together**.
Layerwise dissection: the destruction site is the trunk (at 300k, MSE's enc is still negative-definite at 0.68, while
down0–pen are all indefinite at
0.41–0.61); MIP's deep trunk at the same location is 0.78.

## II. Intervention experiments

### ET2: frozen early trunk + ridge head (Verdict: no rescue)
| Frozen trunk | λ | trainR2 | SR | cross4 |
|---|---|---|---|---|
| s10k | 1e-3 | 0.999 | 39 | 0.78 |
| s25k | 1e-3 | 0.999 | 53 | 0.59 |
| s10k/s25k | 1e-1 | 0.997 | 13/17 | 0.97/0.92 |

Even an on-manifold fit of 0.999 cannot suppress crossing — deployed-state accuracy ≠ training-state fit.

### EXP6 lazy family, full table (Verdict: total casualty across the line)
| Variant | SR | Operator cos | Negative-definite |
|---|---|---|---|
| MSE baseline | 71–79 | 0.5–0.6 | ✗ |
| headonly (frozen random trunk, SGD head) | **0** | **0.82** | ✓ |
| lowlr (lr/10) | 50 | **0.77** | ✓ |
| best early stopping (s10k) | 51 | 0.84 | ✓ |
| wd×1000 | 60 | 0.51 | ✗ |
| dup100 without grad-clip (B4) | 52 | 0.53 | ≈ |
| trunklr 0.1× (E) | in flight | | |

Lazy variants preserve geometry but lose accuracy; the net effect is ≤ MSE in every case. **The only intervention that
improves both simultaneously is the τ-weighted auxiliary view.**

### EXP3: AUX_GRAD_BLOCK / NO_SHARED_TRUNK (Verdict: the shared trunk is the locus of the mechanism)
| Variant | head0 SR | Operator |
|---|---|---|
| Normal MIP | 94–99 | 0.73 negative-definite |
| **AUX_GRAD_BLOCK** (aux gradients enter only final_conv) | **64** (2-step 59) | cos 0.91 **indefinite** (+0.022) |
| NO_SHARED_TRUNK (≡ MSE + standalone denoiser, by construction) | 71–79 | indefinite |

The aux gradients must enter the shared trunk; training only the late branch is ineffective and even harmful.

### ET3: mid-training takeover from MSE s10k (5 continued-training runs of 150k, geometry curves on a 25k grid)
(Methods note: the first version of ET3 was voided because `model_path` also loaded the optimizer/scheduler — the lr was
decayed to 1e-15 by scheduler
residue and the override was overwritten; v2 was rerun with INIT_CKPT loading weights only, with the lr verified run by run.)

**ET3 v2 final table (all from the same MSE s10k starting point, +150k steps; starting point: cos 0.84 / cleanErr 0.0109 / SR 51):**

| Variant | Final geometry cos / rotsv / rot@amp | cleanErr p50 | SR | cross4 | drift([2,4)) | SR\|crossed-4 |
|---|---|---|---|---|---|---|
| a standard MSE | **0.41 / 0.179 / 0.162 (destroyed within 17k steps)** | 0.0055 | 62 | 0.42 | **+3.67 (repulsive)** | 10 |
| **b switch to MIP objective** | **0.72 / 0.055 / 0.020 (stable throughout)** | **0.0030 (best overall)** | **84** | **0.21** | **−0.27 (pull-back)** | 24 |
| c low-LR MSE | 0.87 / 0.080 / 0.011 (preserved) | 0.0069 | 73 | 0.32 | −0.19 | 16 |
| d strong-WD MSE | 0.41 / 0.175 / 0.157 (destroyed all the same) | 0.0053 | 64 | 0.38 | −0.21 | 5 |
| e frozen trunk | 0.82 / 0.073 / 0.008 (trivially preserved) | 0.0115 (**unmoved over 150k**) | 51 | 0.60 | −0.06 | 18 |

b's geometry trajectory: 0.80 (26k)→0.74→0.73→0.72→0.71→0.72 — it descends smoothly from the MSE starting point of 0.84
to the MIP characteristic value of 0.72 and then holds; rot-amp ≤0.020 throughout; cleanErr 0.0060→0.0030, matching
from-scratch MIP at the same stage (0.0032@100k).

**Adjudication by the preregistered decision rules: Case 1 + Case 3 jointly hold, Case 2 partially holds, Case 4 is ruled out.**
1. (Case 1) From the same starting point: continued standard-MSE training reproduces the destruction within 17k steps
   (with accuracy maturing), whereas switching to the MIP objective keeps the geometry stable throughout **and matures
   accuracy faster and better** (0.0030 < a's 0.0055) — MIP is not merely a protective
   regularizer but also a better accuracy optimizer; "MIP prevents late-phase degradation" is proven by an intervention experiment;
2. (Case 3) Freezing the trunk preserves the geometry, but cleanErr does not budge over 150k and SR is 51 — geometry
   alone is insufficient;
   accuracy maturation requires trunk updates;
3. (Case 2, partial) Low LR is the best "slow-drift" path within the MSE family (0.87/73), but its accuracy cost keeps it
   far from b; strong WD is completely ineffective (destroyed all the same) — "generalized late-training stabilization"
   has only one route,
   reducing the trunk's effective step count, and that route necessarily sacrifices accuracy; **only the MIP objective
   simultaneously decouples
   "trunk updates buying accuracy" from "trunk updates destroying geometry"**.
(b's 84 vs from-scratch MIP's 94–99: total budget 160k vs 300k, plus residue of the MSE starting point; the mechanistic
conclusion is unaffected.)

## III. Representation geometry (core of Round 3)

### CKA / kernel drift
| Comparison | CKA on | CKA Δφ | Δφ restricted to the former's servo subspace | Servo-subspace principal angles |
|---|---|---|---|---|
| RAND vs MSE | 0.546 | 0.426 | 0.407 | [86.9°, 89.0°] |
| RAND vs MIP | 0.524 | 0.394 | 0.600 | [83.2°, 88.9°] |
| **MSE vs MIP** | **0.994** | 0.922 | 0.924 | **[47.6°, 53.5°]** |

**MSE and MIP learn nearly the same representation (on-support CKA 0.994); the entire difference lies in a ~2-dim servo
subspace mutually rotated by ~50°**; neither preserves the random-kernel directions (principal angles vs RAND 83–89°).

### EXP9B adapter distillation (Verdict: not linearly transferable)
adapterR2 = **1.000** (a linear map exactly aligns MSE→MIP features on-support); at deployment SR **77** /
cross4 0.31 / cos 0.60 / rotsv 0.140 = the MSE baseline. **The entire advantage lies in off-manifold extrapolation;
on-support the representations are isomorphic.**

### Random features (A3 / EXP5)
- Static: best RAND φ₀/φτ ridge cos 0.65–0.71, never negative-definite, bias 3× that of trained models;
- Closed loop: λ∈{1e-6,1e-4,1e-2}×{φ₀,φτ} plus a headonly-SGD version, **all 0/100, cross4=1.00**.
**The architectural prior provides direction only — neither accuracy nor stability.**

### The servo-subspace ablation saga (including honest unresolved items)
- Static specificity holds: deleting top-1/2 → [2,4) operator cos 0.73→0.40/0.34; deleting null directions → operator
  intact (0.69/0.66);
- Refit closed loop: top-k dose-response (94→79→67, cross4 0.10→0.30→0.40);
  **but null-refit fares worse (52/38, cross4 0.77/0.82) — direction-specific, reproducible, and unexplained by any
  static first-order
  metric (the [1,2) band is also intact at 0.95). Unresolved; most likely explanation: interpolation accuracy in the
  deployed state (≠ training state) is damaged, consistent with the f term of the two-factor law but lacking a direct measurement.**
- Band cross-section: the [1,2) band direction is uniformly good for all models (0.91–0.96 negative-definite, no
  discriminative power);
  only the [2,4) band separates them — near-field direction is not the scarce resource; far-field negative-definiteness is.

### EXP2 view invariance (Verdict: the mechanism fails; t-pathway collapse discovered)
After training (for both MSE and MIP), φ₀≡φτ holds globally (viewdiff≈0.000, CKA=1.000); at random initialization,
servo (0.388) is no more invariant than bad (0.305). The effect of two-view does not operate through "selecting a
view-invariant subspace".

## IV. Dose of perturbation-correction supervision

### B1: τ-grid (zero input, τ × natural weight 1/(1−τ)²)
| τ | Weight | Operator cos | Negative-definite | slice-SR |
|---|---|---|---|---|
| 0.25 | 1.8× | 0.43 | ✗ (sv 1.58) | 77 |
| 0.5 | 4× | 0.45 | ✗ | 73 |
| 0.75 | 16× | 0.59 | **✓** | 84 |
| 0.9 | 100× | 0.65 | ✓ | 75 |
| 0.99 | 10⁴× | 0.66 | ✓ | 78 |

Negative-definiteness undergoes a phase transition at τ≈0.75; cos is monotone in τ. B2 (fixed τ=0.9, sweeping w∈{1..300},
two-view decoupled) is in flight.

## IVb. Direct measurement of the two-factor law (rollout drift metrics, new)

Per-step support-distance sequence: drift = mean(d_{t+1}−d_t) by band; time of first crossing of 4; maxd quantiles.

| Model | SR | cross2 | cross4 | drift(d<2) | **drift([2,4))** | SR\|crossed-4 | maxd p90 |
|---|---|---|---|---|---|---|---|
| MIP-step1 | 95 | 0.91 | 0.07 | +0.011 | **−0.43** | 29 | 3.4 |
| MIP-ridge | 94 | — | 0.10 | +0.009 | −0.40 | 40 | 3.8 |
| MSE | 71 | 0.84 | 0.32 | +0.011 | −0.004 | 9 | 6.9e4 |
| AUXDET | 64 | 0.98 | 0.50 | +0.017 | −0.09 | 28 | 4.5e4 |
| DUP100 | 54 | 0.90 | 0.47 | +0.021 | **+2.02 (repulsive)** | 2 | 1.8e4 |
| MSE early stop s10k | 51 | 1.00 | 0.56 | +0.023 | −0.07 | 12 | 2.4e5 |
| LOWLR | 50 | 0.99 | 0.58 | +0.026 | −0.08 | 14 | 1.1e5 |

**Amendments to the law**:
1. **The near-field accuracy hypothesis is partially overturned**: MSE and MIP have nearly identical drift(d<2) and
   cross2 — everyone drifts into the buffer band at the same rate;
2. **The main separator for cross4 is the buffer-band pull-back speed drift([2,4))**: MIP −0.40~−0.43 (a strongly
   attracting field),
   everything else ≥ −0.09, and the failing family is positive (DUP100 +2.0, actively repulsive) — **the corrective
   operator's main battleground is preventing arrival at the PNR within the buffer band, not rescue after crossing**;
3. drift(d<2) is a secondary factor (0.009→0.026, ordered by accuracy), explaining why cross4 decreases with training
   within the MSE family;
4. Final form of the law: cross2 ≈ accuracy; **cross4|entered-band ≈ buffer-band pull-back (geometry)**; SR|crossed-4 ≈
   far-field geometry.
   (Positive drift means are heavy-tailed and dominated by deep-space episodes; be cautious when ranking positive values
   by their means.)

### Servo ablation × drift (the null paradox resolved)
| Ablation | static [2,4) cos | drift(d<2) | drift([2,4)) | cross4 | SR |
|---|---|---|---|---|---|
| k=0 | 0.73 | +0.009 | −0.40 | 0.10 | 94 |
| top-1 | 0.40 | +0.135 | +4.3* | 0.30 | 79 |
| top-2 | 0.34 | +0.017 | +3.2* | 0.40 | 67 |
| top-3 | 0.28 | +0.028 | +0.14 | 0.74 | 34 |
| **null-1** | **0.69 (intact)** | **+0.029 (3×)** | **−0.04 (pull-back collapses by 90%)** | 0.77 | 52 |
| null-2 | 0.66 | +0.030 | −0.01 | 0.82 | 38 |

**Verdict on the null paradox: the static probe (linearized responses on off-policy DART states) ≠ the deployed vector
field (actual behavior on on-policy states).** The top directions carry the statically visible far-field directional
content; the null (small-singular-value) directions carry near-field
stabilization and buffer-band pull-back at deployment. Both classes of directions are causally necessary for the closed
loop, each with its own role. Deleting top-1 flips the buffer band from an attracting field to
a repulsive one (−0.40 → +4.3) — far more dramatic than the static cos change (0.73→0.40).

## V. Honest unresolved items / boundaries

1. ~~null-dir closed-loop fragility~~ **resolved** (§IVb: static probe ≠ deployed field; the null directions carry
   deployment pull-back);
2. ~~deployed-state accuracy lacked a direct measurement~~ **resolved** (§IVb: band-resolved drift metrics);
3. The destruction window near s25k shifts across seeds (25–50k);
4. Positive drift means are heavy-tailed (dominated by deep-space episodes); ranking among positive values should use
   medians (future improvement);
5. Why the small-singular-value feature directions carry deployment pull-back while the static [2,4) linear probe cannot
   see it — the mechanism is uncharacterized
   (anisotropy/nonlinearity of the deployed field vs linearization on DART states);
6. EXP7 (completing the abs representation), EXP9A/C (feature-matching distillation training), EXP10A/C (anchor robustness,
   phase-conditioned operators) are paused per the advisor's instruction, to be launched after the ET3 readout.

## VI. In flight
ET3 a–d (standard / MIP takeover / low-LR / strong-WD continued training + full geometry curves + final SR),
w2-trajmip (MIP 300k full-trajectory operator + layerwise), trunklr 0.1×, B2 weight sweep ×6.

## VII. Mechanistic statement for theory (current final form)

On the data manifold, MSE and MIP learn linearly isomorphic representations (CKA 0.994; linear adapter R²=1.0);
their difference is the orientation of a ~2-dim servo feature subspace. The corrective geometry emerges spontaneously
early in training (<10k) under any objective (direction near GT, negative-definite, no bad modes), but single-view MSE
undergoes a fast,
irreversible feature phase transition during the accuracy-polishing period at 25–50k: the far-field
negative-definiteness is destroyed and a rotational amplifier grows, after which the closed loop is capped at ~79 by
"crossing rate (set by accuracy) × death rate after crossing (set by geometry)". The high-weight τ-slice auxiliary view
(input content irrelevant; the transition holds for τ≥0.75 with weight ≥16×) anchors that subspace in a GT-aligned,
negative-definite position through
gradients into the shared trunk (AUX_GRAD_BLOCK proves this is necessary), allowing the extrapolation geometry to
survive the accuracy-maturation period —
a step-1 deployment therefore enjoys both at once. Open theoretical questions: why single-view accuracy polishing
necessarily rotates the servo
subspace (a characterization of the implicit bias), and the first-order mechanism by which the τ-embedding anchors that
subspace in the shared trunk.
