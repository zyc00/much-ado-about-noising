# Mechanism validation · Round 3 (causality and origin of the feature geometry)

Protocol as before (pairs [2,4), held-out seeds 21000–21100, unified badmode convention).

## Check 1: servo-subspace ablation (causal verdict, complete)

Servo feature subspace = left singular directions of the cross-covariance between δφ and G_GT δz, effective rank ≈ 2
(singular values 1.0 / 0.19 / 0.011 / …). Project out the top-k from the 2048-dim features; the ridge head is not allowed to refit:

| k | on-support R² | cos vs G_GT | closed-loop SR | cross4 |
|---|---|---|---|---|
| 0 | 1.000 | 0.73 | 94 | 0.10 |
| 1 | 0.77 | 0.20 | **0** | **1.00** |
| 2 | 0.82 | −0.02 | **0** | 1.00 |
| 3 | 0.80 | −0.03 | **0** | 1.00 |

**Deleting the single top-1 servo direction causes total escape (all 100 episodes cross PNR=4).**

**Specificity control result (unexpected)**: deleting null directions carrying almost no servo content (sval≈0.002-0.006):
| Control | onR2 | static operator cos | SR |
|---|---|---|---|
| NULL1 (delete direction 6) | 0.951 | **0.71 (operator intact)** | **1/100** |
| NULL2 (delete directions 5-6) | 0.907 | 0.68 | 0/100 |

**The ablation is not specific at the SR level**: deleting any 1-2 directions from this 6-dim action-relevant feature
family kills the closed loop, even when the static operator is intact. Specificity holds only at the operator level
(top-1/2 → cos 0.20/−0.02, null → 0.71/0.68).

**Refit version (head refit on the ablated features; a pure information-deletion test)**:

| refit ablation | onR2/chunkR2 | operator cos | SR | cross4 |
|---|---|---|---|---|
| k=0 | 1.000 | 0.73 | 94 | 0.10 |
| top-1 | 1.000 | 0.40 | 79 | 0.30 |
| top-2 | 1.000 | 0.34 | 67 | 0.40 |
| null-1 | 1.000 | **0.69 (intact)** | **52** | **0.77** |
| null-2 | 1.000 | 0.66 | **38** | 0.82 |

Readings: (i) after refit, on-support is perfect (deleting 2 of 2048 dims is lossless); the top-k series shows a
dose-response (cos 0.73→0.40→0.34, cross4 0.10→0.30→0.40), and failures are still 100% explained by crossing;
(ii) **unexpected: deleting the null directions hurts more, and direction-specifically**
(operator intact but cross4 0.77/0.82) —
the [2,4) operator probe cannot see the function of these directions. New hypothesis: **the servo mechanism is two-band —
the top directions carry far-field ([2,4)) correction, while the small-singular-value directions may carry near-field
([0,2)) stabilization**. Band-resolved static ablations
(sa-band-top/null, fit on the two bands [1,2) and [2,4)) are running, to adjudicate.
Current honest conclusion: single-direction causal attribution at the SR level fails (under both designs); what holds is
(a) directional specificity at the operator level, and (b) the dose-response of the top-k refit series.

## Check 6: random features + ridge closed loop (complete)

RAND-RIDGEHEAD: chunk-head trainR2 = 0.929 (trained models reach 1.000),
**SR = 0/100, cross2 = cross4 = 1.00**, median maxd 42.9.
**A directional prior (cos 0.70) ≠ a stable recovery representation**: random features are neither accurate (7% of
on-support variance cannot be expressed linearly → per-step bias) nor stable (indefinite + positive eigenvalue);
the closed loop escapes immediately. Combined with Check 2: what matters is not the
random kernel itself, but where training relocates the ~2-dim servo subspace.

## Check 2: kernel drift / CKA (complete)

Linear CKA, penultimate φ (t=0, zeros); RAND = fixed seed-0 untrained network:

| Comparison | CKA on-support | CKA Δφ (off-support differences) | CKA Δφ restricted to the former's servo subspace | Servo-subspace principal angles |
|---|---|---|---|---|
| RAND vs MSE | 0.546 | 0.426 | 0.407 | **[86.9°, 89.0°]** |
| RAND vs MIP | 0.524 | 0.394 | **0.600** | [83.2°, 88.9°] |
| MSE vs MIP | **0.994** | 0.922 | 0.924 | **[47.6°, 53.5°]** |

Interpretation:
1. **Strong form refuted**: both models' servo subspaces are nearly orthogonal to that of the random initialization
   (83–89°) — the post-training servo mechanism is not "random-kernel directions preserved" but a wholesale relocation
   to new directions. Both are also globally far
   from the random kernel (CKA≈0.5/0.4); this is not the lazy regime.
2. **Weak form survives**: restricted to the random servo subspace, MIP's Δφ correlates more strongly with the random kernel
   (0.600 vs 0.407) — MIP's behavior in that subspace is closer to the random prior.
3. **The most important surprise**: MSE and MIP are nearly identical on-support (CKA 0.994!), and their overall Δφ is
   also highly similar (0.922); their difference is concentrated in a ~2-dim servo subspace rotated by ~50°.
   This interlocks precisely with Check 1's "rank-2 causal subspace": **MSE and MIP learn nearly the same representation;
   closed-loop life or death hinges on where 2 of its directions point.**

## EXP-A3 / Check 6 background: random-init feature probe (complete)

| Features | cos | Negative-definite |
|---|---|---|
| RAND φ₀ | 0.69 | ✗ (+0.015) |
| RAND φτ | 0.71 | ✗ (+0.018) |
| MSE after training | 0.56–0.61 | ✓ (weak gain) |
| MIP after training | 0.73–0.74 | ≈ |

The random architectural prior comes with cos≈0.70 directional extrapolation but is indefinite; after MSE training,
cos actually drops.
Combined with CKA: MSE does not "destroy the random kernel in place"; rather, when relocating the servo subspace it moves
the directions astray
(rotational amplifier + gain shrinkage), whereas MIP relocates it to a GT-aligned position.
The closed-loop SR of random features + ridge (Check 6, `rh-rand`) is running — to adjudicate "directional prior ≠ stable representation".

## EXP-A6: operators on policy self-visited states (complete)

| | MIP | MSE |
|---|---|---|
| SR / number of off-support windows | 99 / 1237 | 71 / **2337** |
| Negative-definite | ✓ | ✗ |
| rot top-sv | 0.130 | 0.135 (the amplifier persists on self-visited states) |
| gain@bad1 | −0.008 | −0.007 |
| cos (R²) | 0.12 (0.19) | 0.42 (0.50) |

The structural discriminants transfer; on the visited distribution both models' cos drops (MIP's responses are small and
the linear fit weak;
cos is noisy on near-zero responses), so negative-definiteness / bad modes are the reliable readouts.

## Wave-4 results (GOAL batch)

### Band-resolved ablation (refit, static)
k=0 baseline: **band[1,2) cos = 0.96 negative-definite** / band[2,4) cos = 0.73.
- TOP ablation: k=1/2 → band[2,4) 0.40/0.34, while band[1,2) stays at 0.94/0.90 → **the top directions carry only far-field extrapolation**;
- NULL ablation: both bands intact (band12 0.95, band24 0.69/0.66) → the null-direction closed-loop collapse
  (cross4 0.77/0.82) is **not explained by any static first-order metric** (the near-field hypothesis is also refuted);
  an honest unresolved channel.
- k=6 (delete all; TOP=NULL converge): band12 0.76, band24 0.20.

### EXP1 early trajectory (25k short training, snapshots at 1k/5k/10k/25k)
| ckpt | cos | rot@mseamp | gain@bad1 | CKArand_servo |
|---|---|---|---|---|
| MSE s1000 | 0.78 | 0.009 | −0.011 | 0.069 |
| MSE s5000 | 0.83 | 0.010 | −0.014 | 0.052 |
| MSE s10000 | **0.84** | 0.006 | −0.013 | 0.040 |
| MSE s300000 (reference) | 0.56–0.61 | **0.152** | +0.003 | — |
| ZEROIN s1000–10000 | 0.73–0.79 | 0.014–0.018 | −0.012~−0.014 | ~0.07 |

**Key reversal: within its first 10k steps, pure MSE already forms a corrective geometry with cos 0.84, no rotational
amplifier, and negative bad1;
it is late training that destroys it (destructive late-phase feature learning) — not an inability to learn it.**
The servo subspace leaves the random position within 1k steps (CKArand_servo ≈ 0.05) — final version of the
"random-kernel preservation" narrative:
**the corrective structure emerges spontaneously early in training; MIP's role is to prevent late-phase destruction,
whereas MSE's late phase replaces it with bad modes.**
(The full 25k–300k curves for mse_traj/mip_traj are running and will give the destruction timeline.)

### EXP2 view invariance (Verdict: hypothesis refuted, but t-pathway collapse discovered)
| Model | viewdiff(servo) | viewdiff(bad) | viewdiff(all) | viewCKA(all) |
|---|---|---|---|---|
| RAND | 0.388 | 0.305 | 1.368 | 0.998 |
| MSE after training | 0.000 | 0.000 | 0.000 | 1.000 |
| MIP after training | 0.001 | 0.001 | 0.002 | 1.000 |

Training (regardless of objective) globally collapses φ₀ and φτ into identity (the t-embedding becomes inert under zeros input);
at random initialization, servo is no more view-invariant than bad → the "invariant-subspace selection" mechanism fails.

### EXP5 random-feature closed loop (complete)
λ∈{1e-6,1e-4,1e-2} × {φ₀,φτ}: **all 0/100, cross2=cross4=1.00** (best static cos 0.65,
never negative-definite, bias 3× that of trained models). Directional prior ≠ stable representation — settled.

### EXP9B adapter distillation (Verdict: not linearly transferable)
adapterR2 = **1.000** (a linear map sends MSE features exactly to MIP features on-support),
chunkR2 = 1.000; but at deployment: operator cos 0.60, rotsv 0.140, **SR 77, cross4 0.31 — no different from the MSE
baseline (71, 0.32)**. **The two representations are linearly isomorphic on-support; MIP's entire advantage lies in
off-manifold extrapolation, and cannot be transferred through a map fit on-support.**

### head0 baseline (with distance tracking)
MSE: 71%, cross4 0.32, SR|stay=100; MIP-step1: 95%, cross4 0.07, SR|stay=100.

## ET batch (after the early-trajectory reversal)

### ET1 + trajectory probe: MSE destruction timeline (complete)
| step | operator cos (pen ridge) | rotsv | gain@bad1 | closed-loop SR | cross4 |
|---|---|---|---|---|---|
| 1k | 0.78 | 0.090 | −0.011 | 21 | 0.87 |
| 5k | 0.83 | 0.082 | −0.014 | 46 | 0.70 |
| 10k | 0.84 | 0.078 | −0.013 | 51 | 0.56 |
| 25k | 0.83 (early run) / 0.66 (traj run 27k) | 0.012/0.063 | −0.014/−0.011 | 41 | 0.65 |
| 51k | **0.49** | **0.124** | ~0 | 44 | 0.58 |
| 76k–300k | 0.46–0.56 (stable) | 0.10–0.14 | ~0/+ | 63→71→**79** | 0.39→0.34→**0.22** |

**The destruction occurs in the ~25k–50k window, fast and irreversible (after 50k the model sits in the destroyed state).**
Yet closed-loop SR rises monotonically instead: cross4 is dominated by on-support accuracy (error 0.0086→0.0049),
and the accuracy gains outweigh the operator loss; the destroyed operator caps MSE at 79 (0.22 crossing × ~95% death after crossing).

### EXP6D early-stopping verdict: no rescue
Best early-stopping SR = 51 (s10k) « the converged 79. The s10k operator has cos 0.84 negative-definite, yet SR|crossed is only 12 —
**the early direction is correct but the gain is weak and accuracy poor (median maxd after crossing 6–107, deep-space drift);
a good static direction ≠ closed-loop rescuability.**
Under single-view MSE, accuracy and extrapolation cannot be had together: by the time accuracy is in place, extrapolation is already destroyed.

### ET4: MIP trajectory maintenance (closed loop)
s25k → 82 (cross4 0.20), s100k → **96 (0.07, SR|crossed 43)**, s300k → 95.
The operator stays at 0.73 throughout, and clean error is simultaneously better than MSE's (ET5) — both curves arrive
together; no tradeoff.

### ET5: clean error vs geometry timeline
MSE clean p50: 0.0109 (10k)→0.0086 (25k)→0.0062 (50k)→0.0049 (200k), gap≈1.0;
the destruction window (25–50k) coincides with the period of marginal error improvement 0.0086→0.0062 — the feature drift
is not classical overfitting.
MIP's clean error is lower at every time point (100k: 0.0032 vs 0.0056).

### Layerwise (300k MSE): the destruction is in the trunk, not the encoder
enc/cond remain negative-definite at cos 0.68; down0–pen are all indefinite at 0.41–0.61.
The encoder is never destroyed; the bad modes grow in the deep trunk — consistent with the layerwise comparison against
MIP (deep trunk 0.78).

## EXP3 Verdict: the shared trunk is the locus of the mechanism (complete)

| Variant | head0 SR | Operator |
|---|---|---|
| Normal MIP | 94–99 | 0.73 negative-definite |
| AUX_GRAD_BLOCK (aux gradients enter only final_conv) | **64** (2-step 59) | cos 0.91 **indefinite** (+0.022) |
| NO_SHARED_TRUNK (≡ MSE + standalone denoiser) | 71–79 | indefinite |

The aux gradients must enter the shared trunk; training only the late head is ineffective and even harmful. AUXDET is
also the third instance of
"high cos ≠ stability" (0.91 indefinite → 64; jacreg-1e4 0.92 indefinite → 24; TAU99 0.84 indefinite → 95 with other
compensation).

## The lazy family, full table (EXP6, complete)

| Variant | SR | Operator cos | Negative-definite | Notes |
|---|---|---|---|---|
| MSE baseline | 71–79 | 0.5–0.6 | ✗ | cross4 0.22–0.32 |
| headonly (frozen random trunk) | **0** | **0.82** | ✓ | best geometry, worst accuracy |
| lowlr (lr/10) | 50 | **0.77** | ✓ | good geometry, effectively undertrained |
| early stopping (s10k) | 51 | 0.84 | ✓ | same as above |
| frozen s10k trunk + ridge | 39–53 | 0.91/0.86 (two bands) | — | same as above |
| wd×1000 | 60 | 0.51 | ✗ | gains on neither front |
| dup100 without clipping (B4) | 52 | 0.53 | ≈ | clipping ruled out |
| trunklr 0.1× (E) | pending | | | |

In the geometry–accuracy plane, the entire lazy family falls below MSE's Pareto frontier or on the wrong side of it;
the only intervention that improves both simultaneously is the τ-weighted auxiliary view. Band-x2 cross-section:
the [1,2) band direction is uniformly good for all models
(0.91–0.94) and carries no discriminative power → **cross4 is determined by per-step absolute accuracy in the deployed
state (ET5 errors correspond monotonically
to cross4), not by the direction in any band; direction (negative-definiteness) determines only SR|crossed.**

## In-flight matrix

| Experiment | Job | ETA |
|---|---|---|
| Check 1 specificity control (null-dir k=1/2) | sa-null1/2 | ~1h |
| Check 6 random-feature closed-loop SR | rh-rand | ~1.5h |
| Check 5 AUX_GRAD_BLOCK (A1v2) training + full evaluation suite | tr-auxdet / w-auxdet | ~3h |
| Check 4 lazy-MSE: lr/10, wd×1000, frozen-trunk | tr-lowlr/wd/headonly + waiters | ~3h |
| Check 3 trajectory probe (25k snapshots × cos/badmode/U_rand principal angles + layerwise) | w2-trajmse/mip | ~3h |
| B1 τ-grid zeroin ×4 | tr-zt* + w-zt* | ~3h |
| B2 two-view zero-input weight sweep ×6 | tr-zaw* + w-zaw* | ~3h |
| B4 dup100 without clipping | tr-dupnoclip + w-dupnc | ~3h |
