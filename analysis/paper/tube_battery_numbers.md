# Clean-tube battery + PD-chunk wave-1 (numbers only)

All SR: held-out seeds 21000+, official train-harness eval, n=100.
Recovery ratio = pdrec2 mode A (manifold-normal push, median r/r_GT over m=1/2/3).

## T1. EXP-1 clean-tube inward signal (200 clean demos, phase-windowed kNN tube coords)

| radius bin | N | rad_p50 | r_clean_p50 | r_p90 | frac_r>0 | |da|_p50 |
|---|---|---|---|---|---|---|
| 0-50% | 1999 | 0.22 | +0.0017 | +0.096 | 0.61 | 0.035 |
| 50-80% | 1200 | 0.48 | +0.0019 | +0.093 | 0.64 | 0.044 |
| 80-95% | 600 | 0.73 | +0.0017 | +0.069 | 0.62 | 0.042 |
| 95-100% | 200 | 1.12 | +0.0026 | +0.087 | 0.56 | 0.073 |

G_tube fit: R2cv=0.021; poseig=[-0.0051,-0.0023,-0.0009] (negdef but 20-50x smaller than G_GT k~0.1);
k_iso=0.0028; aniso=0.54; cos(G_tube,G_GT)=0.22, cos(G_tube,G_MSE)=0.25, cos(G_tube,G_MIP)=0.25.
Stage-binned: R2cv 0.007-0.156; cosGT -0.17..0.47.

## T2. EXP-2 held-out clean first-action error by tube radius (20kB demos 5000-5059)

err_p50 (raw pos units); rad q50=0.35 q80=0.56:

| model | 0-50% | 50-80% | 80-100% edge | edge/center |
|---|---|---|---|---|
| MIP-300k | 0.0021 | 0.0018 | 0.0021 | 1.00 |
| MSE-300k | 0.0027 | 0.0030 | 0.0036 | 1.33 |
| ZEROIN | 0.0027 | 0.0031 | 0.0036 | 1.33 |
| PDCHUNK (k0.03 h16) | 0.0046 | 0.0044 | 0.0054 | 1.17 |
| s10k | 0.0077 | 0.0097 | 0.0124 | 1.61 |
| RAND | 0.890 | 0.974 | 1.027 | - |

r_resid_p50 ~= -0.0000..-0.0005 for all trained models (no systematic inward/outward bias on clean states).

## T3. EXP-3 structured Jacobian (finite-diff, eef-pos; rJ = inward gain; aniso = Ln/Lt)

| model | center rJ (aniso) | edge rJ (aniso) | off1.5 rJ | off3.0 rJ | frac_rJ>0 @off3 |
|---|---|---|---|---|---|
| MIP-300k | +0.046 (2.35) | +0.027 (1.09) | +0.036 | +0.038 | 0.89 |
| PDCHUNK | +0.030 (0.99) | +0.030 (0.99) | +0.030 | +0.030 | 1.00 |
| MSE-300k | +0.015 (2.74) | +0.006 (1.04) | +0.012 | +0.012 | 0.84 |
| s10k | +0.017 (2.56) | +0.010 (1.20) | +0.012 | +0.015 | 0.78 |
| ZEROIN | +0.014 (2.19) | +0.007 (1.22) | +0.015 | +0.014 | 0.79 |
| RAND | -0.003 | -0.000 | -0.000 | +0.000 | 0.50 |

## T4. EXP-4 G alignment at tube edge (m=2 normal push)

| model | cos(da, G_tube d) | cos(da, G_GT d) | gain/GT |
|---|---|---|---|
| PDCHUNK | +0.94 | +0.55 | 0.22 |
| MIP-300k | +0.70 | +0.40 | 0.52 |
| MSE-300k | +0.57 | +0.21 | 0.27 |
| s10k | +0.54 | +0.37 | 0.30 |
| ZEROIN | +0.47 | +0.17 | 0.44 |
| RAND | +0.20 | +0.11 | 0.05 |

## T5. PD-chunk wave-1 (s10k+150k) k x H' x target dose-response

| variant | prescribed/measured recovery ratio | SR | cross4 | SR|cross4 |
|---|---|---|---|---|
| k0.02 H'16 | 0.20 | 82 | 0.26 | 31 |
| k0.025 H'16 | 0.25 | 72 | 0.30 | 7 |
| k0.03 H'16 | 0.30 | 87 | 0.15 | 13 |
| k0.04 H'16 | 0.39 | 93 | 0.11 | 36 |
| k0.02 H'8 | 0.20 | 86 | 0.19 | 26 |
| k0.025 H'8 | 0.25 | 83 | 0.19 | 11 |
| k0.03 H'8 | 0.30 | 78 | 0.24 | 8 |
| k0.04 H'8 | 0.39 | 78 | 0.27 | 19 |
| k0.03 H'4 | 0.30 | 89 | 0.13 | 15 |
| k0.03 H'2 | 0.30 | 67 | 0.35 | 6 |
| PDTGT=gt H'16 (full GT field) | 1.00 | 79 | 0.21 | 0 |
| PDTGT=gtdamped x0.5 H'16 | 0.50 | 93 | 0.11 | 36 |
| (ref) k0.03 H'16 orig run | 0.30 | 87 | 0.13 | 0 |

All variants: SR|stay=100, cleanErr held p50 0.0053-0.0076, gap 0.94-0.97x (precision healthy),
EXCEPT k0.03 H'2: cleanErr held p50 0.0124 (~2x elevated, slot-conflict regime) -> SR 67.
H' dose (k=0.03 fixed): H'2=67, H'4=89, H'8=78, H'16=87 (H'2 clearly bad, rest noisy).
pdrec2 measured ratios exactly match prescribed k/0.1 in every variant (frac_r>0=1.00, textbook field).

## T6. SR vs measured recovery ratio (cross-method, all from-scratch or s10k+150k)

| model | recovery ratio (m=1/2/3 median) | SR |
|---|---|---|
| RAND | 0.00 | - |
| MSE-cont (s10k+150k) | 0.13-0.23 | 62 |
| MSE-300k | 0.23-0.44 | 71 |
| PDC k0.02 | 0.20 | 82-86 |
| PDC k0.03 | 0.30 | 78-89 |
| PDC k0.04 H'16 | 0.39 | 93 |
| PDC gtdamped 0.5 | 0.50 | 93 |
| PDC gt full | 1.00 | 79 |
| MIP-cont (takeover) | 0.33-0.40 | 84 |
| MIP-300k | 0.45-0.58 | 95 |

## T7. From-scratch PD-chunk (300k, no s10k start) — filling in

| variant | SR | cross4 | maxd p95 | cleanErr held p50 |
|---|---|---|---|---|
| pds_k0.015_h16 | 96 | 0.06 | 4.74 (no deep escapes) | 0.0042 |
| pds_k0.02_h16 | 94 | 0.11 | 48.2 | |
| pdc_scratch (k0.03 h16 flagship) | 92 | 0.12 | 6067 | 0.0042 |
| pds_near_k0.03 | 90 | 0.10 | 11669 | 0.0038 |
| pds_near_k0.02 | 93 | 0.11 | 27846 | 0.0036 |
| pds_k0.03_l300 | 94 | 0.12 | 51.9 | 0.0052 |
| pds_k0.04_h8 | 96 | 0.07 | 5.85 (no deep escapes) | 0.0046 |
| pds_k0.02_h8 | 89 | 0.18 | 59.6 | 0.0046 |
| pds_k0.03_h8 | 91 | 0.17 | 53.7 | 0.0044 |
| pds_k0.03_l30 | 68 | 0.39 | 43560 | 0.0038 |
| pds_k0.05_h16 | 80 | 0.26 | 11138 | 0.0042 |
| pds_k0.04_h16 | 85 | 0.19 | 40654 | 0.0041 |

BATCH COMPLETE (12/12).
pdrec2 measured recovery ratio (mode A) again == prescribed k/0.1 exactly, frac=1.00, for every variant:
k0.015->0.15, k0.02->0.20, k0.03->0.30, k0.04->0.39, k0.05->0.49 (h8 identical to h16).
SR vs measured ratio FROM SCRATCH: 0.15->96, 0.20->89-94, 0.30->90-92, 0.39->85/96, 0.49->80
(optimum ratio ~0.15-0.2 from scratch, vs 0.39-0.50 when starting from damaged s10k -- dose depends on start point).
k dose (h16, lambda=100): 0.015->96, 0.02->94, 0.03->92, 0.04->85, 0.05->80 (monotone decline past 0.015-0.02).
k dose (h8, lambda=100): 0.02->89, 0.03->91, 0.04->96 (opposite trend vs h16).
lambda dose (k0.03 h16): 30->68, 100->92, 300->94.
near-band: k0.02->93, k0.03->90 (mixed-band not required).
From-scratch band 85-96 for lambda>=100 (11/12); sole sub-80: l30=68.
vs baselines from scratch: MIP 94-99, MSE 71.
(its [2,4) drift dd_mean +0.014 vs -0.03..-0.11 for all others -> buffer-band pullback lost).
Zero-deep-escape variants: k0.015_h16 (maxd p95 4.7), k0.04_h8 (5.9).

## T9. EXP-6 tube-width data ablation (MSE on modified datasets)

| dataset | SR | cleanErr held p50 | pdrec2 ratio m=3 (frac) | note |
|---|---|---|---|---|
| edgeheavy3000 (all + edge500 x3) | 76 | 0.0036 | 0.27 (0.82) | vs MSE-300k baseline 71 |
| center1000x2 (narrow tube) | 3 | 0.0437 (12x worse) | 0.21 (0.77) | cross4=1.00; both bands dd_mean>0 (no pullback anywhere). CONFOUND NOTE: heldErr 12x worse on standard held demos means the center-1000 subset does not cover the eval initial-state distribution -- narrow tube = narrow initial coverage, not a pure width manipulation |
| MIP edgeheavy3000 | 92 | 0.0024 | 0.38 (0.89) | vs MIP-300k baseline 95: no gain |
| MIP center1000x2 | 8 (cross4=0.98) | 0.0439 (12x worse, same confound as MSE center) | 0.39 (0.86) | MSE center 3, MIP center 8: both collapse under the init-coverage confound |

Edge-oversampling asymmetry: MSE 71->76 (+5), MIP 95->92 (-3). The MIP-MSE gap shrinks
from 24 to 16 points under edge-heavy data -- direction consistent with data-sourced
signal, magnitude far from closing the gap.

Note: from-scratch cleanErr (0.0041-0.0042) beats s10k+150k wave-1 (0.0053-0.0076).
Flagship pdc_scratch milestones: s100999 cleanErr 0.0056, s200999 0.0053; final SR pending.

## T8. EXP-5 one-trajectory (demo_0 x2000, 300k)

| model | SR | cleanErr held p50 | pdrec2 ratio_p50 m=3 (frac) |
|---|---|---|---|
| ONETRAJ-MSE | 0/100 (cross4=1.00, fc4_p50=15) | 0.3746 | 0.25 (0.77) [OOD-anchor artifact] |
| ONETRAJ-MIP | 0/100 (cross4=1.00, fc4_p50=15 -- same init-coverage collapse as OT-MSE; SR uninformative) | 0.3197 | see T8b |
| ONETRAJ-PDC | (waiter pending; SR expected 0 for same reason) | | see T8b |

### T8b. DECISIVE: demo_0-anchored recovery probe (matched normalizer, same anchors/dirs)

56 anchors ON demo_0 (the one-traj training support), eef-pos pushes orthogonal to
trajectory tangent, ratio = r/r_GT p50 at m=1/2/3 (frac_r>0):

| model | m=1 | m=2 | m=3 | frac | cleanErr demo0 |
|---|---|---|---|---|---|
| FULL-MIP (2k demos) | 0.28 (0.84) | 0.27 (0.86) | 0.30 (0.88) | | 0.0011 |
| FULL-MSE (2k demos) | 0.05 (0.71) | 0.10 (0.75) | 0.12 (0.75) | | 0.0045 |
| OT-MIP (1 demo) | 0.05 (0.71) | 0.06 (0.73) | 0.09 (0.77) | | 0.0002 |
| OT-MSE (1 demo) | 0.03 (0.59) | 0.05 (0.61) | 0.10 (0.66) | | 0.0002 |

ONE-TRAJECTORY MIP LOSES THE ENTIRE RECOVERY COMPONENT: OT-MIP (0.05-0.09) ==
OT-MSE (0.03-0.10) == FULL-MSE level, vs FULL-MIP 0.27-0.30. Both OT models fit
demo_0 perfectly (cleanErr 0.0002), so this is not underfitting.
(A first run with mismatched normalizer showed OT-MIP 0.34 -- artifact, retracted;
matched-normalizer numbers above are the valid ones. LIKEWISE the auto-waiter pdrec2
lines "ONETRAJ-mip ratio 0.37 / ONETRAJ-pdc 0.33" carry BOTH artifacts (2k-file
normalizer + OOD anchors from other demos) -- do not cite them; T8b supersedes.)
OT-PDC control: ratio 0.32/0.32/0.32, frac 1.00, cleanErr 0.0002 -- the explicit PD
objective creates its full prescribed field even from ONE trajectory (it manufactures
its own perturbation data), while MIP's implicit mechanism does not.

Synthesis (numbers): recovery source is neither pure-data nor pure-objective.
- Data variation alone (FULL-MSE): weak extrapolated recovery 0.10-0.13, destroyed
  at 25-50k (ET timeline), SR 71.
- MIP objective alone (OT-MIP, zero tube width): recovery absent (0.06), == OT-MSE.
- MIP objective x data variation (FULL-MIP): 0.27-0.30, frac 0.86, SR 95.
- Explicit synthetic-perturbation objective (PD): creates prescribed field with OR
  without data variation (0.32 from one traj; 0.30 from 2k), SR 87-96.
MIP's recovery is data-sourced but objective-gated: the tube signal is the raw
material; the two-view objective determines whether it survives into off-support
extrapolation (V2/V3 84-96 vs V5 52-68; ET destruction window).

## T3b. EXP-2 v2 residual metrics (sampler-based inference, held-out clean)

r_hat vs r_clean (edge bin 80-100%): ratio / sign_agree / corr(rad,r_hat):

| model | err_p50 edge | errPos/errRot edge | r_hat_p50 | ratio r_hat/r_clean | sign_agree | corr |
|---|---|---|---|---|---|---|
| MIP300k | 0.0019 | 0.0014/0.0010 | +0.0013 | 0.85 | 0.89 | -0.09 |
| MSE300k | 0.0046 | 0.0036/0.0019 | +0.0014 | 0.98 | 0.82 | -0.08 |
| PDSk015 | 0.0052 | 0.0041/0.0021 | +0.0021 | 1.44 | 0.86 | -0.11 |
| PDCHUNK | 0.0068 | 0.0054/0.0026 | +0.0027 | 1.83 | 0.84 | -0.09 |
| ZEROIN | 0.0086 | 0.0070/0.0037 | +0.0021 | 1.40 | 0.81 | -0.11 |
| s10k | 0.0148 | 0.0124/0.0054 | +0.0024 | 1.63 | 0.76 | -0.10 |
| JR1e4 | 0.0292 | 0.0218/0.0066 | +0.0020 | 1.37 | 0.68 | -0.14 |
| FLOW | 0.0255 | 0.0220/0.0074 | +0.0021 | 1.44 | 0.71 | -0.09 (stoch act_0 inflates err) |
| JR1e3 | 0.0644 | 0.0431/0.0099 | +0.0008 | 0.56 | 0.53 | -0.14 |
| RAND | 1.18 | 1.03/0.54 | -0.179 | -122 | 0.46 | +0.08 |

Numeric facts: every trained model reproduces the tiny clean tube-edge inward residual
(ratio 0.5-1.9, sign_agree 0.68-0.90); corr(radius, r_hat) ~ 0 for all. The clean-state
residual signal does NOT separate methods; only the off-support response does.

## T4b. EXP-3 v2 structured Jacobian (sampler-based; rJ/GTk = inward gain / GT k=0.1)

rJ/GTk by region (center / edge / off1.5 / off3.0), frac_rJ>0 at off3.0:

| model | center | edge | off1.5 | off3.0 | frac@off3 | Lt edge | Ln_p90 edge |
|---|---|---|---|---|---|---|---|
| MIP300k | 0.38 | 0.41 | 0.23 | 0.33 | 0.85 | 0.101 | 0.232 |
| FLOW ns=10 (corrected) | 0.34 | 0.28 | 0.48 | 0.43 | 0.89 | 0.065 | 0.265 |
| PDCHUNK | 0.30 | 0.30 | 0.30 | 0.30 | 1.00 | 0.030 | 0.031 |
| PDSk015 | 0.15 | 0.15 | 0.15 | 0.15 | 1.00 | 0.015 | 0.016 |
| ZEROIN | 0.15 | 0.09 | 0.20 | 0.12 | 0.80 | 0.039 | 0.244 |
| s10k | 0.24 | 0.11 | 0.13 | 0.14 | 0.76 | 0.037 | 0.156 |
| MSE300k | 0.12 | 0.11 | 0.13 | 0.13 | 0.85 | 0.046 | 0.244 |
| JR1e4 | 0.07 | 0.05 | 0.07 | 0.11 | 0.77 | 0.021 | 0.097 |
| JR1e3 | 0.02 | 0.01 | 0.01 | 0.02 | 0.66 | 0.009 | 0.037 |
| RAND | -0.03 | -0.00 | -0.01 | -0.00 | 0.49 | 0.009 | 0.012 |

GALIGN2 (edge, m=2): cos_Gtube / cos_GGT / gain_vs_GT:
PDCHUNK 0.94/0.58/0.21; PDSk015 0.94/0.58/0.10; MIP 0.73/0.46/0.53; FLOW 0.67/0.36/0.60;
s10k 0.61/0.36/0.35; MSE 0.58/0.21/0.32; ZEROIN 0.50/0.23/0.54; JR1e4 0.39/0.12/0.17;
JR1e3 0.20/0.07/0.06; RAND 0.23/0.04/0.05.

## T1b. EXP-1 coordinate variants

COORD=pos (eef-pos 3D only): bin 0-50% DEGENERATE (local cloud full-rank, no normal);
50-80% r=+0.0016 (frac 0.67); 80-95% r=+0.0105 (0.81); 95-100% r=+0.0283 (0.81).
Linear operator R2=0.000, not negdef -- signal exists at extreme edge but is NOT a linear field.
COORD=posquat (dims 44:51): r_p50 +0.0004..+0.0065, frac 0.53-0.69, R2=0.000, not negdef.

## T10. EXP-7 structured Jacobian vs SR (historical SR; re-verification running)

| model | SR | r/GT (pdrec2 m=3) | rJ/GTk off-support (v2) | Lr center (total-J proxy) | Lt edge |
|---|---|---|---|---|---|
| PDS k0.015 (scratch) | 96 | 0.15 | 0.15 (frac 1.00) | 0.015 | 0.015 |
| MIP-300k | 95 | 0.58 | 0.23-0.33 (frac 0.85) | 0.116 | 0.101 |
| FLOW ns=10 | 95 (cross4 0.08, maxd p95 23; 56 at ns=1 was a 1-step artifact) | 0.54 (approx probe) | 0.28-0.48 (frac 0.89-0.93) | 0.122 | 0.065 |
| PDCHUNK k0.03h16 | 87 | 0.30 | 0.30 (frac 1.00) | 0.030 | 0.030 |
| ZEROIN | 75 (hist) | not measured | 0.12-0.20 (frac 0.80) | 0.078 | 0.039 |
| MSE-300k | 71 | 0.44 | 0.13 (frac 0.85) | 0.093 | 0.046 |
| JR1e4 (explicit J-shrink) | 24 (hist) | 0.02-0.03 | 0.07-0.11 (frac 0.77) | 0.031 | 0.021 |
| JR1e3 | 1 (hist) | 0.03 | 0.01-0.02 (frac 0.66) | 0.011 | 0.009 |
| RAND | - | 0.00 | -0.01 (frac 0.49) | 0.010 | 0.009 |

Numeric facts:
- Total response norm does NOT predict SR: PDSk015 (L=0.015, smallest of trained) SR 96 while
  MIP (L=0.116, largest) SR 95; MSE in between (0.093) SR 71.
- Off-support inward gain rJ + its reliability (frac_rJ>0) orders the table almost perfectly:
  1.00x0.15 (96), 0.85x0.3 (95), 0.83x0.45 (93), 1.00x0.30 (87) at top; 0.13 (71), 0.07 (24), 0.01 (1) below.
- Explicit total-J shrinkage (JR) removes precisely the inward gain (0.13 -> 0.07 -> 0.01) as SR
  collapses 71 -> 24 -> 1, while clean err also degrades (0.005 -> 0.023 -> 0.061).

## T11. CONDITIONAL tube coordinates (plan-conditioned recovery signal)

4000 anchors, 200 clean demos. stateResid in z-units; r_clean = inward comp of action
residual vs residual direction (pos); edge = top-20% radius.

| center type | stateResid p50 | radius/A | actResid p50 | r_clean p50 (frac>0) | r_edge p50 (frac) |
|---|---|---|---|---|---|
| A phase-mean (old) | 2.586 | 1.00 | 0.065 | +0.0033 (0.62) | +0.0053 (0.63) |
| B1 obj-pose kNN | 0.692 | 0.27 | 0.042 | +0.0022 (0.62) | +0.0041 (0.65) |
| B2 eef-pose kNN | 2.334 | 0.90 | 0.015 | +0.0010 (0.60) | +0.0011 (0.62) |
| B4 full-53 kNN | 0.673 | 0.26 | 0.040 | +0.0018 (0.62) | +0.0033 (0.62) |
| C own-demo smooth | 1.110 | 0.43 | 0.118 | +0.0182 (0.75) | +0.0497 (0.70) |
| D ridge h(p,xi) | 2.191 | 0.85 | 0.455 | +0.1582 (0.78) | +0.1885 (0.75) |

D: stateVE=0.828 actionVE=0.755 vs global mean; CAVEAT actResid p50=0.455 -- linear h_a
badly underfits the action, residual contains real action structure; r for D not trustworthy.
E scripted-waypoint coordinate: NOT AVAILABLE (no waypoint targets in hdf5 export).
Window sensitivity: A stable (0.99/1.04 at W=0.006/0.025); B4 0.31/0.25.

Numeric answers:
- 73-74% of the old "tube radius" is legitimate plan variation (B1/B4 conditioning: 0.27/0.26x).
  Conditioning on object pose (B1) alone achieves nearly all of the shrinkage; eef pose (B2) almost none.
- Measured against each demo's OWN plan (C), clean data DOES contain an inward signal an order
  of magnitude larger than the unconditional estimate: r=+0.018 overall, +0.050 at edge
  (frac 0.70-0.75), vs +0.002-0.003 in A/B coordinates.

## T12. Tube projection diagnostics (conditional B1 coords: phase + object-pose kNN48)

N=2630 held states: dperp_p50=0.358 p90=0.718; dtang_p50=0.517; NN dist p50=0.774;
phase mismatch p50=0.0058. K sweep dperp_p50: K5=0.561, K10=0.475, K20=0.384, K50=0.361
(stable K>=20). EXP-7 puredart states N=250.

## T13. Action closeness to tube-edge action (D_edge / D_excess, raw units)

off[2,4) row (worst region), p50 / p90:

| model | Dtot | Ledge p90 | Dexc[Gtube] | Dexc[GT50] | SR |
|---|---|---|---|---|---|
| JR1e3 | 0.111 / 0.404 | 0.110 | 0.120 / 0.338 | 0.304 / 0.522 | 1 |
| JR1e4 | 0.147 / 0.487 | 0.131 | 0.150 / 0.470 | 0.297 / 0.583 | 24 |
| s10k | 0.219 / 0.729 | 0.193 | 0.189 / 0.621 | 0.343 / 0.684 | - |
| MIP | 0.235 / 0.626 | 0.168 | 0.208 / 0.598 | 0.336 / 0.680 | 95 |
| MSE | 0.237 / 0.760 | 0.207 | 0.224 / 0.734 | 0.353 / 0.792 | 71 |
| PDC | 0.238 / 0.786 | 0.208 | 0.210 / 0.734 | 0.342 / 0.799 | 87 |
| PDS015 | 0.248 / 0.703 | 0.187 | 0.226 / 0.624 | 0.342 / 0.752 | 96 |
| FLOW10 | 0.259 / 0.699 | 0.186 | 0.247 / 0.634 | 0.334 / 0.701 | 95 |
| ZEROIN | 0.283 / 0.962 | 0.267 | 0.241 / 0.982 | 0.361 / 0.965 | 75 |
| RAND | 1.165 / 1.902 | 0.506 | 1.161 / 1.895 | 1.247 / 1.908 | - |

p50 nearly identical for MIP/MSE/PDC/PDS (0.235-0.248); separation is in p90 tails
(MIP 0.626 vs MSE 0.760 vs ZEROIN 0.962) and JR sits CLOSEST to a_edge yet fails worst.

## T14. eps-sweep FD Lipschitz at off[1,2) (conditional-normal frame; note: this normal
is object-dim dominated, so rJ here is small for ALL models -- direction semantics differ
from the eef-pos normals of T4b)

At eps=0.1: Lrand p50 / Lt / Lbad / Lrot / rJ:

| model | Lrand | Lt | Lbad | Lrot | rJ |
|---|---|---|---|---|---|
| MIP | 0.149 | 0.054 | 0.164 | 0.063 | +0.0017 |
| FLOW10 | 0.131 | 0.028 | 0.113 | 0.103 | +0.0086 |
| s10k | 0.193 | 0.038 | 0.128 | 0.056 | +0.0058 |
| MSE | 0.113 | 0.022 | 0.115 | 0.077 | +0.0024 |
| ZEROIN | 0.116 | 0.039 | 0.122 | 0.076 | +0.0049 |
| PDC | 0.106 | 0.034 | 0.030 | 0.064 | +0.0014 |
| PDS015 | 0.110 | 0.033 | 0.015 | 0.057 | +0.0039 |
| JR1e4 | 0.023 | 0.019 | 0.034 | 0.021 | +0.0029 |
| JR1e3 | 0.014 | 0.012 | 0.020 | 0.014 | +0.0005 |
| RAND | 0.037 | 0.013 | 0.020 | 0.016 | -0.0055 |

Lipschitz estimates eps-stable (0.05-0.5 within ~20%). PD-priors uniquely suppress Lbad
(0.015-0.030 vs 0.11-0.16 for everything else trained).

## T15. Extrapolation curves from s_proj + EXP-7 puredart projection

CAVEAT (EXTRAP): normal dir points from s_proj toward the real held state, so small-m
points are near-support; r/GT only becomes meaningful by m=3-4.
m=4: r/GT MIP 0.03, MSE 0.06, PDC 0.12, PDS 0.07, FLOW 0.12, ZEROIN 0.07, JR1e4 0.08, RAND -0.05.
Dedge(m=4): JR1e3 0.044 < RAND 0.047 < JR1e4 0.124 < MSE 0.171 ~ s10k 0.172 ~ PDS 0.181 ~
PDC 0.184 ~ MIP 0.185 < FLOW 0.208 < ZEROIN 0.239.

PUREDART projection onto delta_GT (band [2,4), N=49):

| model | projratio p50 | orth_err p50 / p90 | sign_agree |
|---|---|---|---|
| MIP | -0.02 | 0.089 / 0.952 | 0.39 |
| JR1e4 | -0.01 | 0.076 / 0.883 | 0.45 |
| s10k | -0.06 | 0.121 / 0.948 | 0.35 |
| PDS015 | +0.04 | 0.178 / 0.913 | 0.57 |
| ZEROIN | -0.05 | 0.228 / 0.959 | 0.39 |
| PDC | -0.03 | 0.230 / 0.927 | 0.45 |
| MSE | -0.01 | 0.240 / 1.066 | 0.43 |
| FLOW10 | +0.05 | 0.091 / 1.021 | 0.65 |
| JR1e3 | +0.07 | 0.295 / 0.915 | 0.59 |
| RAND | +0.18 | 0.884 / 1.193 | 0.73 (meaningless, huge orth) |

Median projection ratio ~0 for ALL models (no G_GT-proportional component at median on
real puredart states); MIP/FLOW have the smallest orthogonal deviation from the
a_edge + span(delta_GT) plane (0.089/0.091 vs MSE 0.240, 2.7x).

## T16. Autograd Jacobian sanity (30 clean anchors, deployment map f(0,0,s))

| model | op | Fro | pos | rot | Jtan | Jnorm | Jbad | Jrotamp | FD-AG corr | FD/AG |
|---|---|---|---|---|---|---|---|---|---|---|
| MSE | 1.135 | 1.193 | 0.956 | 0.475 | 0.079 | 0.040 | 0.138 | 0.037 | 0.99 | 0.99 |
| MIP | 1.550 | 1.640 | 1.253 | 0.487 | 0.067 | 0.045 | 0.203 | 0.025 | 0.99 | 1.02 |
| PDC | 1.171 | 1.239 | 0.995 | 0.483 | 0.068 | 0.033 | 0.030 | 0.047 | 1.00 | 1.04 |
| PDS015 | 1.366 | 1.405 | 1.089 | 0.379 | 0.074 | 0.037 | 0.015 | 0.025 | 0.99 | 1.00 |
| JR1e4 | 0.217 | 0.221 | 0.197 | 0.080 | 0.064 | 0.036 | 0.033 | 0.003 | 0.97 | 1.02 |
| ZEROIN | 1.170 | 1.174 | 0.985 | 0.323 | 0.068 | 0.034 | 0.126 | 0.007 | 1.00 | 0.99 |

FD validated (corr 0.97-1.00, ratio ~1.0). MIP op/Fro norm is the LARGEST (1.36x MSE).
MIP Jbad NORM is larger than MSE (0.203 vs 0.138) even though its SIGNED bad-mode
amplification (gain@bad1, traj probes) is negative -- norm and signed gain dissociate.

## T17. EXP-6 correlations + EXP-8 verdict

Full population (n=8, incl JR arm):
  corr(SR, Dedge_p90)=+0.70; corr(SR, Dexc_p90)=+0.63; corr(SR, totalJ)=+0.95;
  corr(SR, rotamp)=+0.83; corr(SR, rJ/GTk)=+0.75; corr(cross4, Dedge_p90)=-0.86.
  ALL smallness metrics anti-predict SR (JR arm: smallest everything, SR 1-24).
Healthy subset (n=6, no JR):
  corr(SR, Dedge_p90)=-0.71; corr(SR, Dexc_p90)=-0.76; corr(SR, totalJ)=+0.44;
  corr(SR, rJ/GTk)=+0.58; corr(SR, rotamp)=-0.09.

Ratios vs MSE (MIP): R_total_op=1.37, R_fd_p90=1.19, R_edge_p50=0.99, R_edge_p90=0.82,
R_exc_p90[Gtube]=0.81, R_rotamp=0.68(AG)/0.82(FD), R_tangent=2.3, R_bad_norm=1.47
(signed bad gain negative), R_inward=2.2-2.5, R_cross4=0.22.

VERDICT TABLE:

| model | literal total-J note? | structured smooth-ext note? | key numbers |
|---|---|---|---|
| MIP | NO (R_total 1.19-1.37 >= 1) | PARTIAL: 4/6 criteria (edge_p90 0.82 ok, exc_p90 0.81 ok, inward 2.5 ok, cross4 0.22 ok; rotamp 0.68 not <0.5, tangent 2.3 not <1) | SR 95 with largest total J |
| FLOW | NO (R_total ~1.2) | similar profile to MIP | SR 95 |
| PDC/PDS | total-J ~MSE (op 1.17-1.37) but bad-dir response 5-9x smaller | YES on attraction axis: frac 1.00 prescribed field | SR 87-96 |
| JR | YES by construction (op 0.19x) | NO: kills inward gain (0.01-0.07) | SR 1-24 |

DECISION (advisor rules): literal reading REJECTED (A fails); structured reading
PARTIALLY supported (B holds on tail-closeness/excess/inward/rot-signed axes, fails on
tangent axis); D CONFIRMED -- explicit attraction regularization (PD-chunk/PDS) reaches
87-96 while suppressing bad-mode response 5-9x below any implicit method, making it the
cleanest realization of the note's useful content.

## T18. TOY-2D (curved path, controlled tube recovery; scripts/toy2d_recovery.py, analysis/toy2d/)

Variant B (K_data=0 before p=0.85) -- ex-nihilo test:
  K_zero-zone mean: MSE -0.027, MIP -0.007, FLOW -0.014 (all ~0: NO spontaneous recovery).
  K(p)-curve tracking: MIP cos 0.840 / sign 1.00 / R2 0.581; FLOW 0.795/0.50/0.528; MSE 0.676/0.67/0.072.
Variant C (zigzag) -- override test:
  K_outward-zone mean: MSE -0.066, MIP -0.070, FLOW -0.078 (data -0.245): all follow the
  data's outward sign, damped ~0.3x; no inward override.
Variant A (constant k=0.25): non-discriminative (all cos 0.71-0.75, SR 1.00; R2 vs
  constant K_data undefined).
Closed-loop SR: B: FLOW 1.00 / MSE 0.76 / MIP 0.64; C: FLOW 0.99 / MIP 0.93 / MSE 0.78.
  PDoracle C: full gain 0.71 vs damped0.6 0.97 (damping > full gain; mirrors real
  gt=79 < gtdamped=93).
Flow field slope: cos(K_recon(t), K_data) -> 1.00 and gain -> 0.95 as t -> 0.9
  (velocity field encodes K_data per the reconstruction formula).
Jacobian: MIP/FLOW total J norms == MSE level (0.2-0.4); total-J note rejected again.
LIMITATION: toy MSE also tracks K_data reasonably and shows no 25-50k destruction --
the objective-gating half of the real-robot story does not reproduce in a 2D MLP;
toy validates the data-sourced half + flow-encoding + no-ex-nihilo + damping story.

## In flight
- EXP-5 one-trajectory: onetraj_mse / onetraj_mip / onetraj_pdc (300k, demo_0 x2000)
- EXP-6 tube-width: tw_mse_center / tw_mip_center (center1000x2), tw_mse_edge / tw_mip_edge (edgeheavy3000); demo radius p10/p50/p90 = 0.226/0.381/0.666
- from-scratch pds batch (12 x 300k) still training

## Width dose-response curve (2026-07-07, dart02 results landed)
| demonstration-tube width (mm) | data | MSE | MIP | gap |
|---|---|---|---|---|
| 0.46 | wp15 | 28 | 55 | +27 |
| 1.42 | wp3 | 52 | 63 | +11 |
| 2.80 | wp3+dart0.2 (transit-gated) | 63 | 84 | +21 |
| 3.33 | wp3+dart0.27 | pending | pending | |
| 5.24 | original | 71 | 95 | +24 |
Both curves are monotone in width; dart02_mip maxd p90=12.4, bounded excursion tails (mse 109/1042); dart02_mse r/GT=0.77 frac 0.86.
DART recovery from real states (63) >> PD recovery from a hypothesized field (wp15pd 14-22) -> further confirmation that coverage is irreplaceable.

## Data x objective matrix, filling in cells (2026-07-07)
- wpmatch (5.05mm, self-anchor saturated): MSE 36 (cross4 .67, maxd p50 53) / MIP **88** (gap +52, largest to date) / MIP@step1 87 (inference-time projection contributes ~0)
- dart27 (3.33mm): MSE **31, SR|stay=54 (first arm with a broken precision channel)** -- the 0.27 noise leaks into the precision segment, plus survivorship bias; flagged "excluded from the width curve"
- PDS@dart02: k0.015->68 / k0.008->71 (baselines MSE 63, MIP 84) -> the explicit prior only compensates for a field the data lacks (original +25); when the data already carries the field it is redundant (+5~8), and with no coverage it backfires (wp15)
- MIP1 vs MIP2 @wpmatch: 87 vs 88; @original: 89 vs 95; @pick2ins (local): 100/100
- Triangle geometry (full_mip_2000): inside the tube ‖y0−y1‖ .007 < ‖y0−a‖ .012 (consistency-dominated, high-dimensional orthogonal composition); at the edge (data tail d>3) y0 rejects the label (.04) while ‖a−y1‖ stays constant at .008-.009; from the deployment view ‖y1(y0)−a‖ is flat at .008 throughout (basin = a perfect label library inside the support); off-support the fields of the two slices are identical (edgejac MIP1≈MIP2)
- Temporal structure of corrections (lagcorr): dart02 immediate peak (k=4: +.098) / wp3 persistently positive / original weakly positive / wpmatch anti-correlated at mid lags (k≥8: −.03) -- the wpmatch toxic pattern = "deviate, then subsequently move outward"

## Policy arm: Cauchy robust loss loses on both testbeds (2026-07-06/07)
| data | MSE | Cauchy | MIP | Cauchy vs MSE |
|---|---|---|---|---|
| original | 71 | 57 | 95 | **−14** |
| wpmatch | 36 | 27 (cross4 .77, SR|cross4=5, maxd p90 16857) | 88 | **−9** |
Verdict: the heavy-tailed regression loss falls below the MSE baseline on both testbeds, and its tails are as unbounded as MSE's (maxd p90 in the 1e4 range). MIP's +52 immunity is not a "robust-loss effect" -- down-weighting large-residual labels provides neither the basin nor deep supervision, and instead discards signal. Merged with EXP-B (bnn=79, deep supervision +43): the immunity comes from the two-view deep-supervision structure, not from the loss shape. (The cauchy +13 on human data is a separate matter: human noise is genuinely heavy-tailed, df≈2, and the benefit comes only from matching the noise distribution.)

## Policy arm: three-arm aux surgery (wpmatch, 2026-07-07; baselines MIP=88 / step1=87 / bnn=79 / MSE=36)
| arm | surgery | SR | cross4 | SR|cross4 | fc4_p50 | maxd_p90 |
|---|---|---|---|---|---|---|
| C1 wpm-ctan | aux target with the inward component removed (tangential correction only) | **0** | 1.00 | 0 | **14** | 30682 |
| C2 wpm-cflip | aux target with the inward component flipped (outward) | **0** | 1.00 | 0 | **8** | 6777 |
| D wpm-ddet | aux gradient blocked from the encoder (stop-grad) | **64** | 0.52 | 31 | 176 | 5076 |
Training-convergence sanity: all three arms' losses converge smoothly (2.8e-4/6.8e-4/9.0e-5); the 0% results are not training failures.
Verdict:
1. The λ=81-weighted aux view IS the radial servo itself. bnn (aux without a noise neighborhood) = 79 still survives -- nothing teaches the wrong direction; C1 keeps the noise neighborhood but strips out the radial correction -> a signal carrying 81x weight teaches "maintain the radial offset" -> at deployment every rollout exits the demonstration tube within 14 steps, total failure (0%); C2 teaches "outward" -> out of the tube within 8 steps. The inward/denoising component is not an optional regularizer: with the correct direction it is a servo, with the wrong direction it is toxic, and it dominates everything at the gradient level.
2. Roughly 1/2 of the servo flows through the representation: cutting the aux gradient to the encoder -> 88->64 (−24), deep-region recovery SR|cross4 31, tail bound lost (maxd p90 54->5076); yet with the action head alone receiving aux, it still beats MSE by +28.
3. Full decomposition merged with EXP-B (wpmatch immunity +52): deep-supervision structure +43, noise neighborhood +9; the noise neighborhood's value rests entirely on the premise that the radial direction is correct (C1/C2); roughly half of the effect flows through encoder co-training (D).

## Policy arm: λ sweep + distillation (wpmatch, 2026-07-07; baselines MIP(λ=81)=88 / MSE=36)
| arm | SR | cross4 |
|---|---|---|
| lam1 (λ=1, aux weighted equally with y0) | **70** | .36 |
| lam3 | 82 | .29 |
| lam9 | 82 | .24 |
| lam27 | 80 | .23 |
| λ=81 (canonical) | 88 | — |
| dstl (single-net co-train, y0→sg(y1)) | 79 | .29 |
| dstl2 (two-net) | 78 | .23 |
| dseq (sequential training) | **0** (questionable, protocol under review) | 1.00 |
Verdict: weak λ dependence -- **equal weighting (λ=1) already buys +34/52, and λ≥3 plateaus at 80-88**. Mechanisms based on gradient magnitude (wd balancing, 100x amplification) are downgraded; the "task structure" of the denoising view is the principal component. Distillation co-train (79) ≈ bnn (79): having y0 imitate y1's output already captures most of the effect; the sequential-training collapse remains to be investigated.

## Objective-decomposition panel on the original data (2026-07-08, orig_bnn results landed)
| arm | SR | maxd p90/p95 | note |
|---|---|---|---|
| MSE | 71 | ~109/1042 | baseline |
| **orig_bnn (noise-free multi-slice aux)** | **81** | **181/4712 (tail bound lost)** | deep supervision +10 |
| MIP (λ=81) | 95 | — | noise ball adds another +14 |
| headmse-30k (frozen MIP trunk + MSE head) | 94 | **4.6** (the bounded excursion tails transfer with the features) | feature-borne |
Decomposition comparison: on original the noise ball accounts for 14/24 (58%); on wpmatch deep supervision accounts for 43/52 (83%) -- the two mechanisms' shares vary with the data: in the toxic-label setting deep supervision dominates (isolating bad labels), in the clean-data setting the noise ball dominates (fine feature sculpting + bounded excursion tails). headmse (94) > bnn (81): the features carved out by noisy targets are worth more than end-to-end training with noise-free targets.

## Gate-family final table (original, 2026-07-08)
none (bnn) 81 | grid0.1 87 | grid0.3 63 | noise0.01 94 | noise0.03 **97** | noise0.1 95 | noise0.3 87 | noise1.0 84 | annealed0.003→3 (single-view) **96** | headmse-150k **96** | MSE 71
