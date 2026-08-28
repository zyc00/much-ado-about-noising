# CLOSURE round + must-have batch · Results summary (numbers only)

Protocol as before; new metrics: GT proxy a_GT(s)=clip(a_anchor+G_GT·δz,−1,1); band-resolved action error = ‖â−a_GT‖;
excursion = events of first entry into [2,4) from d<2; we tally the fraction that returns to d<2 or escalates to d≥4 within 5/10/20 steps.

## Table 1: ABS (EXP-1)

| Model | RAW_ABS R²/cos_GTabs/poseig/rotsv | IMPLIED_Δ R²/cos_GTdelta/poseig/gain@bad1/rot@amp | RESIDUAL(G−I) cos/eigs | FEATRIDGE trainR2/opR2/cos | pos target error p50/90/95/99 (m) |
|---|---|---|---|---|---|
| abs-MSE φ₀ | .442/.74/[+.0007,+.0035,+.0061]/.058 | .672/.61/[−.021,−.017,−.014]/−.020/.029 | .65/[−.0010,−.0008,−.0007] | .861/.233/.18 | .0079/.058/.066/.106 |
| abs-MIP φ₀ | .434/.80/[+.0008,+.0037,+.0059]/.053 | .662/.64/[−.028,−.015,−.009]/−.027/.021 | .68/[−.0014,−.0007,−.0005] | .865/.377/.17 | .0073/.054/.058/.063 |
| abs-MIP φτ | identical to φ₀ digit-for-digit | same | same | .865/.378/.17 | same |
| abs-RAND | .586/−.03/≈0/.009 | .985/.39/[−.138,−.087,−.029]/−.110/.009 | .41 | .741/.570/.16 | .209/.265/.271/.281 |

The rot target error is constant ≈1.57 rad for all models (including RAND) — suspected coordinate-convention artifact; annotation pending verification.
Insertion segment (progress>0.75, N=102) pos p50: abs-MSE .0099, abs-MIP .0264 (p90 reverses: .060 vs .058).

## Table 2: Direct measurement of the two factors (9 models)

| Model | SR | cross2 | cross4 | SR\|4 | maxd p50/90 | fc4 p50/90 | d<2 err p50 | d<2 dd_mean | [2,4) err p50 | [2,4) dd_mean | escalate-to-4 w10/20 | return-to-<2 w10/20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MIPstep1 | 95 | .93 | .07 | 29 | 2.75/3.4 | 159/162 | .1042 | +.0081 | .2228 | −.084 | .04/.05 | .69/.77 |
| ET3b | 84 | .82 | .21 | 24 | 2.77/2.8e4 | 159/173 | .1110 | +.0080 | .2202 | −.037 | .17/.23 | .64/.74 |
| MSE | 71 | .85 | .32 | 9 | 2.94/6.9e4 | 156/167 | .1138 | +.0092 | .3012 | +.283 | .08/.12 | .63/.73 |
| TRUNKLR01 | 69 | .97 | .40 | 22 | 3.45/2.8e4 | 152/161 | .1201 | +.0175 | .2521 | −.019 | .16/.22 | .66/.75 |
| AUXDET | 64 | .98 | .50 | 28 | 4.08/4.5e4 | 160/174 | .1198 | +.0132 | .2930 | +.015 | .07/.14 | .56/.70 |
| ET3a | 62 | .92 | .42 | 10 | 3.20/64 | 146/165 | .1215 | +.0091 | .2468 | +1.206 | .12/.25 | .59/.69 |
| DUP100 | 54 | .92 | .47 | 2 | 3.85/1.8e4 | 160/174 | .1185 | +.0130 | .2987 | +.560 | .16/.20 | .57/.66 |
| MSEs10k | 51 | 1.00 | .56 | 12 | 6.05/2.4e5 | 157/172 | .1358 | +.0203 | .3070 | −.014 | .16/.22 | .59/.73 |
| LOWLR | 50 | .99 | .58 | 14 | 11.4/1.1e5 | 151/165 | .1293 | +.0237 | .2901 | −.012 | .23/.29 | .61/.72 |

## Table 3: ET3 replication (4 starting points, +150k; SR / cross4 / (final geometry cos))

| Starting point (initial cos) | A standard MSE | B switch to MIP | C low LR | D/E frozen trunk |
|---|---|---|---|---|
| s10k s1 (0.84) | 62/.42 (0.41) | 84/.21 (0.72) | 73/.32 (0.87) | 51/.60 (0.82) |
| s5k s1 (0.83) | 65/.37 (0.54) | 86/.16 (0.70 negative-definite) | — | — |
| s25k s1 (0.83) | 65/.37 (0.41) | 69/.35 (0.73) | — | — |
| s10k s2 (0.65) | 66/.37 (0.49) | **90/.16** (0.62) | 28/.76 (0.65) | 45/.66 (0.65) |

- All B variants: cleanErr p50 = .0030–.0033; A variants .0052–.0055; C .0069–.0072; D/E .0115–.0120 (flat line).
- A destruction timeline (all starting points): drops to 0.52–0.72 at +26k → 0.37–0.54 at +50k → settles at 0.41–0.54,
  rotsv 0.09–0.19, gain@bad1 turns positive; B stable throughout (±0.03), rot-amp ≤0.037.
- seed2-B details: SR 90, SR|crossed-4 = 38, maxd p90/95 = 8.7/51 (never escapes deep), [2,4) dd=−.049, escalate-to-4 w20=.08.
- e3rs25b: [2,4) dd=−.011; e3rs5b: −.059 (in-band dd_mean convention).

## Table 4: Servo ablation closure (complete)

| Ablation | featshift | chunkerr | [1,2)/[2,4)/[4,6) cos | SR | cross4 | SR\|4 | d<2 err p50/dd_mean | [2,4) err p50/dd_mean | escalate-to-4 w10/20 | return-to-<2 w10/20 |
|---|---|---|---|---|---|---|---|---|---|---|
| k=0 (MIP-ridge) | .000 | .0121 | .96/.73/.20 | 94 | .10 | 40 | .0981/+.0087 | .1998/−.088 | .02/.04 | .81/.91 |
| top-1 | .377 | .0143 | .94/.40/−.29 | 79 | .30 | 30 | .0958/+.1463 | .2156/+.193 | .13/.15 | .70/.81 |
| top-2 | .567 | .0161 | .90/.34/−.27 | 67 | .40 | 18 | .1015/+.0164 | .2415/+.496 | .18/.22 | .69/.81 |
| top-3 | .649 | .0214 | .89/.28/−.09 | 34 | .74 | 11 | .1140/+.0250 | .2905/+.025 | .26/.30 | .65/.73 |
| null-1 | .172 | .0152 | .95/.69/.34 | 52 | .77 | 38 | .1277/+.0255 | .2874/−.018 | .12/.21 | .55/.78 |
| null-2 | .256 | .0193 | .95/.66/.15 | 38 | .82 | 24 | .1399/+.0275 | .2972/+.010 | .10/.17 | .65/.77 |
| MSE k=0 (MSE-ridge) | .000 | .0151 | .91/.60/.32 | 77 | .29 | 21 | .1008/+.0135 | .2521/−.040 | .09/.12 | .67/.81 |

## Table 5: Phase-conditioned operators

| bin (progress) | N | GT R²cv | GT sv | MSE cos | MIP cos | MSEs10k cos | ET3b cos |
|---|---|---|---|---|---|---|---|
| 0 (.00–.26) | 471 | .688 | .50/.41/.21 | .96 | .99 | .97 | .98 |
| 1 (.26–.51) | 472 | .238 | .68/.07/.06 | .51 | .79 | .70 | .51 |
| 2 (.51–.74) | 380 | .340 | 1.35/.75/.43 | .31 | .40 | .83 | .51 |
| 3 (.74–.93) | 563 | .708 | 1.05/.50/.25 | .84 | .92 | .87 | .87 |

rotsv per bin: MSE .031/.003/.297/.103; MIP .031/.002/.247/.029.

## Table 6: Cross-task feasibility (experiments not yet launched)

| Task | Scripted expert | Clean cloud | Recovery data | Distance/PNR | Actions | Conclusion |
|---|---|---|---|---|---|---|
| Square | must write ourselves (moderate effort) | available after collection | scripted expert in hand suffices | ✓ | delta OSC | feasible, first choice |
| Can | script is simple | same as above | same as above | ✓ | delta OSC | feasible, second choice |
| Lift | trivial | — | — | — | — | not recommended (saturated) |
| ToolHang init2grasp / pick2ins | data already on PVC | ✓ | needs additional within-segment puredart | ✓ | ✓ | zero collection cost, run first |

## CLOSURE B series (partial results out)

### B4 operator-anchor oracle (complete)
| γ | Geometry throughout | cleanErr p50 | SR | cross4 | SR\|4 | drift[2,4) dd_mean | escalate-to-4 w10/20 |
|---|---|---|---|---|---|---|---|
| 1e-1 | cos .83–.84 negative-definite, rot-amp .004–.006 (throughout) | .0054 | 63 | .40 | 8 | −.028 | .08/.20 |
| 1e-2 | cos .83–.84 negative-definite, rot-amp .006–.008 | .0053 | 62 | .44 | 14 | −.010 | .16/.26 |

(Reference: standard MSE continued training 62/.42; s10k starting point 51/.56.)

### B1 tau-teacher value anchor (complete, all negative)
| β | Final geometry | cleanErr | SR | cross4 |
|---|---|---|---|---|
| 1 | 0.41 destroyed | .0057 | 67 | .41 |
| 3 | 0.41 destroyed | .0064 | 51 | .50 |
| 10 | 0.60 | .0078 | 55 | .50 |
| 30 | 0.63 | .0090 | 41 | .66 |
| 100 | 0.67 | .0105 | 33 | .72 |
| 300 | 0.65 | .0117 | 35 | .70 |
| t0-control β=10/100 | 0.60/0.63 | .0126/.0139 | 41/37 | .65/.72 |

### B2 tau-feature anchor (identity P; complete, all negative)
| β | Final geometry | cleanErr | SR | cross4 |
|---|---|---|---|---|
| 1 | 0.47 destroyed | .0056 | 63 | .40 |
| 10 | 0.51 | .0064 | 59 | .47 |
| 100 | 0.69 negative-definite, rot-amp .031 | .0083 | 47 | .54 |

### B3 bad-mode penalty (complete, partially positive)
| λ | Geometry | gain@bad1 | cleanErr | SR | cross4 | maxd p90/95 |
|---|---|---|---|---|---|---|
| 1e-2 | 0.60–0.66 negative-definite throughout | −.024 | .0053 | **71** | .33 | **60/269 (deep escape sealed off)** |
| 1e-1 | 0.56–0.63 negative-definite | −.025 | .0053 | 53 | .50 | 5.5e4 |

### V batch: target-correcting tau-view (complete, 19 runs)

V1 TAU-ZERO-TARGET (aux input = 0):
| Start / w | SR | cross4 | cleanErr |
|---|---|---|---|
| s10k-s1 w=10/30/100/300 | 58/61/62/58 | .44/.42/.46/.45 | .0052–.0054 |
| s25k w=100 | 72 | .35 | .0052 |
| s10k-s2 w=100 | 70 | .37 | .0052 |
| s5k w=100 | 65 | .45 | .0053 |
| s1k w=100 | 55 | .50 | .0054 |

V2 TAU-TUBE-TARGET (aux input = a*+0.1ε):
| Start / w | SR | cross4 | SR\|4 | cleanErr | maxd p90/95 |
|---|---|---|---|---|---|
| s10k-s1 w=10 | 75 | .29 | 14 | .0039 | 6.8e3 |
| s10k-s1 w=30 | 76 | .28 | 14 | .0034 | 1.5e4 |
| s10k-s1 w=100 | **84** | **.19** | 16 | .0031 | 2.9e3 |
| s10k-s1 w=300 | 83 | .18 | 6 | .0032 | 2.9e3 |
| s1k w=100 | 83 | .25 | 32 | .0031 | 1.4e4 |
| s5k w=100 | 79 | .27 | 22 | .0031 | 1.7e5 |
| s25k w=100 | 79 | .25 | 16 | .0031 | 8.2e3 |
| **s10k-s2 w=100** | **96** | **.06** | 33 | .0032 | **3.3/4.6** |

V3 TAU-SCRAMBLE-TARGET (aux input = Q(a*+0.1ε)):
| w | SR | cross4 | SR\|4 | cleanErr | maxd p90 |
|---|---|---|---|---|---|
| 30 | 83 | .20 | 15 | .0034 | 6.5e3 |
| **100** | **93** | **.10** | 30 | .0031 | **3.8** |
| 300 | 81 | .21 | 10 | .0033 | 4.5e3 |

### CLOSURE scoreboard (continued training +150k from an MSE early ckpt; reference: MIP takeover 84–90, from-scratch MIP 94–99)

| Constraint type | Best SR | By criterion |
|---|---|---|
| B1 value anchor | 67 (β=1, ≈unconstrained) | ✗ (monotone degradation) |
| B2 feature anchor | 63 (β=1) | ✗ |
| B4 operator anchor | 63 | ✗ (static pinning still has zero effect) |
| B3 bad-mode penalty | 71 | partial (+9, seals off deep escape) |
| **V1 zero-target** | 72 | ✗ (cleanErr also stuck at the MSE level) |
| **V2 tube-target** | **84 (s1) / 96 (s2)** | **✓ meets the criterion / full closure** |
| **V3 scramble-target** | **93** | **✓ strong success** |

Against the advisor's criteria: SR≥85 supports the theory ✓ (93/96); ≥90 strong support ✓; ≥94 matches MIP ✓ (96, s2).
Interpretation key: tube succeeds while zero fails, and scramble succeeds → **distributional non-degeneracy** of the aux input is necessary; semantics are not.

### B2 servo-P / null-P variants and B5: not launched; +300k extended training not launched

## Appendix: infrastructure notes for this round
- GT-proxy accuracy / band-resolved dynamics metrics have been merged into eval_twofactor.py / eval_servo_ablation.py;
- teacher mechanism = deepcopy of the network at the INIT_CKPT starting point (verified that the anchor term = 0 at step 0);
- B3 assets: closure_assets.npz (1688 pairs of [2,4) normalized windows + bad-mode directions in normalized space).
