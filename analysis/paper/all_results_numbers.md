# Consolidated results of all experiments (numbers only, no analysis)

## 0. Protocol definitions

- Task: ToolHang init→insertion (full2ins), scripted clean data, delta-action OSC, ChiUNet 20M, 300k steps (unless noted).
- SR: held-out seeds 21000–21100, official protocol settle+H16, assembled criterion.
- Support distance d: 1-NN distance of the per-dim z-scored 53-dim obs to the anchor cloud of 40 clean demos; PNR=4.0.
- cross2/cross4: fraction of episodes with within-rollout max d ≥2 / ≥4; SR|cross4 and SR|stay are conditional success rates.
- drift(band): per-step mean of d_{t+1}−d_t within that band over the rollout (heavy-tailed; mean-based protocol).
- Operator probe (OPPROBE/pairs protocol): test = held-out puredart recovery data; band [2,4) (unless noted);
  ridge fit of δa ~ G δz; cosGT = sample-average cosine of the predicted direction against G_GT; poseig = eigenvalues of the de-z-scored
  symmetrized eef-pos block; rotsv = largest singular value of the de-z-scored rot block.
- G_GT: fit to the scripted recovery actions on the same pairs; linear R²≈0.6 (+velocity 0.65, de-saturated 0.70); rank-3;
  49% of recovery actions hit the clip; G_within (lateral variation within the clean data) R²=0.045.
- FEATRIDGE: frozen penultimate features (input to the last layer of final_conv, 2048d), clean ridge readout (with intercept);
  the operator is induced by differencing the readout.
- badmode: bad1 = positive eigen-direction of the G_MSE pos block; rot-amp = largest right-singular direction of the G_MSE rot block;
  gain@bad1 = bad1ᵀ sym(B) bad1; rot@mseamp = ‖R·v_rot‖ (uniform de-z-scored protocol).
- cleanErr: L2 of the first executed action of f(0,0,s) against GT; train = demos 0–59 of the 2k file;
  held = demos 5000–5059 of the 20kB file (unseen by the 2k models).
- CKA: linear CKA; RAND = torch seed-0 untrained network.

## 1. Baselines and main models, closed loop (with drift metrics)

| model | SR | cross2 | cross4 | SR\|cross4 | SR\|stay | drift(d<2) | drift([2,4)) | maxd p50/90/95 | first-cross4 p50 |
|---|---|---|---|---|---|---|---|---|---|
| MSE-2k s1 | 71 | 0.84 | 0.32 | 9 | 100 | +0.0106 | −0.0036 | 2.91/6.9e4/1.3e5 | 155 |
| MIP-2k s1 step1 | 95 | 0.91 | 0.07 | 29 | 100 | +0.0111 | −0.4274 | 2.75/3.42/6.21 | 158 |
| MIP-ridge head(λ1e-4) | 94 | — | 0.10 | 40 | 100 | +0.0093 | −0.4041 | 2.69/3.77 | 160 |
| AUXDET | 64 | 0.98 | 0.50 | 28 | 100 | +0.0166 | −0.0898 | 4.08/4.5e4/3.0e5 | 159 |
| DUP100 | 54 | 0.90 | 0.47 | 2 | 100 | +0.0209 | +2.0220 | 3.85/1.8e4/4.1e4 | 159 |
| MSE early-stop s10k | 51 | 1.00 | 0.56 | 12 | 100 | +0.0226 | −0.0724 | 6.05/2.4e5/2.6e5 | 156 |
| LOWLR | 50 | 0.99 | 0.58 | 14 | 100 | +0.0258 | −0.0815 | 11.4/1.1e5/2.3e5 | 150 |
| TRUNKLR01 | 69 | 0.97 | 0.40 | 22 | 100 | +0.0232 | −0.1646 | 3.52/2.8e4/3.8e4 | 151 |
| ZAW10 | 87 | 0.84 | 0.15 | 13 | 100 | +0.0106 | −0.3072 | 2.83/36.8/64.8 | 155 |
| ZAW100 | 52 | 0.92 | 0.51 | 6 | 100 | +0.1619 | +51.96 | 4.26/3.1e4/5.1e4 | 117 |
| mip_traj s300k step1 | 89 | 0.87 | 0.19 | 42 | 100 | +0.0093 | −0.3206 | 2.71/62/2746 | 150 |

Multi-seed SR (no drift metrics): MSE-2k = 71/79 (+ a third seed, 3 seeds 73±5.3); MIP-2k = 94–99 (96.7±2.1);
MIP-2k step1 94 / 2-step 96 (seed0).

## 2. Operator probe (OPPROBE, policy mode = f(0,0,s) response)

| model | R² | cosGT | poseig | rotsv/sv | negdef | SR |
|---|---|---|---|---|---|---|
| MIP (operator_fit) | — | 0.73 | negdef | — | ✓ | 94–99 |
| MSE (operator_fit) | — | ~0.5–0.6 | +0.021 | 5× rot amplification | ✗ | 71–79 |
| Standalone denoiser (denoiser mode, tube) | 0.917 | 0.70 | [−.018,−.011,−.007] | sv .54/.33/.19 | ✓ | 98 (composed with frozen MSE) |
| ZEROIN (denoiser, zeros) | 0.620 | 0.65 | [−.019,−.011,−.002] | sv .87/.60/.33 | ✓ | 75 (deployed standalone) |
| ZEROFLAT | 0.605 | 0.46 | [−.015,−.012,+.022] | sv 1.5/.83/.38 | ✗ | — |
| RANDIN | 0.646 | 0.69 | [−.022,−.010,+.013] | sv .80/.59/.33 | ✗ | 78 (deployed standalone) |
| SCRAMBLE | 0.914 | 0.69 | [−.018,−.013,−.010] | sv .54/.33/.23 | ✓ | — |
| EQW | 0.621 | 0.42 | [−.019,−.013,+.015] | sv 1.53/.86/.40 | ✗ | 83 |
| TAU50 | 0.621 | 0.68 | [−.019,−.010,+.011] | sv .86/.59/.45 | ✗ | 90 |
| TAU99 | 0.662 | 0.84 | [−.023,−.010,+.018] | sv .86/.56/.32 | ✗ | 95 |
| FLOW | — | — | — | — | — | 93 |
| DUP100 | 0.555 | 0.67 | [−.016,−.010,+.010] | sv .82/.63/.40 | ✗ | 54 |
| DUP100 no clip | 0.628 | 0.53 | [−.013,−.012,−.004] | sv .94/.88/.33 | ✓ | 52 |
| AUXDET | 0.611 | 0.91 | [−.024,−.010,+.022] | sv .87/.58/.31 | ✗ | 64 (step1) / 59 (2step) |
| JACREG λ=1e-4 | 0.710 | 0.92 | [−.021,−.009,+.023] | sv .85/.55/.33 | ✗ | 24 |
| JACREG λ=1e-3 | — | — | — | — | — | 1 |
| OBSNOISE σ=0.02 | — | — | — | — | — | 68 |
| LOWLR | 0.731 | 0.77 | [−.023,−.015,−.009] | sv .95/.39/.32 | ✓ | 50 |
| WD×1000 | 0.604 | 0.51 | [−.023,−.012,+.001] | sv 1.17/.93/.41 | ✗ | 60 |
| HEADONLY (frozen random trunk + SGD head) | 0.765 | 0.82 | [−.015,−.007,−.002] | sv .59/.37/.28 | ✓ | 0 |
| TRUNKLR01 | 0.748 | 0.84 | [−.023,−.010,−.007] | sv .92/.43/.32 | ✓ | 69 |
| ZAW1 | 0.631 | 0.53 | [−.013,−.012,+.023] | sv 1.49/.84/.38 | ✗ | 76 |
| ZAW3 | 0.636 | 0.43 | [−.015,−.012,+.016] | sv 1.44/.79/.37 | ✗ | 67 |
| ZAW10 | 0.559 | 0.48 | [−.014,−.008,+.015] | sv 1.14/.77/.56 | ✗ | 87 |
| ZAW30 | 0.583 | 0.59 | [−.015,−.010,−.003] | sv .94/.73/.32 | ✓ | 78 |
| ZAW100 | 0.543 | 0.55 | [−.017,−.010,0.0] | sv .89/.80/.46 | ✗ | 52 |
| ZAW300 | 0.546 | 0.58 | [−.015,−.011,+.007] | sv .89/.64/.38 | ✗ | 78 |
| ABS-MIP | 0.353 | 0.34 | [+.001,+.003,+.006] | sv 2.77/.26/.11 | ✗ | 64 |
| ABS-MSE | 0.085 | 0.35 | [+.001,+.003,+.006] | sv 1.59/.30/.11 | ✗ | 4 |

τ-grid (denoise_zeroin single view, weight = 1/(1−τ)²):
| τ | weight | R² | cosGT | poseig | negdef | slice-SR |
|---|---|---|---|---|---|---|
| 0.25 | 1.8× | 0.638 | 0.43 | [−.015,−.012,+.022] sv1.58 | ✗ | 77 |
| 0.5 | 4× | 0.629 | 0.45 | [−.019,−.012,+.008] sv1.39 | ✗ | 73 |
| 0.75 | 16× | 0.614 | 0.59 | [−.015,−.011,−.000] sv0.94 | ✓ | 84 |
| 0.9 | 100× | 0.620 | 0.65 | [−.019,−.011,−.002] | ✓ | 75 |
| 0.99 | 10⁴× | 0.644 | 0.66 | [−.019,−.011,−.001] sv0.94 | ✓ | 78 |

ZAW head1 (deployed as f(0.9,0,s)): w=1/3/10/30/100/300 → 73/68/86/81/52/78.

## 3. Feature-ridge (frozen features + clean linear readout)

| features | opR² | cosGT | poseig | negdef |
|---|---|---|---|---|
| Denoiser φ₁ (tube, aligned version) | 0.905 | 0.67 | [−.016,−.012,−.005] | ✓ |
| SCRAMBLE φ₁ (aligned version) | 0.905 | 0.68 | [−.017,−.013,−.008] | ✓ |
| ZEROIN φ₁ | 0.649 | 0.66 | [−.014,−.010,+.001] | ✗ |
| MSE φ₀ (seed1) | 0.654 | 0.56 | [−.011,−.010,−.006] | ✓ |
| MIP φ₀ (seed1) | 0.697 | 0.74 | [−.017,−.010,0.0] | ✗ |
| MIP φ₁ (aligned version) | 0.852 | 0.93 | [−.018,−.011,+.021] | ✗ |
| RAND φ₀ | 0.733 | 0.69 | [−.015,−.005,+.015] | ✗ |
| RAND φτ | 0.795 | 0.71 | [−.015,−.005,+.018] | ✗ |
| AUXDET φ₀ | 0.646 | 0.90 | [−.021,−.010,+.019] | ✗ |
| AUXDET φ₁ (tube) | 0.684 | 0.90 | [−.018,−.010,+.021] | ✗ |
| LOWLR φ₀ | 0.720 | 0.76 | [−.017,−.015,−.004] | ✓ |
| WD φ₀ | 0.621 | 0.49 | [−.024,−.011,+.006] | ✗ |
| HEADONLY φ₀ | 0.671 | 0.71 | [−.011,−.008,+.001] | ✗ |
| TRUNKLR01 φ₀ | 0.705 | 0.79 | [−.018,−.012,−.005] | ✓ |
| ZAW1/3/10/30/100/300 φ₀ | .654/.656/.571/.578/.561/.559 | 0.48/0.41/0.47/0.63/0.57/0.63 | mostly indefinite | ✗ (unless noted) |

With an intercept, the on-support readout trainR2 of all trained models = 1.000 (RAND = 0.929–0.982).

## 4. Layerwise feature-ridge (cosGT)

| layer | MSE-t0 | MIP-t0 | MIP-τ | DEN-τ | MSE300k(traj) | MIP300k(traj) |
|---|---|---|---|---|---|---|
| enc | 0.65 | 0.72 | 0.72 | 0.78 | 0.68 (negdef) | 0.69 |
| cond | 0.65 | 0.69 | 0.69 | 0.77 | 0.68 (negdef) | 0.69 |
| down0 | 0.45 | 0.53 | 0.54 | 0.67 | 0.41 | 0.76 |
| down1 | 0.49 | 0.41 | 0.41 | 0.56 | 0.61 | 0.71 |
| down2 | 0.50 | 0.78 | 0.77 | 0.55 | 0.60 | 0.82 |
| mid | 0.49 | 0.77 | 0.78 | 0.51 | 0.60 | 0.77 |
| up0 | 0.51 | 0.78 | 0.77 | 0.54 | 0.55 | 0.79 |
| up1 | 0.60 | 0.72 | 0.72 | 0.58 | 0.58 | 0.71 |
| pen | 0.59 | 0.73 | 0.73 | 0.69 | 0.55 | 0.70 |

## 5. Head surgery / ridge-head λ sweep (MSE/MIP seed1 features)

Static (plateau for λ≤1e-2): MSE cos 0.59–0.61, rotsv 0.13–0.14, bias[2,3)=0.36–0.37;
MIP cos 0.73–0.74, rotsv 0.086–0.089, bias[2,3)=0.24. λ=1: MSE rotsv 0.084, bias[1,2) 0.15;
λ=1e2: trainR2 0.65, bias 0.58 (both collapse).

Closed-loop SR:
| λ | MSE | MIP |
|---|---|---|
| 1e-8 | 53 (cross4 .53) | 83 (.23) |
| 1e-6 | 75 (.29) | 88 (.15) |
| 1e-4 | 77 (.29) | 94 (.10) |
| 1e-2 | 46 (.62) | 67 (.49) |
| 1 | 1 | 6 |
| 1e2 | 0 | 0 |

Head-replacement baselines: MSE original head 71 → ridge 74; MIP original head 99 → ridge 95 (trainR2 = 1.000 for both).
RAND-feature ridge closed loop: λ∈{1e-6,1e-4,1e-2}×{φ₀,φτ} all 0/100, cross4=1.00;
RAND+SGD head (HEADONLY) also 0/100.
RAND static λ sweep: best cos 0.65 (λ1e-6/1e-2), never negdef (mid λ), bias[1,2) ≥0.18.

## 6. Servo-subspace ablation (MIP seed1; U from the SVD of the δφ×(G_GT δz) cross-covariance; svals 1.0/0.194/0.011/0.006/0.002/0.0)

Fixed head (head fit on the unablated features):
| k | onR2 | cosGT | SR | cross4 |
|---|---|---|---|---|
| 0 | 1.000 | 0.73 | 94 | 0.10 |
| 1 | 0.772 | 0.20 | 0 | 1.00 |
| 2 | 0.823 | −0.02 | 0 | 1.00 |
| 3 | 0.804 | −0.03 | 0 | 1.00 |
| null-1 | 0.951 | 0.71 | 1 | 1.00 |
| null-2 | 0.907 | 0.68 | 0 | 1.00 |

Refit head (head refit on the ablated features; onR2/chunkR2 all 1.000):
| ablation | [2,4) cos | [1,2) cos | SR | cross4 | SR\|cross4 | drift(d<2) | drift([2,4)) |
|---|---|---|---|---|---|---|---|
| k=0 | 0.73 | 0.96 (negdef) | 94 | 0.10 | 40 | +0.0093 | −0.4041 |
| top-1 | 0.40 | 0.94 | 79 | 0.30 | 30 | +0.1345 | +4.34 |
| top-2 | 0.34 | 0.90 | 67 | 0.40 | 18 | +0.0165 | +3.25 |
| top-3 | 0.28 | 0.89 | 34 | 0.74 | 11 | +0.0278 | +0.137 |
| null-1 | 0.69 | 0.95 | 52 | 0.77 | 38 | +0.0289 | −0.044 |
| null-2 | 0.66 | 0.95 | 38 | 0.82 | 24 | +0.0301 | −0.009 |
| k=5 | 0.18 | 0.75 | — | — | — | — | — |
| k=6 | 0.20 | 0.76 | — | — | — | — | — |

Band cross-section ([1,2) / [2,4) cos, refit k=0): MSE-s10k 0.91/0.86; MSE-s300k (traj) 0.92/0.57;
MSE-s1 0.91/0.60; DUP100 0.93/0.68; ZEROIN 0.94/0.68.

## 7. Bad-mode projections (uniform protocol)

| model | gain@bad1 | rot@mseamp | rot top-sv |
|---|---|---|---|
| G_GT | −0.0100 | 0.013 | 0.063 |
| MSE | +0.0027 | 0.152 | 0.152 |
| MIP | −0.0197 | 0.036 | 0.075 |
| Standalone denoiser | −0.0097 | 0.001 | 0.012 |
| SCRAMBLE | −0.0120 | 0.001 | 0.012 |
| ZEROIN | −0.0076 | 0.021 | 0.053 |
| ZEROFLAT | +0.0138 | 0.115 | 0.134 |
| EQW | +0.0083 | 0.099 | 0.122 |
| RANDIN | −0.0006 | 0.025 | 0.055 |
| TAU50 | −0.0003 | 0.025 | 0.063 |
| TAU99 | +0.0038 | 0.016 | 0.041 |
| DUP100 | +0.0002 | 0.027 | 0.077 |
| JACREG1e4 | +0.0068 | 0.006 | 0.076 |
| ABS-MSE | +0.0046 | 0.168 | 0.565 |
| ABS-MIP | +0.0040 | 0.185 | 0.604 |
| MSE-ridge λ1e-4/1e-2 | −0.0063/−0.0075 | 0.050 | 0.140/0.141 |
| MIP-ridge λ1e-4/1e-2 | −0.0030/−0.0053 | 0.035 | 0.089/0.086 |

## 8. Representation geometry

CKA (penultimate, t0/zeros):
| pair | CKA on | CKA Δφ | Δφ restricted to the former's servo subspace | servo-subspace principal angles |
|---|---|---|---|---|
| RAND vs MSE | 0.546 | 0.426 | 0.407 | [86.9°, 89.0°] |
| RAND vs MIP | 0.524 | 0.394 | 0.600 | [83.2°, 88.9°] |
| MSE vs MIP | 0.994 | 0.922 | 0.924 | [47.6°, 53.5°] |

View invariance (viewdiff ratio / viewCKA):
| model | servo | bad | all |
|---|---|---|---|
| RAND | 0.388/0.996 | 0.305/0.988 | 1.368/0.998 |
| MSE | 0.000/1.000 | 0.000/1.000 | 0.000/1.000 |
| MIP | 0.001/1.000 | 0.001/1.000 | 0.002/1.000 |

Adapter distillation (linear T: φ_MSE→φ_MIP, fit on clean data): adapterR2=1.000, chunkR2=1.000;
deployed: operator cos 0.60, rotsv 0.140, SR 77, cross4 0.31, SR|cross4 26.

Policy-visited operator (rollout states with d∈[2,4)):
| | MIP | MSE |
|---|---|---|
| SR / window count | 99 / 1237 | 71 / 2337 |
| R² / cosGT | 0.188 / 0.12 | 0.502 / 0.42 |
| poseig | [−.022,−.007,−.004] ✓ | [−.020,−.009,+.001] ✗ |
| gain@bad1 / rot@amp / rotsv | −0.008 / 0.042 / 0.130 | −0.007 / 0.062 / 0.135 |

## 9. Training trajectories

MSE (s1k–25k are early short runs; s27k+ is an independent traj seed):
| step | operator cos | rotsv | gain@bad1 | rot@amp | SR | cross4 | cleanErr p50 (train/held) |
|---|---|---|---|---|---|---|---|
| 1k | 0.78 | 0.090 | −0.011 | 0.009 | 21 | 0.87 | 0.0292/0.0274 |
| 5k | 0.83 | 0.082 | −0.014 | 0.010 | 46 | 0.70 | 0.0164/0.0162 |
| 10k | 0.84 | 0.078 | −0.013 | 0.006 | 51 | 0.56 | 0.0109/0.0108 |
| 25k (early) | 0.83 | 0.064 | −0.014 | 0.012 | 41 | 0.65 | 0.0086/0.0083 |
| 27k (traj) | 0.66 | 0.063 | −0.011 | 0.027 | — | — | — |
| 51k | 0.49 | 0.124 | −0.000 | 0.105 | 44 | 0.58 | 0.0062/0.0061 |
| 77k | 0.54 | 0.110 | +0.001 | 0.078 | — | — | — |
| 102k | 0.52 | 0.129 | +0.005 | 0.091 | 63 | 0.39 | 0.0056/0.0059 |
| 126–152k | 0.46–0.47 | 0.131–0.138 | +0.004~+0.010 | 0.108–0.111 | — | — | — |
| 176–276k | 0.51–0.56 | 0.099–0.115 | −0.006~+0.002 | 0.067–0.087 | 71 (200k) | 0.34 | 0.0049/0.0051 |
| 300k | 0.56 | 0.104 | −0.000 | 0.071 | 79 | 0.22 | — |

MSE per-ckpt SR\|cross4 / SR\|stay: s1k 9/100, 5k 23/100, 10k 12/100, 25k 9/100,
50k 3/100, 100k 5/100, 200k 15/100, 300k 5/100.
MSE poseig timeline (traj seed): 27k [−.017,−.013,+.002]; 51k [−.021,−.012,+.010];
102k [−.021,−.012,+.013]; 152k [−.020,−.012,+.018]; 200k [−.025,−.013,0.0];
300k [−.020,−.013,+.008].

MSE per-snapshot layerwise cosGT (down2/mid/pen): 27k .56/.62/.61; 51k .62/.65/.48;
77k .66/.67/.53; 102k .64/.73/.52; 126k .58/.63/.46; 152k .59/.67/.47; 176k .62/.61/.54;
201k .59/.62/.50; 227k .59/.64/.51; 252k .59/.61/.54; 276k .60/.60/.55; 300k .60/.60/.55.

MIP (early + traj seed):
| step | operator cos | rotsv | rot@amp | SR (step1) | cross4 | cleanErr p50 |
|---|---|---|---|---|---|---|
| 1k | 0.81 | 0.078 | 0.003 | — | — | 0.0297 |
| 5k | 0.73 | 0.058 | 0.024 | — | — | 0.0135 |
| 10k | 0.70 | 0.056 | 0.022 | — | — | 0.0095 |
| 25k | 0.74/0.68 | 0.056/0.089 | 0.018/0.032 | 82 | 0.20 | 0.0066 |
| 51k | 0.72 | 0.067 | 0.025 | — | — | 0.0048 |
| 76–101k | 0.72/0.70 | 0.082/0.081 | 0.030 | 96 (100k) | 0.07 | 0.0032 |
| 126–300k | 0.70–0.71 | 0.063–0.072 | 0.025–0.029 | 89 (300k, traj seed) | 0.19 | 0.0030→0.0021 |

MIP per-snapshot layerwise cosGT (down2/mid/pen): 26k .63/.75/.67; 51k .85/.85/.70;
76k .85/.85/.72; 101k .85/.82/.70; 126k .85/.77/.70; 151k .83/.76/.70; 176k .86/.76/.70;
201k .82/.72/.69; 226k .83/.79/.70; 251k .83/.79/.70; 276k .82/.77/.70; 300k .82/.77/.70.

ZEROIN early: 1k/5k/10k/25k → cos 0.78/0.79/0.73/0.81, rot@amp 0.014–0.018.

Trajectory CKA (t0 features vs RAND): CKArand_all ≈ 0.54–0.55 throughout (all models);
CKArand_servo: MSE 0.069 (1k) → 0.016–0.018 (50k+); MIP 0.065 (1k) → 0.025–0.028 (100k+).

## 10. ET2 (frozen early trunk + ridge head, closed loop)

| trunk | λ | trainR2 | SR | cross4 |
|---|---|---|---|---|
| MSE s10k | 1e-3 | 0.999 | 39 | 0.78 |
| MSE s10k | 1e-1 | 0.996 | 13 | 0.97 |
| MSE s25k | 1e-3 | 0.999 | 53 | 0.59 |
| MSE s25k | 1e-1 | 0.997 | 17 | 0.92 |

## 11. ET3 v2 (from the MSE s10k starting point +150k; start cos 0.84 / cleanErr 0.0109 / SR 51)

| variant | endpoint geometry cos/rotsv/rot@amp | cleanErr p50 | SR | cross4 | SR\|cross4 | drift(d<2) | drift([2,4)) |
|---|---|---|---|---|---|---|---|
| a standard MSE | 0.41/0.179/0.162 | 0.0055 | 62 | 0.42 | 10 | +0.0118 | +3.67 |
| b switch to MIP | 0.72/0.055/0.020 | 0.0030 | 84 | 0.21 | 24 | +0.0095 | −0.266 |
| c lr=1e-5 | 0.87/0.080/0.011 | 0.0069 | 73 | 0.32 | 16 | +0.0192 | −0.192 |
| d wd=1e-2 | 0.41/0.175/0.157 | 0.0053 | 64 | 0.38 | 5 | +0.0121 | −0.210 |
| e frozen trunk | 0.82/0.073/0.008 | 0.0115 | 51 | 0.60 | 18 | +0.0257 | −0.065 |

Per-variant geometry timeline (cos, +26k/+51k/+76k/+101k/+126k/+150k):
- a: 0.60 / 0.37 / 0.41 / 0.40 / 0.41 / 0.41
- b: 0.80 / 0.74 / 0.73 / 0.72 / 0.71 / 0.72
- c: 0.85 / 0.87 / 0.87 / 0.87 / 0.87 / 0.87
- d: 0.61 / 0.42 / 0.39 / 0.40 / 0.41 / 0.41
- e: 0.83 / 0.83 / 0.82 / 0.82 / 0.82 / 0.82

b's cleanErr timeline: 0.0060→0.0041→0.0033→0.0031→0.0031→0.0030;
a: 0.0076→0.0083→0.0056→0.0058→0.0056→0.0055; e: 0.0117→…→0.0115 (flat);
c: (from 26k) →0.0071 (52k) →0.0069 (76k) →0.0069 (150k).

Per-variant gain@bad1 / rot@mseamp timeline (+26k → +150k):
- a: −.0095/.059 → +.0009/.147 → +.0120/.157 → +.0098/.167 → +.0077/.162 → +.0093/.162
- b: −.0103/.017 → −.0099/.016 → −.0104/.017 → −.0117/.017 → −.0116/.019 → −.0111/.020
- c: −.0118/.007 → −.0128/.009 → −.0128/.011 → −.0127/.011 → −.0127/.011 → −.0127/.011
- d: −.0088/.048 → +.0047/.148 → +.0070/.165 → +.0041/.161 → −.0010/.158 → −.0011/.157
- e: −.0123/.008 → −.0125/.008 → −.0129/.008 → −.0130/.008 → −.0131/.008 → −.0131/.008

Endpoint poseig: a [−.021,−.010,+.021]; b [−.018,−.013,0.0]; c [−.017,−.012,−.000] (negdef);
d [−.026,−.013,+.016]; e [−.016,−.011,−.001] (negdef).

## 12. Other controls (earlier rounds)

- Composition: frozen MSE proposal + standalone denoiser refinement: static bias 0.495→0.141 (shared MIP-step1 0.158);
  closed loop 98/100 (MSE alone 69; shared MIP step1 94 / full 96).
- Unconditional manifold projection: destroys every policy; frozen unconditional manifold in the loss: 49%.
- Capacity: MSE at 3.5×/10× parameters → 69→49→46; MIP-step1@20M = 94.
- Weight dose (tube, two-view): 1× (EQW) 83 / 4× (TAU50) 90 / 100× (MIP) 94–99 / 10⁴× (TAU99) 95.
- G_GT anchor density 40→20k demos: R² 0.549→0.572 (flat); |δz_t|/|δz|=0.14.
- abs representation: A_abs = I + A_delta verified to hold; abs-MSE 4% (response R²=0.085), abs-MIP 64% (0.353);
  both learn λ≈GT (0.68/0.72 vs 0.78).
- DART: MSE+DART-2k/4k/6k = 92/97/(6k data in an earlier round); dual-noise 4%→24% (human 84%).
- Seed error: MSE-2k 73±5.3, MIP-2k 96.7±2.1.

## 13. Requested but not measured (explicit list)

1. Static bias bins for trajectory/ablation ckpts: only measured in the ridge-λ sweep (§5), not per ckpt;
2. ‖â−a_GT‖ on rollout states (before-cross per-step action error): rollout states have no scripted GT,
   so it cannot be measured directly; the operationalized surrogate = drift(d<2) (given in §1/§6/§11);
3. "operator gain scale vs G_GT": not tabulated separately; the numeric magnitude of poseig (vs G_GT's [−.024,−.008,−.001])
   can serve as a proxy;
4. Closed-loop SR at intermediate ET3 ckpts: only endpoints measured (advisor annotated "if feasible");
5. cleanErr p90: the loss probe outputs p50/mean/p95; p90 was not printed (p95 is given);
6. Full 300k trajectory snapshots for ZEROIN/SCRAMBLE: not saved (only the early 25k short runs + endpoints);
7. EXP7 (abs feature-ridge + implied delta), EXP9A/C (feature-matching distillation training),
   EXP10A/C (kNN/phase-anchor robustness, stage-conditioned operators): paused per advisor instruction, not started.
