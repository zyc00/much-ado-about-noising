# GR00T N1.7 / RoboCasa GR1 Tabletop — HT vs flow results

All closed-loop numbers: official benchmark protocol (`run_gr00t_server.py --use-sim-policy-wrapper` +
`rollout_policy.py --max-episode-steps 720 --n-action-steps 8 --n-envs 5`), 24 tasks × 20 episodes,
success = mean of per-task rates. Both methods trained with NVIDIA's released recipe
(`examples/finetune.sh`, 60k steps, global batch 512, lr 1e-4, warmup 5%); only the action-head
objective differs (`--loss-type=flow` vs `hetero_t`, ht_sbias −1.088, ν=2).

## Success-rate curve (24-task average)

| steps | flow (4 NFE) | HT (1 NFE) |
|---|---|---|
| 2,000 | 1.46 % (fresh control run, 60k schedule stopped at 2k) | 4.45 % |
| 16,000 | 3.75 % | 29.0 % |
| 32,000 | — (ckpt rotated; retrain waived) | 43.0 % |
| 50,000 | 41.3 % | — |
| 52,000 | — | 48.3 % |
| 60,000 | 44.1 % | 44.7 % |
| NVIDIA published final (60k) | 44.5 % | — |

Regression-family controls (same recipe, only --loss-type changed; same 24x20 eval):
| steps | MSE | L1 |
|---|---|---|
| 2k | 6.4 % | 3.3 % |
| 16k | 19.6 % | 12.2 % |
| 32k | 23.9 % | — |
| 60k | 37.8 % | 17.2 % |
HT 44.7 > flow 44.1 >> MSE 37.8 >> L1 17.2 at the endpoint. L1 is not a viable
head on GR1 (27 pts below HT) — the OFT near-parity with L1 is specific to
clean LIBERO specialist data on a recipe tuned around L1, not a general
property of L1 regression. Mechanism for L1's collapse not identified
(candidate: scale-blind constant-magnitude gradients on high-dimensional
heteroscedastic targets — hypothesis only). Run dirs ft_mse / ft_l1; evals
evalout_{mse,l1e}{2k,16k,32k,60k|end}_*.log.

### Schedule-length family (HT): shorter schedules with COMPLETED LR decay beat the 60k endpoint
Each row is a separate run whose cosine schedule ENDS at that step (so decay is
complete), evaluated at its own endpoint on the same 24x20 protocol:
| schedule | endpoint SR |
|---|---|
| 30k | 46.1 % |
| 45k | 46.8 % |
| **50k** | **47.4 %** |
| 60k | 44.7 % |
| 120k | 45.2 % (also 45.3 % @60k, 41.3 % @90k within that run) |
Reading: HT's best full-decay endpoint is 47.4 % at a 50k schedule — ABOVE the
60k endpoint (44.7) and above flow's 60k endpoint (44.1) and NVIDIA's published
44.5. The 60k schedule is not HT's optimum; the 120k schedule is worse still,
i.e. HT saturates early on GR1 and over-long schedules cost ~2-3 pts.
CAVEAT for reporting: we do NOT have a matched 50k-schedule FLOW run (flow was
trained on the 60k schedule only), so 47.4 vs 44.1 compares HT-at-its-best-
schedule against flow-at-its-only-schedule. The strictly matched comparison
remains the 60k endpoint: HT 44.7 vs flow 44.1 vs published 44.5.
Data evalout_ht{30k,45k,50k}end_*.log, evalout_ht120*_*.log.

- HT@52k = 48.3 % (per-task, fanout order: 0.50 0.30 0.10 0.10 0.10 0.50 0.80 0.55 0.857 0.70
  0.55 0.50 0.15 0.55 0.30 0.40 0.40 0.45 0.70 0.80 0.70 0.85 0.55 0.190) — best single
  checkpoint measured; 52k-vs-60k gap (48.3 vs 44.7) is ~1.1 joint SE, treated as plateau noise
  unless a second eval seed confirms. Under DP-style best-checkpoint selection the HT headline
  would be 48.3 vs flow's best 44.1.

- HT@60k = 44.7 % vs flow@60k = 44.1 % vs published 44.5 %: endpoint parity within eval noise at
  1 NFE, with no late-training sag (43.0 → 44.7 across 32k→60k, final LR decay included).
- HT@60k per-task (fanout order): 0.60 0.45 0.05 0.190 0.10 0.55 0.75 0.15 0.75 0.85 0.35 0.65
  0.190 0.45 0.238 0.40 0.45 0.40 0.75 0.60 0.60 0.60 0.50 0.10.
- MSE (single-pass, homoscedastic) ablation: 6.4 % @2k (ahead of HT early — expected: the t-NLL
  down-weights large residuals during initial mean-fitting), crossover by 16k: 19.6 % vs HT
  29.0 % (−9.4 pts, ≈4 SE). Both 1-NFE regressors ≫ flow (3.75 %) at 16k — the convergence
  advantage is the single-pass objective family; the Student-t adds the further margin. MSE@16k
  per-task (fanout order): 0.20 0.095 0.00 0.05 0.10 0.15 0.35 0.143 0.25 0.25 0.10 0.25 0.15
  0.35 0.10 0.143 0.15 0.25 0.25 0.30 0.30 0.45 0.190 0.143.
- MSE@60k = 37.8 % — full curve 6.4 → 19.6 → 23.9 → 37.8. The apparent 32k plateau was
  schedule-position (LR-decay consolidation added +13.9, as for flow and HT); final ordering
  HT 44.7 > MSE 37.8 ≈ flow-K1 37.4 (both 1-NFE-without-the-objective land together), with the
  Student-t margin over MSE at +6.9 pts (~3 SE). MSE@60k per-task (fanout order): 0.50 0.25
  0.190 0.286 0.05 0.40 0.50 0.50 0.762 0.60 0.50 0.45 0.190 0.35 0.143 0.05 0.50 0.35 0.55
  0.40 0.60 0.55 0.30 0.10. MSE@32k per-task (fanout order): 0.60 0.35 0.40 0.25
  0.10 0.35 0.35 0.40 0.40 0.35 0.143 0.190 0.15 0.30 0.00 0.190 0.143 0.227 0.15 0.25 0.15
  0.143 0.143 0.00. - Schedule-length sweep (HT endpoints at full decay, same recipe): 30k → 46.1 %, 45k → 46.8 %,
  50k → 47.4 %, 60k → 44.7 %, 120k → 45.2 %. Endpoints flat at 44.7–47.4 across a 4× schedule
  range, all ≥ published flow final — HT saturates by the 30k schedule; no overfitting in any
  arm, including at double the standard budget. ht50k-endpoint per-task (fanout order): 0.70
  0.70 0.40 0.15 0.286 0.50 0.70 0.30 0.85 0.55 0.40 0.60 0.30 0.45 0.10 0.40 0.30 0.40 0.65
  0.70 0.70 0.55 0.50 0.20. ht120k-endpoint per-task (fanout order): 0.65 0.50 0.25 0.45
  0.25 0.50 0.55 0.45 0.90 0.70 0.35 0.524 0.40 0.35 0.10 0.190 0.40 0.35 0.65 0.55 0.50 0.80
  0.30 0.190. ht45k
  per-task (fanout order): 0.75 0.35 0.30 0.40 0.30 0.45 0.75 0.40 0.85 0.55 0.238 0.55 0.40
  0.35 0.227 0.286 0.190 0.30 0.90 0.65 0.75 0.65 0.35 0.30.
  ht30k per-task (fanout order): 0.55 0.70 0.50 0.15 0.10 0.20 0.65 0.25 0.810 0.80 0.05 0.75
  0.35 0.25 0.15 0.40 0.50 0.30 0.80 0.70 0.75 0.80 0.45 0.10.
- Pretraining ablation (random-init action head, frozen pretrained backbone, 60k schedule, 24-task
  fanouts): flow-scratch 14.0 %, HT-scratch 33.5 % — 2.4× at identical init (~8 SE), so the HT
  convergence advantage is intrinsic to the objective, not inherited from the flow-pretrained head.
  Asymmetric pretraining dependence: +30.1 pts for flow (14.0→44.1) vs +11.2 for HT (33.5→44.7).
  flow-scratch per-task (fanout order): 0.00 0.00 0.00 0.05 0.00 0.00 0.182 0.238 0.30 0.143
  0.190 0.10 0.182 0.05 0.05 0.25 0.05 0.182 0.30 0.190 0.35 0.35 0.15 0.05. ht-scratch:
  0.55 0.15 0.00 0.00 0.05 0.40 0.60 0.40 0.80 0.45 0.10 0.50 0.30 0.40 0.15 0.40 0.190 0.190
  0.45 0.50 0.50 0.45 0.45 0.05.
- HT 120k-schedule run (budget-scaling / sag stress test): 35.9 (16k) → 32.6 (32k) → 45.3 (60k,
  mid-decay) — already at the 60k-schedule endpoint level (44.7) at half schedule; corroborates
  the ~45 % HT level across two independent runs. Schedule-position lesson: at matched steps,
  SR depends strongly on cosine position (ht120@32k 32.6 vs ht3@32k 43.0), so all headline
  comparisons are schedule-matched. Endpoint (120k, full decay): 45.2 % — no sag at 2× the
  standard budget. MSE@60k done (37.8).

- flow@60k = 44.1 % vs published 44.5 %: reproduction confirmed within eval noise (SE ±2.3 pt at
  480 episodes) — recipe, training, and harness all validated against NVIDIA's own benchmark.
- HT@16k vs flow@16k is steps-matched: 7.7× success at identical budget.
- L1 arm (OFT-style L1 on the identical single-pass path, 60k schedule): L1@2k 3.70 %,
  L1@16k 12.2 % (24-task fanouts). Mid-training ordering HT 29.0 > MSE ~19.6 > L1 12.2 >
  flow 3.75 at 16k: plain L1 converges SLOWER than MSE on this backbone, so the single-pass
  early-convergence advantage is not generic regression — the heteroscedastic weighting
  carries it (HT 2.4× over L1 at matched steps). L1@2k per-task (fanout order): 0.0 0.0 0.0
  0.0 0.0 0.0 0.095 0.0 0.143 0.0 0.0 0.1 0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.0 0.3 0.05 0.0 0.0.
  L1@16k per-task: 0.0 0.0 0.0 0.0 0.1 0.05 0.182 0.1 0.45 0.143 0.0 0.1 0.143 0.227 0.0
  0.238 0.1 0.190 0.15 0.25 0.15 0.25 0.1 0.0. L1@60k endpoint = 17.2 % (final; per-task
  fanout order: 0.30 0.05 0.0 0.10 0.0 0.05 0.60 0.30 0.50 0.35 0.095 0.35 0.0 0.143 0.0
  0.182 0.0 0.15 0.15 0.10 0.25 0.40 0.05 0.0). Full L1 curve 3.7 → 12.2 → 17.2: never
  recovers — 2.6× below HT and <1/2 of MSE at matched steps on the identical single-pass
  path. Cross-benchmark contrast: the same L1 objective scores ~98 % on OpenVLA-OFT+LIBERO —
  loss choice is regime-dependent, and the heteroscedastic weighting is what makes
  single-pass regression competitive on heavy-tailed data (see residual battery).
- HT@32k (43.0 %) is statistically indistinguishable from flow's converged endpoint (44.1 %),
  reached at 53 % of the training budget and 1/4 the inference NFE.
- flow@60k per-task (fanout order): 0.30 0.35 0.00 0.20 0.05 0.238 0.476 0.40 0.75 0.60 0.190
  0.75 0.35 0.45 0.190 0.35 0.40 0.45 0.80 0.80 0.90 0.80 0.65 0.143.
- HT@32k per-task (fanout order): 0.60 0.70 0.15 0.238 0.05 0.60 0.60 0.35 0.60 0.55 0.190 0.45
  0.286 0.25 0.10 0.40 0.40 0.35 0.80 0.70 0.65 0.70 0.35 0.25.

## Inference cost (A800-SXM4-80GB, PyTorch eager, batch 1, per 8-step chunk; repo benchmark_inference.py, 50 iters)

| config | data | backbone | action head | E2E | replan rate |
|---|---|---|---|---|---|
| flow K=4 (deployed) | 2 ms | 40 ms | 72 ms | 114 ms | 8.8 Hz |
| flow K=1 (truncated) | 2 ms | 40 ms | 20 ms | 62 ms | 16.1 Hz |
| HT single-pass | 2 ms | 39 ms | 19 ms | 60 ms | 16.6 Hz |

- E2E 1.9× faster; action head 3.8×; replan latency 114→60 ms.
- Head-excluded cost is the shared frozen backbone — any backbone optimization widens the E2E gap
  toward the 3.8× head ratio.
- NFE ablation (closed-loop, 24 tasks): flow@60k K=1 = 37.4 (−6.7 vs K=4), K=4 = 44.1 (native),
  K=8 = 29.8 (−14.3) — NON-MONOTONE in NFE with a brittle optimum at the shipped K=4; K=8 also
  degrades open-loop (MSE 0.067→0.082, +22 %), so it is a model property (Beta(1.5,1) t-sampling
  leaves off-grid velocity regions under-trained), not an eval artifact. flow@50k: K=1 = 36.6,
  K=4 = 41.3, K=8 = 21.7 (−19.6) — the K=8 penalty grows away from convergence, consistent with
  under-trained off-grid velocities. HT has no NFE dial: 44.7 at 1 NFE / 60 ms. flow@60k-K8 per-task (fanout order):
  0.15 0.05 0.00 0.00 0.00 0.15 0.55 0.30 0.75 0.50 0.30 0.55 0.10 0.20 0.15 0.20 0.25 0.30
  0.40 0.45 0.40 0.75 0.45 0.20. flow@50k-K1 per-task
  (fanout order): 0.60 0.55 0.00 0.35 0.10 0.50 0.75 0.40 0.75 0.65 0.25 0.30 0.15 0.30 0.00
  0.40 0.30 0.30 0.429 0.30 0.50 0.45 0.30 0.15. Per-task K=1 rates (fanout order): 0.25 0.190
  0.05 0.190 0.35 0.35 0.65 0.35 0.75 0.55 0.190 0.45 0.45 0.45 0.20 0.50 0.05 0.45 0.75 0.45
  0.35 0.60 0.35 0.05.

## Open-loop (3 train trajs, PnPBottleToCabinetClose, execution-horizon 8)

| checkpoint | MSE | MAE |
|---|---|---|
| flow@16k, K=4 | 0.209 | 0.107 |
| flow@16k, K=1 | 0.068 | 0.099 |
| HT@2k, 1-pass | 0.103 | 0.084 |

## Per-task success (fanout order = alphabetical = NVIDIA README order)

| # | task | NVIDIA final | flow@16k | flow@50k | HT@2k | HT@16k |
|---|---|---|---|---|---|---|
| 1 | PnPBottleToCabinetClose | 70.0 | 0 | 25 | 0 | 15 |
| 2 | PnPCanToDrawerClose | 70.0 | 0 | 20 | 0 | 20 |
| 3 | PnPCupToDrawerClose | 35.0 | 0 | 10 | 0 | 10 |
| 4 | PnPMilkToMicrowaveClose | 45.0 | 0 | 10 | 0 | 35 |
| 5 | PnPPotatoToMicrowaveClose | 40.0 | 0 | 10 | 0 | 35 |
| 6 | PnPWineToCabinetClose | 65.0 | 0 | 35 | 0 | 15 |
| 7 | ...CuttingboardToBasket | 10.0 | 5 | 70 | 5 | 40 |
| 8 | ...CuttingboardToCardboardbox | 30.0 | 0 | 35 | 10 | 25 |
| 9 | ...CuttingboardToPan | 40.0 | 5 | 75 | 10 | 70 |
| 10 | ...CuttingboardToPot | 45.0 | 5 | 80 | 10 | 45 |
| 11 | ...CuttingboardToTieredbasket | 25.0 | 10 | 19.0 | 0 | 10 |
| 12 | ...PlacematToBasket | 40.0 | 10 | 70 | 5 | 60 |
| 13 | ...PlacematToBowl | 40.0 | 0 | 35 | 0 | 9.5 |
| 14 | ...PlacematToPlate | 40.0 | 0 | 66.7 | 0 | 20 |
| 15 | ...PlacematToTieredshelf | 18.2 | 0 | 5 | 0 | 20 |
| 16 | ...PlateToBowl | 50.0 | 5 | 45 | 5 | 14.3 |
| 17 | ...PlateToCardboardbox | 35.0 | 10 | 35 | 0 | 19.0 |
| 18 | ...PlateToPan | 40.0 | 5 | 25 | 18.2 | 5 |
| 19 | ...PlateToPlate | 75.0 | 10 | 80 | 5 | 55 |
| 20 | ...TrayToCardboardbox | 60.0 | 0 | 40 | 0 | 30 |
| 21 | ...TrayToPlate | 50.0 | 10 | 65 | 15 | 60 |
| 22 | ...TrayToPot | 45.0 | 10 | 60 | 13.6 | 45 |
| 23 | ...TrayToTieredbasket | 55.0 | 5 | 40 | 5 | 28.6 |
| 24 | ...TrayToTieredshelf | 45.0 | 0 | 35 | 5 | 10 |
| — | **mean** | **44.5** | **3.75** | **41.3** | **4.45** | **29.0** |

Notes: flow@50k lags NVIDIA's final mainly on the six two-stage PnP→Close tasks (rows 1–6) —
consistent with the final 10k of LR decay mattering most there; it already exceeds their final on
many single-stage Posttrain tasks. Some cells are n=21–22 episodes (vec-env rounding), shown as
fractional percentages.

Provenance caveats: flow@16k/50k come from the lineage originally mislabeled ft_ht (loss_type
whitelist bug, objective verified flow by sigma-key absence + config audit; renamed ft_flow and
resumed under the patched harness with explicit --loss-type=flow, clean key load). HT numbers from
ft_ht3 (post-fix run; sigma_decoder confirmed present in checkpoints and training). HT@32k pending
completion; flow@2k control and both 60k endpoints pending — table to be updated when they land.

## pi0.5 (LeRobot port) — single-pass HT/MSE vs flow, LIBERO generalist (2026-08-13)

Recipe: finetune lerobot/pi05_libero_base on HuggingFaceVLA/libero (all 4 suites combined,
1693 eps), 30k steps, global batch 32, bf16 + grad ckpt, 8xA800, ~9 h/arm. Single-pass head:
one Euler step from zero noise (a_hat = -v(0,1)); MSE = plain masked MSE; HT = Student-t NLL
nu=2, sbias=-0.396 (softplus^-1 of probed init-residual RMS 0.5146). Eval: lerobot_eval,
1000 episodes per suite per policy, batch 10 envs.

| suite    | HT 30k | MSE 30k | ref flow ckpt (finetuned_v044, same harness) |
|----------|--------|---------|----------------------------------------------|
| spatial  | 98.1   | 96.9    | 86.4                                         |
| object   | 99.8   | 99.8    | 91.2                                         |
| goal     | 97.6   | 97.6    | 92.9                                         |
| long(10) | 94.7   | 92.7    | (pending)                                    |
| avg      | 97.6   | 96.8    |                                              |

- HT >= MSE on every suite; separation appears on the hard suites (long +2.0 ~1.8 SE,
  spatial +1.2); object/goal saturate for both — consistent with the clean-data-parity story.
- 10k-intermediate (spatial, 1000 eps): HT 96.9 / MSE 97.1 — parity at 1/3 schedule.
- External references: OFT per-suite L1 specialists avg 97.1 (long 94.5); pi0 fine-tuned
  94.2 avg (long 85.2). Our HT generalist: 97.6 avg / 94.7 long at 1 NFE.
- The v044 reference checkpoint publishes no eval numbers; 86.4-92.9 is its measured
  performance on our stack (10 denoising steps). Recipe for it is unknown — disclosed as a
  reference point, not a controlled baseline (the controlled comparison is HT vs MSE).
- Infra notes in memory: pi05-ht-experiment (tokenizer gating, eval asset fixes, OOM fixes).

## [DISCARDED 2026-08-15] Cosmos Policy (Predict2-2B, RoboCasa): action-NFE baseline
**Excluded from the paper's claims per user decision:** this ablation reduces the denoising
steps of THEIR already-trained diffusion policy. It says nothing about our HT loss — no HT
model is involved — so it does not support the HT claim and is not reported. Measurements
retained below for the record only.

### (retained for record) action-NFE baseline on the released checkpoint

Zero-code ablation, released Cosmos-Policy-RoboCasa-Predict2-2B.pt, their exact eval
harness (config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference, chunk 32,
open-loop 16, seed 195 deterministic), num_denoising_steps_action K=5 (their shipped
default) vs K=1. Future-state and value frames use 1 step in both arms (their default).
100 episodes per task per arm. Task subset: 2 easy + 2 medium of their 24.

| task | K=1 | K=5 |
|---|---|---|
| TurnOffMicrowave | 1.00 | 1.00 |
| CoffeePressButton | 0.96 | 0.94 |
| OpenSingleDoor | 0.74 | 0.76 |
| PnPCounterToCab | 0.56 | 0.60 |
| mean | 0.815 | 0.825 |

Reading: 5-step vs 1-step action denoising differs by 1.0 pt mean (per-task differences
2-4 pts in both directions, within binomial noise at 100 eps). Iterative refinement of
the action latent contributes ~nothing at execution on this checkpoint; their own config
already single-steps the other two generated modalities. Note K=1 is still a stochastic
diffusion step (noise init) trained with the diffusion objective — the HT post-training
variants (A: action single-pass + t-NLL; B: + video single-pass) attack the training
objective next. Reference: published paper average 67.1 over all 24 tasks; this 4-task
subset is easier (82.5 at K=5) and is used for fast comparisons only.

## GR00T+GR1 nu ablation (user question: is nu=2 or nu=0.5 stronger on a real task?)
HT nu=0.5 arm, identical recipe to the recorded nu=2 arm (60k, batch 512,
sbias -1.088, only --ht-df changed), same 24-task x 20-ep eval convention:
| checkpoint | HT nu=2 | HT nu=0.5 |
|---|---|---|
| 16k | 29.0 | UNAVAILABLE (ckpt rotated away) |
| 60k | 44.7 | 44.8 |
The 16k convergence-speed comparison cannot be run: GR00T's finetune keeps only
the last 5 checkpoints (52k-60k for a 60k schedule), so ft_htnu05/checkpoint-16000
was deleted during training. The 2026-08-14 attempt failed for exactly this reason
(server FileNotFoundError -> client ZMQ error; 7 pods then held GPUs idle for 11 h
until killed). Recovering the point requires retraining the nu=0.5 arm (~9 h on
8 GPUs) and was NOT done: the endpoint is already a tie and the ablation is
secondary. Verdict stands on the 60k endpoint.
Endpoint verdict: statistical tie (0.1 pts). Neither the toy-multimodal advantage
of small nu nor the Cauchy-plateau optimization penalty materialized — consistent
with GR1's converged per-state action distributions being unimodal-with-tails,
where all location estimators coincide. Framing: the nu dial matters exactly when
the data carries the constructed structures (skew/branches/pauses) and measurably
does not otherwise; nu=2 is a safe default, not a tuned choice. No parking
pathology observed (idle-cell hypothesis negative on GR1).
