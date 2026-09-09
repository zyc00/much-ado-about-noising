# Experiment section status (2026-09-03)

Plan blocks as specified: (1) VLA Flow/HT/MSE, (2) RoboMimic HT/MIP/Flow,
(3) real robot, (4) data multimodality,
(5) learned multimodality, (6) loss ablation.
Sources: `vla_results_table.md`, `robomimic_5method_table.md`, `multimodality_table.md`,
`main_table.md`, `main_table_3seeds.md`, `gr1_ht_mechanism.md`.

## 1. VLA: Flow vs HT vs MSE

| stack | flow | HT | MSE | status |
|---|---|---|---|---|
| GR00T GR1 (d=232) | 44.1 @60k | 47.5 / 51.5 @60k | 37.8 @60k | complete |
| GR00T WidowX (56) | 57.1 @20k | 62.9 / 67.4 @20k | — | MSE missing |
| GR00T fractal (56) | 67.3 @20k | 63.5 @20k | — | MSE missing; HT loses |
| OpenVLA-OFT (56), 4 suites | no flow head | 96.0 avg | 84 @20k, budget-mismatched | flow n/a; MSE not matched |
| pi0.5 (350) | published 96.9 | 97.6 @30k | 96.8 @30k | complete |
| Cosmos3 (160) | 96 @2k | 96 @2k | — | MSE missing |

Done: 6/6 HT, 5/6 flow, 2/6 MSE.
Missing: MSE on WidowX, fractal, Cosmos3; a budget-matched MSE on OFT.
OFT has no flow arm by construction (its diffusion head gives 0.0 SR at T_test=1).
In flight (fractal, chasing the one HT loss): see `fractal_debug.md`. Seeded-protocol re-evals of
c=1.5 and beta=0.5, beta=1.0 training, deterministic-flow discriminator, released checkpoint in our
harness, and HT + flow on NVIDIA's actual released recipe (40k steps, dropout 0.2).

## 2. RoboMimic: HT vs MIP vs Flow

Complete as of 2026-09-03: 3 backbones (DiT, Chi-Tf, Chi-UNet) x 5 heads
(Flow, Regression, Straight Flow, MIP, HT) x 10 tasks x {state, image} = 300 cells,
all in our harness under one protocol. Table: `robomimic_5method_table.md`.

| head | State (best/late) | Image (best/late) |
|---|---|---|
| Flow | **0.886/0.824** | 0.834/0.782 |
| MIP | 0.857/0.806 | 0.852/0.759 |
| HT (ours) | 0.850/0.783 | **0.884/0.809** |
| Straight Flow | 0.808/0.740 | 0.785/0.721 |
| Regression | 0.800/0.739 | 0.783/0.718 |

Result: HT wins on image only, and the margin is carried by transport-ph (+0.38 vs flow),
transport-mh (+0.07) and toolhang (+0.06). On state, flow leads on 8/10 tasks and HT is
within 0.01 of plain regression on 8/10.

Outstanding: 4 cells whose late-average collapsed while best stayed high (image Chi-Tf MIP
square-ph 0.82/0.21 and square-mh 0.92/0.04; state Chi-UNet Straight Flow toolhang 0.50/0.00;
image Chi-UNet Straight Flow transport-mh 0.50/0.22) — re-run or footnote. Seed count and
episodes/eval for the 300-cell grid are not recorded. Column labels for tasks 5-8 unverified
(mh reads above ph, reverse of published tables).

## 3. Real robot

Two tasks planned. Push-T is done; Hang mug is not started.

| task | flow | HT | MSE | protocol |
|---|---|---|---|---|
| Push-T (real) | 86 | **88** | 54 | 50 episodes; initial pose randomized -5 to 5 in x and y, +/-30 deg rotation |
| Hang mug | — | — | — | not started |

Reading: the single-pass heads split — HT matches flow while plain MSE collapses by ~33 points,
so the gap is the loss, not the number of forward passes. At n=50 the standard error is about
4.6 points, so flow 86 vs HT 88 is a tie and the MSE gap is far outside noise.

Missing: Hang mug for all three heads; seed count for the Push-T cell; units for the -5 to 5
translation range are not recorded here.

## 4. Data: how multimodal is the data?

Done: push-T recurrent-state conditional-action analysis — unimodal at recurrent states
(epsilon-ball, >=24 visits from >=5 episodes); the earlier "82% bimodal" kNN reading was a
null-criterion artifact. Separately, fractal/bridge instruction-frequency distributions
(fractal Gini 0.650 over 599 instructions; bridge 0.482 over 19,747) — a task-skew result,
not conditional multimodality.

Missing: everything else. No data-side probe on robocasa/GR1, bridge, fractal, LIBERO, or on
any robomimic dataset. The `-ph` vs `-mh` contrast (single vs multiple human operators) is the
sharpest available test and is entirely absent; it is also the natural explanation for HT and
MSE being indistinguishable on 8/10 robomimic state tasks. Offline, no training needed.

## 5. How multimodal does the flow model learn?

Complete for push-T MIP/HT, GR1 flow, pi0.5 flow, and WidowX flow. GR1 and pi0.5 each have a
3000-state training screen with Monte-Carlo calibration and fresh-sample verification of the
extremes. They contain zero alternative-action states; all resolved structure is hand/gripper
event timing (95% upper bound 0.1% per stack for the searched training states).

WidowX adds 3000 Bridge states, 1000 actual Flow-rollout states, a 100-state uniform on-policy
confirmation set (K=32 discovery + independent K=64 confirmation), and a 42-cell adjacent-frame
probe. In the uniform sample: 27 full, 22 gripper, and 6 arm tests are positive. A geometric/raw
trajectory audit leaves 3/100 clearly resolved arm-dominated profiles. An executed-prefix check
finds 10 arm positives and 6 resolved profiles. The direct same-state fork then screens 165 states,
deep-tests 66, and intervenes on four confirmed modes (two episodes per task; 12 branches per
component).
Among the three modes that actuate the arm, pose decoding drops from 65/72 during the forced chunk
to 40/72 mid-rollout and 31/72 late; all late spatial between/within ratios are 0.24--0.35 and no
late-path, endpoint-metric, or success test is significant. The fourth action mode causes no
measurable arm motion. Therefore Section 1 should say "little persistent semantic-plan
multimodality," not "the distributions are unimodal." Full record:
`widowx_mm/widowx_experiment_report.md`.

Added continuous/contact-state audit: on the same 30 adversarial GR1 states,
held-out 2G beats 1G on independent PC1 for 1/30 arm+waist distributions but
16/30 articulated-hand and 16/30 all-joint distributions; exact normality is
rejected for 1/30, 26/30 and 26/30, respectively. On pi0.5 arm, 1G wins in
32/36 adversarial states. WidowX's 10 detected prefix arm cases all favor 2G,
while the other 90 favor 1G on average. Full record:
`gaussianity/action_basin_report.md`.

Added exact-state continuous-mean intervention on WidowX: 40 on-policy states,
64 draws for the arm mean and 8 fresh reference branches, with the gripper
matched rather than averaged. The mean falls outside the empirical 95%
reference-path envelope in 0/40 states. Median path-centroid ratios are
0.415 position and 0.382 orientation, versus 0.770 and 0.575 for the sampled
arm medoid. Full record: `widowx_mm/widowx_arm_mean_report.md`.

Optional additions: fractal and RoboMimic flow-head probes. OFT has no flow head. Cosmos3 remains
excluded with the stated reason (seeded world-model latent noise).

## 6. Ablations: HT vs Student-t vs HG vs L1 vs Huber

| stack | HT | Student-t (homosc.) | HG | L1 | Huber | MSE |
|---|---|---|---|---|---|---|
| GR1 @60k | 47.5 / 51.5 | missing | 20.6 @22k, mismatched | 17.2 | 37.4 / 40.7 | 37.8 |
| OFT-long | 94.0 @150k | missing | 73 @50k, mismatched | released 94.5 | 97 / 95 @50k, mismatched | 84 @20k, mismatched |
| pi0.5 | 97.6 | missing | missing | missing | missing | 96.8 |
| WidowX / fractal / Cosmos3 | done | missing | missing | missing | missing | missing |
| RoboMimic (3 backbones x 2 obs) | done | missing | missing | missing | missing | done (Regression) |

Done: L1 and Huber on GR1 (matched, 60k); L1 on OFT; MSE on GR1/pi0.5/RoboMimic;
Straight Flow on RoboMimic as the single-pass-flow control.
Missing: homoscedastic Student-t on every stack (0/6) — the ablation that separates the sigma
head from the heavy tail. HG and Huber exist only at mismatched budgets. No single stack
currently carries all five arms at one budget.

### 6b. Robust-loss ladder (budget-matched; 2026-09-07)

Five objectives on the same recipe, same steps, same eval. OFT-long = 10 tasks x 10 eps; GR1 = 24 x 20 official.

| stack | HT (c=2) | L1 | simple Huber | hetero-Huber | hetero-L1 (sum|r|/sigma + d log sigma) | MSE | HG |
|---|---|---|---|---|---|---|---|
| OFT-long @20k | 89 (nu=224); nu=1024 94 | 66 | 88 | 84 | **69** (`ev20000_hl1`) | 84 | 59 |
| OFT-long @50k | -- | -- | 97 | 95 | not reached: arm stopped by the user 2026-09-07 at step 40446/50005 | -- | 73 |
| GR1 @60k | 47.5 / 51.5 | 17.2 | 37.4 | 40.7 | **44.5** (`gr1hl1`, 24x20 @60k) | 37.8 | -- |

OFT-long @20k: hetero-L1 (69) lands between L1 (66) and MSE (84), 20 under HT c=2 (89); the sigma head does not
rescue the L1 kernel at this budget the way it partly rescues Huber on GR1 (37.4 -> 40.7).
GR1 @60k: hetero-L1 44.5 (per task, fanout order: 0.80 0.45 0.15 0.25 0.35 0.40 0.70 0.30 0.90 0.65 0.05 0.45 0.40 0.45 0.20 0.55 0.429 0.45 0.75 0.45 0.65 0.65 0.25 0.00), above MSE 37.8, simple Huber 37.4 and
hetero-Huber 40.7, level with flow 44.1, 3.0 under HT nu=464 (47.5) and 7.0 under HT c=2 (51.5; single seed, SE ~2.3).
On GR1 the ordering is L1 17.2 < Huber 37.4 = MSE 37.8 < hetero-Huber 40.7 < hetero-L1 44.5 < HT 47.5 / 51.5:
sigma normalization is worth more with a heavier kernel, and the Student-t gate adds the last 3-7 points. The two
stacks disagree on hetero-L1 (GR1 second-best, OFT-long @20k second-worst); the 50k OFT eval will say whether
that is a budget effect.

## Extra, already done (not in the plan but available)

Mechanism section: gradient-share tables over the surprise tail for HG/flow/HT (E1-E8,
including the across-training version E8a-T), gate-scale theory-vs-measured (E6), ML-nu vs
SR-optimal nu (E5), the GR1 mechanism study (uniform precision gain on continuous arm
channels), the decile probe (HT is 26% worse on what it suppresses, 46% better on what it
favors), and the fractal sigma/gate probe (gate inert, sigma is the between-family knob,
log-log slope +1.07).

## Recommended queue

| # | run | cost | closes |
|---|---|---|---|
| 1 | robomimic ph-vs-mh data-side multimodality probe | offline | block 4, the weakest |
| 2 | GR1 homoscedastic Student-t @60k | 1 run | block 6 |
| 3 | GR1 HG @60k, endpoint-matched | 1 run | block 6 |
| 4 | MSE on WidowX + fractal @20k | 2 runs | block 1 |
| 5 | re-run the 4 collapsed robomimic cells | 4 short runs | block 2 |
| 6 | Hang mug real-robot, flow / HT / MSE | 3 real-robot runs + eval | block 3 |

## MSE control cells (2026-09-05)

Four missing MSE cells of the VLA table are in flight, all on the incumbent recipes with only the loss swapped:

| cell | run | recipe | state |
|---|---|---|---|
| Google Robot (fractal) MSE | `groot/ft_fr_mse` (`run_fr_mse.sh`) | 20k, dropout 0.5, batch 1024, `--loss-type=mse` | DONE 2026-09-06: **62.8** (flow 67.3, HT c=2 63.5) |
| WidowX MSE | `groot/ft_wxmse` (`run_wx_mse.sh`) | 20k, state dropout 0.8, batch 1024 | DONE 2026-09-06: **63.1** (flow 57.1, HT 62.9 / c=2 67.4) |
| GR1 random-init MSE | `groot/ft_msesc` (`run_gr1_msesc.sh`) | 60k, batch 512, `--reinit-action-head` | DONE 2026-09-06: **5.8** (24x20 fanout `msesc` @60k; flow-scratch 14.0, HT-scratch 33.5; six Close tasks 0/20) |
| Cosmos3 LIBERO-10 MSE | `cosmos3 libero10_mse_a800` (`run_c3_mse.sh`, `HT_MSE=1`) | nu512 recipe, 2000 iters (~25 h) | DONE 2026-09-07: **97** (97/100, 10x10 @2000; flow 96, HT nu=320 93, nu=512 / c=2 96) |
| fractal HT + Muon (`ft_fr_muon_ht`, GROOT_MUON=1, plain DDP) | HT c=2 recipe, 20k | launched 2026-09-07 10:39 on .20, stopped by the user at step 910/20000 before any checkpoint; no result. flow+Muon control deleted at 10 min |
| fractal MSE-then-HT (`ft_fr_ms10ht` / `ft_fr_ms15ht`, `--ht-mse-steps 10000/15000`) | HT c=2 recipe, 20k | ms10ht DONE 2026-09-06: **61.8** (vs flow 67.3 p=0.012, HT 63.5 tie, MSE 62.8); ms15ht DONE 2026-09-07: **60.0** (vs flow p=0.001). Warm-up does not help; round closed |
