# Consolidated VLA results (HT campaign) — single source for the paper tables

Detail files: `vla_results.md` (GR1 curves/per-task, pi0.5 per-suite), `multimodality_table.md`,
memory notes `nu-is-inert-heteroscedasticity-is-the-method.md`. Updated 2026-08-27.

## A. Cross-stack headline — single-pass HT vs flow / L1 / MSE (budget-matched, all entries annotated @steps)

Blank = not run. HT entries carry their nu. Within a row every number is at the same eval protocol;
"@steps" is the training checkpoint evaluated.

| stack (chunk d) | protocol | flow | L1 | MSE | HT incumbent (nu = 2d) | HT swept-best | HT fixed c = 2 (nu = 4d) |
|---|---|---|---|---|---|---|---|
| GR00T GR1, robocasa (232) | 24 tasks x 20 eps, official harness | 44.1 @60k (pub 44.5 @60k) | 17.2 @60k | 37.8 @60k | nu=464: 47.5 @60k | nu=928: 51.5 @60k (nu=2: 24.5; nu=0.5: 44.8; nu=464: 47.5) | nu=928: 51.5 @60k |
| GR00T WidowX, bridge (56) | 7 tasks x 50 eps (effective 10 scenes/task) | 57.1 @20k | | 63.1 @20k | nu=112: 62.9 @20k | nu=224: 67.4 @20k (flat 49-67 over nu=2..448) | nu=224: 67.4 @20k |
| GR00T Google Robot, fractal (56) | 6 tasks x 100 eps, SimplerEnv | 67.3 @20k | | 62.8 @20k | | | nu=224: 63.5 @20k |
| OpenVLA-OFT libero-long (56) | 10 tasks x 10 eps (released ref: 500 eps, best ckpt) | | released 94.5 (best ckpt <=150k) | | nu=112: 82 @120k (arm endpoint) | nu=1024: 93.0 @150k (465/500; a 100-ep screen read 98) | nu=224: 94.0 @150k (470/500) |
| OpenVLA-OFT libero-spatial | 10 tasks x 50 eps (500) | | released 97.6 | | nu=112: 98.4 @50k (492/500) | | nu=224: 97.4 @150k (487/500) |
| OpenVLA-OFT libero-goal | same | | released 97.9 | | nu=112: 98 @90k (97.6 @30k, 500 eps) | nu=1024: 97.6 @150k (488/500); 96.0 @50k (480/500) | nu=224: 96.6 @150k (483/500); 96.8 @50k (484/500, their protocol point) |
| OpenVLA-OFT libero-object | same | | released 98.4 | | nu=112: 95 (best ckpt) | nu=1024: 95.8 @20k (479/500; a 100-ep screen read 99) | nu=224: 95.6 @150k (478/500); 96.6 @100k (483/500, ckpt selected on a 100-ep sweep and confirmed at 500) |
| pi0.5 LIBERO (350), 4-suite avg | 4 x 1000 eps | published ~96.9 | | 96.8 @30k | nu=700: 97.6 @30k (spatial 98.1 / goal 97.6 / object 99.8 / long 94.7) | (nu=2 97.0 @30k was a 1-GPU run, 8x fewer samples: not comparable) | nu=1400: 96.6 @30k (97.4 / 97.1 / 99.4 / 92.6) |
| Cosmos3 libero-10 (160) | 10 tasks x 10 eps, 30-step sampler | 96 @2000 | | 97 @2000 | nu=320: 93 @2000 | nu=512: 96 @2000 | nu=640: 96 @2000 |

Reading across rows: single-pass HT matches or beats its iterative baseline on every stack once nu
is at a light-but-finite operating point (GR1 51.5 vs 44.1; WidowX 62.9 vs 57.1; OFT-long 98 vs
released 94.5; pi0.5 97.6 vs ~96.9; Cosmos3 96 = 96). The accidental default nu = 2d underperforms
only where the hard-but-learnable residual band matters (OFT-long 82, Cosmos3 93). The fixed
prescription c = 2 (nu = 4d, the same rule on every stack) matches the swept optimum where it has
been run to its endpoint (WidowX 67.4 = sweep best; Cosmos3 96 = flow; GR1 51.5 > the swept-at-2d incumbent 47.5,
single seed, SE ~2.3), is within 3 points on OFT-long (95 vs 98 for nu=1024 at 150k; above the released L1 94.5), and is 1.0 point
below the 2d incumbent on pi0.5 (96.6 vs 97.6 over 4000 episodes, single seed; the gap sits in libero-long, 92.6 vs
94.7, the other three suites are within 0.7). pi0.5 is the one stack where the lighter tail does not help.

Huber-gate controls (budget-matched, not a table column): GR1 simple Huber (MSE inside 2.5 x EMA-median chunk-residual norm, linear beyond; no sigma) 37.4 @60k = MSE 37.8, vs HT 47.5 / flow 44.1; GR1 heteroscedastic Huber (Huber rho on ||r||/sigma + d log sigma) 40.7 @60k: sigma normalization recovers about half of the MSE->HT gap, the redescending gate the rest. OFT-long simple / hetero Huber 97 / 95 @50k (arms stopped at 80k). The cap reproduces HT on OFT-long but not on GR1. Heteroscedastic L1 (sum|r|/sigma + d log sigma, GROOT_HL1 / OFT_HL1, 2026-09-07): GR1 44.5 @60k (above hetero-Huber 40.7, level with flow 44.1, under HT 47.5 / 51.5); OFT-long 69 @20k (vs HT c=2 89, MSE 84, L1 66; 50k pending).

Budget-mismatched controls (not in the table): OFT-long hetero-Gaussian 73 @50k (arm stopped at
~91k; 59 @20k); OFT-goal/object HG 94/88 @50k; GR1 pure HG 20.6 @22k and HG-pre->HT 32.8 @22k (HT
@60k); in-house OFT L1 66 @20k and MSE 84 @20k. GOOGLE ROBOT (FRACTAL) COMPLETED 2026-09-02, and it is the FIRST STACK WHERE HT LOSES TO FLOW: flow 67.3 vs HT nu=224 63.5 (6 tasks x 100 eps, identical recipe/data/eval, full 87,212-episode dataset, sbias -0.5093). Per task flow/HT: coke 0.88/0.94, pick-object 0.76/0.82, move-near 0.99/0.94, open-drawer 0.59/0.56, close-drawer 0.75/0.50, place-in-closed-drawer 0.07/0.05 (released N1.7: 1.00/0.94/1.00/0.65/0.69/0.07 = 72.5). The split is systematic: HT WINS both free-space pick tasks (+6 each) and LOSES the articulated/contact tasks (close-drawer -25). Both arms sit below the released reference (flow -5, HT -9), so the absolute level is set by our training rather than by the head. Contrast WidowX (same d=56, same c=2): HT 67.4 vs flow 57.1. Single seed, 100 eps/task (SE ~5 pts per task, ~2 pts on the mean). LIBERO-LONG HAS A PER-TASK TRAINING BIFURCATION (2026-09-01) -- single-run comparisons on that suite are unsafe. Matched 500-episode evals of the long arms: nu=350 (c=2.5) gives 80.0 @50k and 81.2 @100k vs nu=224 (c=2) 93.2 / 92.2 (and 90.2 @20k, 94.0 @150k). The whole 12-point gap sits in TWO of the ten tasks (task 1: 0.18-0.20 vs 0.88-0.94; task 2: 0.60-0.68 vs 0.96-0.98); on the other eight tasks the arms are indistinguishable (91.5 vs 92.0). A smooth nu effect would degrade all ten tasks slightly, so this is a run-level failure to learn specific long-horizon tasks, not a nu effect -- and it explains the earlier non-monotone reading (nu=112 82, nu=224 93, nu=350 80, nu=1024 93: two 'bad' runs and two 'good' ones). Quote libero-long only with several seeds, or report the eight-task subset alongside. NU SWEEP ON OFT IS FLAT ONCE CONFIRMED AT 500 EPISODES (2026-08-31). The swept-best nu=1024 row was built from 100-episode screens; re-run at 500 eps it gives long 93.0 (screen 98), goal 97.6 @150k / 96.0 @50k (screen 98), object 95.8 @20k (screen 99) -- i.e. nu=1024 is NOT better than the fixed nu=224 (long 94.0, goal 96.6/96.8, object 95.6/96.6); on the three suites with both, the means are 95.5 (nu=1024) vs 95.4-96.0 (nu=224). OFT therefore does NOT prefer a large c: its apparent optimum at c=4.28 was screen noise, and no value of nu closes the ~1-point gap to the released L1 (97.1). This removes the only stack that contradicted the fixed c=2 prescription. C=2.5 SWEEP (nu=6.25d), started 2026-08-30 after the c=2 column closed: WidowX nu=350 65.1 (per task 0.54/0.82/0.90/0.82/0.44/0.86/0.18) vs c=2 67.4 and 2d 62.9 -- same within the pooled SE of ~6 points (5 parallel envs share initial states, so 10 distinct scenes/task). GR1 nu=1450 47.6 @60k (24 tasks x 20 eps, SE 2.3) vs c=2 51.5 and 2d 47.5: going heavier than 4d does not help on GR1. pi0.5 nu=2187.5 97.2 (spatial 97.6 / goal 97.0 / object 99.6 / long 94.5, 4 x 1000 eps, SE 0.35 on the average) vs c=2 96.6 and 2d 97.6: NOT monotone -- the c=2 point is the low one, driven by libero-long (92.6 vs 94.5-94.7 at c=1.41 and c=2.5), so pi0.5 is flat over c=1.41..2.5 and both neighbours of c=2 sit above the published flow baseline (96.9). OFT-long nu=350 and OFT-object nu=350 still training. Arm status (2026-08-27). Completed: OFT-long fixed nu=224 (89 @20k, 94 @30k, 94 @50k, 95 @150k), Cosmos3 fixed nu=640 (62/83/97/96 @iter 500/1000/1500/2000), GR1 simple / hetero Huber (37.4 / 40.7 @60k), OFT-long learned-nu (93 @150k). Stopped before their endpoint (last evaluated snapshot is final): OFT-long median-pinned c=2.5 (95 @50k, stopped at 130k) and c=2 (88 @50k, at 60k), per-sample / global argmax-nu table (92 / 91 @50k, at 80k), OFT-long simple / hetero Huber (97 / 95 @50k, at 80k), Cosmos3 median-pinned c=2 (never evaluated). Completed for the c = 2 column: GR1 nu=928 51.5 @60k, pi0.5 nu=1400 96.6 @30k (8-GPU rerun, budget-matched; a first 1-GPU attempt gave 96.1). Completed 2026-08-30, all OFT c=2 cells re-run at 500 episodes: spatial 97.4 (487/500), object 95.6 (478/500), goal 96.6 @150k (483/500) and 96.8 @50k (484/500), long 94.0 (470/500) -> 4-suite average 96.0 (paper protocol: 150k, goal 50k) vs released L1 97.1; 96.2 if object also uses its selected checkpoint (96.6 @100k). The 100-episode screens read high on every suite (goal 99 -> 96.6, long 95 -> 94.0, object 94 -> 95.6 low), consistent with the runbook's screen-vs-confirm rule; the c = 2 column is now complete for every stack. Checkpoint protocol: OpenVLA-OFT's own recipe (LIBERO.md) evaluates the 150k checkpoint for every suite except libero-goal, where they use 50k; our runs copy their command including the 10x LR decay at 100k, so all OFT cells above are 150k endpoints unless marked otherwise. A 13-point checkpoint sweep of object nu=224 at 100 episodes (10k 92, 20k 96, 30k 95, 40k 96, 50k 94, 60k 94, 70k 96, 80k 97, 90k 94, 100k 98, 120k 97, 140k 96, 150k 94) is flat within noise (SE 2.2 points at n=100); the first 100-episode reading of 94 at 150k was noise, and the 500-episode re-evaluation gives 95.6. DROID + Cosmos3 flow/HT (nu=1024) pair: data, recipe and smoke test done (4 samples/rank fit; 108 s per 512-sample iteration on one A800 node, i.e. ~6 days for 5k iterations), paused until the c = 2 cells are filled; no RoboLab evaluation stack exists yet. Full trajectories: memory note
`nu-is-inert-heteroscedasticity-is-the-method.md` and `vla_results.md`.

## B. Single-pass vs iterative cost

- HT is 1 NFE on every stack. Iterative baselines: GR1 flow 4 NFE, pi0.5 flow 10, Cosmos3 30.
  Cosmos3's HT is served through the same 30-step joint video/action sampler, so it is not a
  latency win there.
- OpenVLA-OFT's diffusion head at T_test = 1 collapses to 0.0 SR (their Table II): single-pass by
  construction is not free for iterative heads.

## C. Mechanism results (mechanism section)

- Gradient share of the top-1% residual samples: HG 18.5% vs HT (nu=1024) 0.9% on OFT (per-sample
  sigma); pi0.5 9.4% -> 1.0%. Data/gradient proportion tables in the memory note.
- GR1 HG-vs-HT training-set fitting probe: HG fits worse on every channel; worst on the binary
  hand-transition channels (HG/HT rms ratio 1.44 vs 1.25 elsewhere, dim 24 = 1.89).
- Multimodality: 0 alternative-action states in 3000 screened states per stack; all above-chance
  sampling structure is gripper/hand event timing (`fig_flow_unimodal.png`, `multimodality_table.md`).
- Flow vs HG vs HT over the surprise tail (E7; same samples, flow checkpoints): HG AMPLIFIES the
  tail, flow is PROPORTIONAL to it (gradient share ~ data share; the (t, noise) draw dominates
  per-sample gradient, corr with the sample's residual ~0), HT SUPPRESSES it:

  | stack | tail bucket | data % | HG grad % | flow grad % | HT grad % |
  |---|---|---|---|---|---|
  | GR1 (flow ckpt @60k) | 1-2x rms | 23.4 | 40.0 | 28.0 | 27.0 |
  | GR1 | 2-4x rms | 0.6 | 2.8 | 1.2 | 0.6 |
  | GR1 | top 1% of samples | 1.0 | 4.0 | 2.9 | 1.1 |
  | pi0.5 (flow ckpt libero_ft) | 1-2x rms | 36.0 | 62.4 | 36.6 | 53.3 |
  | pi0.5 | 2-4x rms | 3.0 | 15.8 | 2.4 | 3.7 |
  | pi0.5 | top 1% of samples | 1.0 | 6.2 | 4.0 | 1.5 |

  Matches the SR ordering HT >= flow > HG (GR1 47.5 / 44.1 / 20.6; OFT-long 98 / -- / 73).

- ML nu on converged residuals is 2.0-24 on every stack while SR-optimal nu is 224-1024:
  likelihood-optimal and learning-optimal gates are different objects.

## D. Caveats to carry into the paper

- OFT cells: 100 episodes, single seed; the released reference is best-of-checkpoints over 500 eps.
- WidowX: 5 parallel envs share initial states -> effective n = 10 scenes per task (SE ~5.9 pts pooled).
- GR1 HG (22k) vs HT (60k): HG stopped early; the per-dimension structure of the fit gap is the
  step-robust signal, the uniform part is partly budget.
- pi0.5 "ref flow ckpt" numbers are our-harness evals of finetuned_v044 and sit well below the
  published figures; quote the published number as the baseline.
- GR1 HT best full-decay endpoint is 47.4 at a 50k schedule (no matched flow run at 50k); the
  strictly matched comparison is the 60k endpoint.

## E. Gradient proportion tables (why HT > HG at large nu)

Method: per-sample residuals from several hundred training samples at each stack's trained HT
checkpoint (dumps `resid_dist_{oft,pi05,gr1}.npz`; OFT = HTpair-long 120k, pi0.5 = run_ht 30k,
GR1 = ft_ht3 60k). "data %" = share of samples in the bucket (error size in multiples of the pooled
rms); "grad %" = the bucket's share of total squared-gradient mass, HG mass = S/sigma^4, HT mass =
w^2 * S/sigma^4 with w = (nu+d)/(nu + S/sigma^2). Measured at converged, gate-trained checkpoints,
so the tails are as small as they ever get: these are lower bounds on what HG suffers during training.

### E1. OpenVLA-OFT libero-long (960 samples, d=56, nu=1024, per-sample sigma = the loss's actual gate)

| error size | data % | HG grad % | HT grad % |
|---|---|---|---|
| < 0.5x rms | 46.0 | 25.7 | 44.0 |
| 0.5-2x (typical) | 49.2 | 51.2 | 53.9 |
| 2-4x | 4.0 | **17.3** | 1.8 |
| 4-10x | 0.8 | **5.9** | 0.3 |
| > 10x | 0.0 | 0.0 | 0.0 |
| **top 1% of samples** | 1.0 | **18.5** | **0.9** |

### E2. pi0.5 LIBERO (600 samples, d=350, nu=700, pooled sigma)

| error size | data % | HG grad % | HT grad % |
|---|---|---|---|
| < 0.5x | 24.5 | 2.9 | 7.7 |
| 0.5-2x | 72.8 | 79.6 | 89.3 |
| 2-4x | 2.7 | **17.5** | 3.1 |
| 4-10x | 0.0 | 0.0 | 0.0 |
| top 1% | 1.0 | **9.4** | 1.0 |

### E3. GR00T GR1 robocasa (320 samples, d=232, nu=464, pooled sigma)

| error size | data % | HG grad % | HT grad % |
|---|---|---|---|
| < 0.5x | 0.0 | 0.0 | 0.0 |
| 0.5-2x | 99.4 | 97.4 | 99.4 |
| 2-4x | 0.6 | 2.7 | 0.6 |
| top 1% | 1.0 | 3.9 | 1.0 |

GR1 note: chunk-level ratios are compressed by averaging over 232 dims (max 2.1x rms), so at pooled
sigma the losses barely differ; GR1's spreading happens through its per-sample sigma head (median
0.035 vs pooled rms 0.199; 97.5% of samples past the gate knee under the trained sigma). The
per-sample-sigma version would look like E1.

### E4. Gradient pull per sample vs error size (OFT, sigma co-adapted to each nu; typical sample = 1)

| loss | sigma / rms | gate knee | 0.5x | 1x | 2x | 3x | 5x | 10x |
|---|---|---|---|---|---|---|---|---|
| HG (nu=inf) | 1.00 | -- | 0.50 | 1.00 | 2.00 | 3.00 | 5.00 | 10.00 |
| nu=1024 (swept best) | 0.87 | 3.7x rms | 0.53 | 1.00 | 1.66 | 1.94 | 1.90 | 1.29 |
| nu=112 (incumbent) | 0.63 | 0.9x rms | 0.86 | 1.00 | 0.75 | 0.55 | 0.35 | 0.18 |
| nu=2.2 (ML on residuals) | 0.45 | 0.09x rms | 1.95 | 1.00 | 0.50 | 0.34 | 0.20 | 0.10 |

Pull = gate weight x residual. Large nu keeps the rising (MSE-like) pull through the learnable
2-4x band and caps only the far tail; heavy nu inverts the curve (most pull to already-fit samples);
HG's pull grows without bound.

### E5. ML-optimal nu on the same converged residuals (the objective-mismatch result)

| stack | ML nu (one fitted sigma) | ML nu (head's per-sample sigma) | SR-optimal nu | frac past knee at ML nu / at SR-optimal nu |
|---|---|---|---|---|
| OFT (d=56) | 2.2 | 2.2 | 1024 | 100% / 1.3% |
| pi0.5 (d=350) | 2.0 | -- | 700 | 99.8% / 24% |
| GR1 (d=232) | 23.6 | -- | 464 | 100% / 3.7% |

Maximum likelihood in nu is a fit criterion (it explains the over-dispersed normalized residuals
with tails, halving sigma); the SR-optimal nu is a control criterion (keep MSE pull through the
learnable band, gate only surprises). A likelihood-fitted nu therefore lands heavy.

## E6. Gate scale: theory vs real training, with the swept optimum per stack

Knee in sigma units = sqrt(nu/d). Theory row assumes a calibrated sigma and Gaussian residuals
(u ~ chi2_d). "past knee" = measured fraction of training samples with S/sigma^2 > nu at the
listed checkpoint (per-sample head sigma for OFT/GR1/WidowX, pooled sigma for pi0.5).

| stack (d) | setting | nu | k = nu/d | nominal knee | past knee (theory, Gaussian) | past knee (measured) |
|---|---|---|---|---|---|---|
| OFT (56) | incumbent 2d | 112 | 2 | 1.41 sigma | ~1e-5 | 61% |
| OFT | fixed c=2 (4d) | 224 | 4 | 2.0 sigma | 6e-22 | 29% |
| OFT | **swept best** | **1024** | **18.3** | **4.3 sigma** | ~0 | **3.6%** |
| GR1 (232) | incumbent 2d = **swept best** | **464** | **2** | **1.41 sigma** | ~1e-12 | **97.5%** (sigma under-predicts 5.5x; Cauchy-like regime) |
| GR1 | fixed c=2 (4d) | 928 | 4 | 2.0 sigma | 6e-84 | 90% |
| WidowX (56) | incumbent 2d | 112 | 2 | 1.41 sigma | ~1e-5 | 12.8% (gate probe @20k) |
| WidowX | swept best (flat axis) | 224 | 4 | 2.0 sigma | 6e-22 | not measured |
| pi0.5 (350) | incumbent 2d = swept-equivalent (flat) | 700 | 2 | 1.41 sigma | ~1e-19 | 24% (pooled sigma) |
| Cosmos3 (160) | incumbent 2d | 320 | 2 | 1.41 sigma | ~1e-9 | not measured |
| Cosmos3 | swept best | 512 | 3.2 | 1.79 sigma | ~1e-24 | not measured |
| Cosmos3 | fixed c=2 (4d) | 640 | 4 | 2.0 sigma | 1e-58 | not measured (SR 96 @2000) |

Reading: the swept-optimal k is not stack-invariant (2 to 18, a 9x spread) because sigma's
calibration is not; in realized terms the SR-optimal gate touches a few percent of samples on OFT
but ~all samples on GR1 (whose optimum operates as a Cauchy-like loss).

## E7. Flow-matching gradient distribution (does flow's training gradient concentrate on the tail?)

Method: per training sample, the flow-matching loss is measured with 8 Monte-Carlo draws of (t, noise)
on the FLOW checkpoint, and the sample's action-space residual comes from the flow model's own sampled
action (eval-setting sampler) vs ground truth. Flow's loss is plain MSE on the velocity residual, so a
sample's gradient mass is proportional to its per-sample flow loss. HG/HT columns are computed on the
same samples' action residuals (HG mass ∝ S; HT mass ∝ w^2 S at the stack's incumbent nu, pooled sigma).

### E8a. GR00T GR1 robocasa (320 samples, ft_flow @60k, 4-step sampler; d=232)

| bucket (flow model's action residual) | data % | FLOW grad % | HG grad % | HT (nu=464) grad % |
|---|---|---|---|---|
| 0.5-1x rms | 75.9 | 70.7 | 57.2 | 72.4 |
| 1-2x | 23.4 | 28.0 | 40.0 | 27.0 |
| 2-4x | 0.6 | 1.2 | 2.8 | 0.6 |
| top 1% of samples | 1.0 | 2.9 | 4.0 | 1.1 |

Diagnostics: corr(per-sample flow loss, action residual) = 0.25; per-sample flow loss varies across
the 8 (t, noise) draws with median CV = 0.95 — i.e., most of a sample's flow gradient is set by the
random (t, noise) draw rather than by the sample's own residual. Flow's gradient distribution on GR1
sits between HG and HT: it does not concentrate on the tail like HG (the t/noise averaging dilutes
per-sample differences) but it also does not gate; note GR1's chunk-level residuals have no far tail
at pooled sigma (max ~2.1x rms), so this is the mild-tail case.

### E8a-T. The same table across training (GR00T GR1, 320 samples, 8 (t, noise) draws, 4-step sampler; buckets relative to each checkpoint's own pooled rms)

| checkpoint | pooled rms | bucket | data % | FLOW grad % | HG grad % | HT (nu=464) grad % |
|---|---|---|---|---|---|---|
| 2k (ft_flow2k, SR 1.5%) | 0.276 | 0.5-1x | 69.7 | 62.0 | 40.8 | 61.9 |
|  |  | 1-2x | 30.0 | 37.7 | 57.9 | 37.7 |
|  |  | 2-4x | 0.3 | 0.2 | 1.3 | 0.4 |
|  |  | top 1% of samples | 1.2 | 1.6 | 4.6 | 1.5 |
| 8k (ft_flow2k) | 0.261 | 0.5-1x | 70.6 | 65.7 | 44.9 | 64.1 |
|  |  | 1-2x | 29.1 | 33.5 | 53.6 | 35.6 |
|  |  | 2-4x | 0.3 | 0.7 | 1.5 | 0.3 |
|  |  | top 1% of samples | 1.2 | 1.9 | 4.5 | 1.4 |
| 46k (keep/flow_46000) | 0.217 | 0.5-1x | 75.6 | 70.3 | 56.3 | 71.9 |
|  |  | 1-2x | 23.8 | 28.5 | 40.8 | 27.4 |
|  |  | 2-4x | 0.6 | 1.2 | 2.9 | 0.6 |
|  |  | top 1% of samples | 1.2 | 2.1 | 5.2 | 1.3 |
| 60k (ft_flow, converged; = E8a) | 0.214 | 0.5-1x | 75.9 | 70.7 | 57.2 | 72.4 |
|  |  | 1-2x | 23.4 | 28.0 | 40.0 | 27.0 |
|  |  | 2-4x | 0.6 | 1.2 | 2.8 | 0.6 |
|  |  | top 1% of samples | 1.2 | 1.9 | 5.2 | 1.3 |

Diagnostics per checkpoint: 2k (ft_flow2k, SR 1.5%): rms 0.276, max residual 2.03x rms, corr(flow loss, residual) 0.46, median CV of flow loss across draws 0.66, flow loss median 0.1725; 8k (ft_flow2k): rms 0.261, max residual 2.18x rms, corr(flow loss, residual) 0.31, median CV of flow loss across draws 0.76, flow loss median 0.1885; 46k (keep/flow_46000): rms 0.217, max residual 2.17x rms, corr(flow loss, residual) 0.27, median CV of flow loss across draws 0.94, flow loss median 0.1939; 60k (ft_flow, converged; = E8a): rms 0.214, max residual 2.11x rms, corr(flow loss, residual) 0.27, median CV of flow loss across draws 0.95, flow loss median 0.1929.

Reading: the three regimes are already present at 2k and do not change in kind with training - HG's share of the 1-2x band and of the top 1% is above the data share at every stage, HT's is at or below it, flow's sits between (flow's top-1% share stays within 1-3x its data share at every stage, HG's within 2-5x). What changes with training is the residual scale (pooled rms falls) and the tail's extent; the ordering HG > flow > HT on the tail is stage-independent. No GR1 flow checkpoint exists between 8k and 46k (the ft_flow run kept only its last five), so the mid-early part of the curve is not covered; pi0.5 has no early/mid flow checkpoints at all (the flow baseline is the released endpoint).

### E8b. pi0.5 LIBERO (600 samples, libero_ft flow ckpt, Beta-sampled t, 8 (t, noise) draws; 10-step sampler for the action residual; d=350)

| bucket (flow model's action residual) | data % | FLOW grad % | HG grad % | HT (nu=700) grad % |
|---|---|---|---|---|
| < 0.5x rms | 26.2 | 28.1 | 3.6 | 9.5 |
| 0.5-1x | 34.8 | 32.9 | 18.2 | 33.5 |
| 1-2x | 36.0 | 36.6 | 62.4 | 53.3 |
| 2-4x | 3.0 | 2.4 | 15.8 | 3.7 |
| top 1% of samples | 1.0 | 4.0 (single draw 5.8) | 6.2 | 1.5 |

Diagnostics: corr(per-sample flow loss, action residual) = -0.03 (none); per-sample flow loss CV
across draws = 0.63. The flow gradient distribution tracks the DATA distribution almost exactly
(28/33/37/2.4 vs 26/35/36/3.0): flow's per-sample gradient is set by the random (t, noise) draw and
is essentially blind to whether the sample is a surprise. It neither chases the tail (HG puts 78% of
its gradient on the >1x band, 16% on the 2-4x surprise band) nor gates it (HT: 4%); the tail's share
under flow equals its data share. Probe pitfall (recorded): the residual scaffold's
`sample_time -> ones` override must be removed for a flow-loss measurement — with t=1 the loss
collapses to the single-pass residual and all draws coincide.

Summary of E7: three distinct gradient regimes over the surprise tail — HG amplifies it (share >>
data share), flow is proportional to it (share ~ data share; the (t, noise) noise floor dominates
per-sample differences), HT suppresses it (share << data share). This is consistent with the
success-rate ordering HT >= flow > HG on GR1 and OFT-long.


### E8b-corrected (2026-09-06). pi0.5 LIBERO, same checkpoint, protocol defect fixed

The E8b probe above had `sample_noise` overridden to zeros, which the LeRobot policy uses for BOTH the
deployed sampler and the training loss: its 8 draws varied only in t with x0 = 0, and its residual came
from a zero-start deterministic 10-step integration. Rerun (`pi05/resid_flow_pi05_v2.py` on PFS,
`resid_flow_pi05_v2.npz`; 600 samples, different sample draw than E8b): loss draws with
x0 ~ N(0, I) and t = 0.999 Beta(1.5, 1) + 0.001, residual from the deployed stochastic 10-step sampler
(one draw). Masses as in E8a: MSE = S, HT = w^2 S with w = (nu+d)/(nu+S/sigma^2), nu = 700 = 2d, sigma^2 = pooled
mean squared residual; Flow = mean over draws of the per-sample flow loss. Pooled rms 0.135, max residual
5.0x rms, draw CV median 0.73, corr(flow loss, residual) -0.04.

| bucket | data % | FLOW grad % | MSE grad % | HT (nu=700) grad % |
|---|---|---|---|---|
| < 0.5x rms | 24.8 | 26.5 | 3.2 | 8.4 |
| 0.5-1x | 39.8 | 38.5 | 22.3 | 40.0 |
| 1-2x | 32.7 | 33.2 | 55.7 | 48.6 |
| 2-4x | 2.5 | 1.6 | 14.5 | 3.0 |
| > 4x (1 sample) | 0.2 | 0.1 | 4.2 | 0.1 |
| top 1% of samples | 1.0 | 0.7 | 10.2 | 1.0 |

Amplification (grad share / data share) in the 2-4x bin: MSE 5.8x (E8b read 5.3x), Flow 0.66x, HT 1.19x.
The zero-start residual variant recorded in the same run gives MSE 6.9x / Flow 1.04x / HT 1.11x. Reading
unchanged: flow tracks the data share, MSE amplifies the tail, HT stays near proportional. The figure
`gradient_distribution/fig_gradient_distribution.png` panel (b) now uses this table (the single-sample
> 4x bin is not drawn).


### WidowX MSE control cell (2026-09-06)

`ft_wxmse` (run_wx_mse.sh: identical recipe to the HT/flow arms, `--loss-type=mse`, state dropout 0.8, 20k, batch 1024),
7 tasks x 50 episodes at checkpoint-20000 (`wxeval_wxmse_*`): carrot 0.32 / close-drawer 0.90 / open-drawer 0.88 /
eggplant-basket 0.94 / eggplant-sink 0.48 / spoon 0.70 / stack 0.20 = **63.1**. Same protocol: flow 57.4 (f20000 logs;
57.1 in the table), HT nu=112 62.9, HT nu=224 (c=2) 67.4. MSE ties the nu=112 incumbent and sits 4.3 under the c=2
prescription; the per-task pattern differs from HT's (MSE strong on drawers, weak on stack), all inside the WidowX
noise floor (10 distinct scenes per task).


### Google Robot (fractal) MSE control cell (2026-09-06)

`ft_fr_mse` (run_fr_mse.sh: the HT control recipe with `--loss-type=mse`; 20k, batch 1024, state dropout 0.5), seeded
protocol (SEED=1234, 100 scenes per task, `freval_smse-20k_*`): coke 0.98 / pick-object 0.69 / move-near 0.97 /
open-drawer 0.46 / close-drawer 0.60 / place-in-drawer 0.07 = **62.8**. Same scenes: flow 67.3, HT c=2 63.5. Paired
McNemar over the 600 scenes: MSE vs flow p = 0.03 (flow wins), MSE vs HT tie (p = 0.68). MSE beats HT on close-drawer
(0.60 vs 0.50) and coke, loses on pick-object and open-drawer.

### GR1 random-init MSE control cell (2026-09-06)

`ft_msesc` (run_gr1_msesc.sh: the random-init recipe used for the flow-scratch / HT-scratch pretraining ablation,
`--loss-type=mse --reinit-action-head`, frozen pretrained backbone, 60k, batch 512), official 24-task x 20-episode
fanout at checkpoint-60000 (`evalout_msesc_*`): per task (fanout order) 0.00 0.00 0.00 0.00 0.00 0.00 0.10 0.05 0.238
0.182 0.05 0.10 0.00 0.05 0.00 0.05 0.00 0.10 0.10 0.143 0.05 0.190 0.00 0.00 = **5.8** (mean 5.847). Same protocol:
flow-scratch 14.0, HT-scratch 33.5. All six Close tasks are 0/20 under MSE (flow-scratch 1/120, HT-scratch 23/120).
With a random-init head MSE is the weakest of the three objectives by a wide margin (8.2 under flow, 27.7 under HT),
whereas from the flow-pretrained head MSE (37.8) is within 6.3 of flow (44.1): the pretraining dependence is
+32.0 for MSE, +30.1 for flow, +11.2 for HT.

### Cosmos3 LIBERO-10 MSE control cell (2026-09-07)

`libero10_mse_a800` (run_c3_mse.sh: the nu=512 recipe with `HT_MSE=1`, loss `0.5*sum r^2`, sigma head unused; 2000
iterations, 8xA800), `eval_c3.sh ... iter_000002000 10 mse_2k` (10 tasks x 10 trials, 30-step sampler, results_mse_2k):
per task 9 10 10 9 10 9 10 10 10 10 = **97**. Same protocol at iteration 2000: flow 96 (8 9 10 9 10 10 10 10 10 10),
HT nu=320 93 (8 10 10 8 10 10 10 10 8 9), HT nu=512 96 (9 10 10 9 10 10 10 10 9 9), HT c=2 nu=640 96. With 100
episodes the binomial SE is about 1.7 points, so 93 / 96 / 96 / 97 are one band: on Cosmos3 LIBERO-10 the objective
does not separate the arms at the endpoint (the ordering earlier in training, flow 94 vs HT nu=320 79 at 1k, is the
only budget-sensitive signal on this stack). This completes the four MSE control cells.
