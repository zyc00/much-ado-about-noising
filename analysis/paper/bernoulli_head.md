# Factorized HT likelihood: Bernoulli on binary channels, Student-t on the rest

Working note, started 2026-09-05. Results are appended as they land.

## Motivation
HT fits one Student-t scale per sample over all action columns. Gripper/hand columns are binary
commands (0/1 or open/close), so the Gaussian-family residual on them is wrong twice: a mistimed
switch costs squared residual ~4 per column, which dominates the per-sample statistic that sets
sigma and the gate (GR1: switch states are 8-12% of states but carry 28-38% of the squared
residual; the gate weights them 0.31-0.34, suppressing the arm channels at exactly the grasp and
release states), and the regression mean is a hedged value that the environment executes as a
half-closed hand/gripper, which never occurs in the demonstrations.

## Data facts
| stack | binary columns | values | normalized | share closed |
|---|---|---|---|---|
| Fractal (GR00T, d=56) | 1 of 7 (col 6) | 0 / 1 | -1 / +1 (min-max) | 0.535 open |
| GR1 (GR00T, d=232) | 12 of 29 (cols 14-25) | per hand: 4 cols ±1.5, 1 col ±3.0 (same sign, 0 mismatches in 10,896 frames of all 24 datasets), thumb const 3.0 | ±1 after switching hands to ABSOLUTE | left 0.18, right 0.37 |
| pi0.5 LIBERO (d=350) | 1 of 7 (col 6) | -1 / +1 exactly | -0.95 / +1.05 (mean-std) | 0.43 |

GR1 caveat: NVIDIA's embodiment config stores hand actions RELATIVE (target minus current finger
state), which is continuous in model space. The Bernoulli arm switches the two hand keys to
ABSOLUTE (`GROOT_GR1_ABS_HANDS=1`, needed at training and at eval-server time). The control keeps
relative hands, so the GR1 comparison includes that representation change.

## Design (identical across stacks)
- Binary columns leave the Student-t residual, sigma, d_eff and the gate.
- One logit per binary group per chunk step (GR1: mean of the group's decoder columns), BCE vs
  (a_norm > 0), weight 1.0, summed over chunk steps, added to the per-sample Student-t NLL.
- nu keeps c=2 on the reduced continuous dimension: Fractal 224 (d=48), GR1 544 (d=136), pi0.5 1200 (d=300).
- Inference emits sign(logit) = ±1 normalized (a full open/close command every step).

## Arms and controls (single seed)
| stack | arm | recipe | control |
|---|---|---|---|
| Fractal | ft_fr_bern_20k | 20k, batch 1024, dropout 0.5 | HT nu=224: 63.5 (seeded, 6 x 100) |
| GR1 | ft_gr1bern | 60k, batch 512, sbias -1.088 | HT c=2 (nu=928): 51.5 (24 x 20) |
| pi0.5 | run_c2bern | 30k, 8 x 4, sbias -0.396 | HT c=2 (nu=1400): 97.4/97.1/99.4/92.6 = 96.6 (4 x 1000) |

## Reads
- Fractal step 500: gripper BCE/elem 0.34, accuracy 0.85 (chance 0.535), Student-t over 48 dims.

## Readout ablation (snap), added 2026-09-05
To separate the training-side effect (gripper out of sigma/gate, CE gradient) from the readout
effect (hard decision instead of a hedged mean), the control checkpoints are re-evaluated with the
binary channels snapped after unnormalization (`GROOT_SNAP_BINARY`, env-gated in
`gr00t/policy/gr00t_policy.py`, no training change):
| ablation | checkpoint | control | label |
|---|---|---|---|
| Fractal HT nu=224 + gripper snapped at 0.5 | ft_fr_nu224 @20k | 63.5 seeded | ssnap-nu224 |
| GR1 HT c=2 + hands snapped (majority of the 5 co-moving columns, thumb 3.0) | ft_gr1c2 @60k | 51.5 | gr1c2snap |
pi0.5: no snap ablation is meaningful. LIBERO's Panda gripper applies `np.sign(action)` with a fixed
speed (robosuite `panda_gripper.py`), so a hedged and a snapped command produce identical motion; on
LIBERO the Bernoulli head can act only through training. Reading of the four-way table: snap alone
reproduces the Bernoulli arm -> readout effect; snap flat but Bernoulli moves -> training effect;
both hurt -> the hard decision at p=0.5 removes the accidental dead-band of the released Fractal
readout (no latch for outputs in 0.25-0.75), to be fixed with a cost-aware threshold on p.

## Reads (continued)
- pi0.5 c2bern: gripper BCE 0.002-0.03, accuracy 99-100% at 30k; early cumulative eval reads level
  with or above the control at the same positions (object 100/100/90/98, goal 99/99/90/99, spatial
  100/100/100/100, libero-10 94; control at the same positions dips to 70-90).
- GR1 gr1bern: hand-bit BCE 0.08 / accuracy 96-97% at 18k, thumbs 0.001; training finished at 60k.

## Decision 2026-09-05 (user): no snap ablation on regression-trained controls
The snap evals of the non-Bernoulli controls (ssnap-nu224, gr1c2snap) were stopped: the readout
question is only meaningful on Bernoulli-TRAINED checkpoints (CE-trained logit, then MAP vs posterior
mean). Readout ablation = same Bernoulli checkpoint, hard readout sign(logit) (default) vs soft
readout tanh(logit/2) (env `GROOT_BERN_READOUT=soft` / `PI05_BERN_READOUT=soft`, banner printed).
- pi0.5 (run_c2bern @30k): 20-episode libero-10 check hard vs soft (pods yuchen-pev-bernro-*). Expected
  identical up to episodes where |logit| < 0.1, since robosuite applies np.sign; the banner logs
  frac(|logit|<0.1) to quantify that.
- GR1 (ft_gr1bern @60k): full 24-task soft-readout eval queued (label gr1bernsoft) behind the hard one.
- Fractal (ft_fr_bern_20k): soft-readout eval to follow the 20k hard eval.

## pi0.5 results (Bernoulli gripper head, run_c2bern @30k, 1000 eps/suite; control run_c2x8 nu=1400)
| suite | control | Bernoulli | note |
|---|---|---|---|
| object | 99.4 | 98.8 | every task within 2 episodes |
| spatial | 97.4 | 96.7 | t5 86 -> 77 (the hardest task for both), others within 2 |
| goal | 97.1 | 96.8 | t9 92 -> 87, t2 89 -> 91, others within 2 |
| libero-10 | 92.6 | 94.2 | t8 69 -> 77, t9 85 -> 89, t0 90 -> 94; others within 2 |
| **4-suite average** | **96.6** | **96.6** | identical to the decimal |
Readout check on the Bernoulli checkpoint (libero-10, 20 episodes, hard MAP vs soft tanh(logit/2)):
20/20 vs 20/20; |logit| mean 7.4, fraction of steps with |logit| < 0.1 = 0.0000 in both runs, i.e. the
CE-trained logit is saturated and the two readouts issue the same sign at every step; on LIBERO
(robosuite applies np.sign) the readout cannot matter and the head acts only through training.

## GR1 result (Bernoulli hand head, ft_gr1bern @60k, hard readout, 24 tasks x 20 eps)
Mean 47.7 vs control HT c=2 (nu=928, relative hands) 51.5; reference points HT 2d 47.5, flow 44.1.
Per task: 6 up (largest Cuttingboard->Cardboardbox 0.14->0.55, Bottle->Cabinet 0.65->0.75,
Placemat->Plate 0.30->0.45), 15 down (largest Placemat->Bowl 0.52->0.19, Tray->Tieredshelf 0.35->0.10,
Tray->Pot 0.85->0.60, Placemat->Basket 0.85->0.65), 3 tied. Single seed; NVIDIA quotes 5-6 points
run-to-run variance on this benchmark; the arm also carries the RELATIVE->ABSOLUTE hand representation
change. Soft-readout eval of the same checkpoint (gr1bernsoft) pending to separate readout from training.

## Fractal 12k mid-schedule reads (Bernoulli arm, seeded, 100 scenes): open-drawer 0.47, close-drawer 0.37
(control nu=224 endpoint @20k: 0.56 / 0.50; no seeded 12k control read on this recipe; close-drawer tracks
annealing, so only the 20k endpoint is comparable).

## GR1 readout ablation (same Bernoulli checkpoint ft_gr1bern @60k, 24 x 20)
| readout | mean | vs control 51.5 |
|---|---|---|
| hard (MAP, sign of logit) | 47.7 | -3.8 |
| soft (posterior mean tanh(logit/2), hedged hand targets) | 48.3 | -3.2 |
soft - hard = +0.6 (paired SE 3.6; 12 wins / 11 losses / 1 tie); per-task swings up to +/-0.25 in both
directions at 20 episodes/task (per-task SE ~11 pts) = evaluation noise. The hand readout does not matter
on GR1; the Bernoulli arm sits ~3.5 below the c=2 control with either readout (t ~ -1.1, not significant).

## Gradient balance of the factorized loss (GR1, 24 batches x 4, eval mode, real training batches)
| checkpoint | gradient into the shared representation: |g_CE| / |g_Student-t| | cos(g_CE, g_Student-t) | per-element |dL/dpred|: continuous (Student-t) / hand (CE) | loss parts (per element) |
|---|---|---|---|---|
| Bernoulli @60k | 0.048 (median 0.039) | +0.001 | 0.00656 / <0.00001 | -1.738 / 0.006 |
| initialization (base model, fresh sigma head) | 0.054 | +0.034 | 0.00051 / 0.00003 | -0.452 / 0.094 |
The CE term sends ~5% of the Student-t's gradient into the trunk, orthogonal to it (cosine ~0), at
initialization and at convergence; per element the hand columns receive 15x less gradient than the
continuous columns at init and ~0 at 60k (saturated logits, 97% accuracy). CE does not compete with the
continuous channels; if anything it is under-weighted, which does not matter for the hand bit.

## GR1 open-loop fit probe (400 identical training states from all 24 datasets; arm+waist columns, normalized)
Bernoulli arm vs c=2 control, mean squared residual ratio bern/control (episode-clustered 95% CI):
| states | n | control msr | bern msr | ratio | bern lower on |
|---|---|---|---|---|---|
| all | 400 | 0.00356 | 0.00194 | 0.54 [0.47, 0.63] | 81% |
| non-flip | 367 | 0.00351 | 0.00195 | 0.56 [0.47, 0.65] | 80% |
| flip (hand bit changes inside the chunk) | 33 | 0.00414 | 0.00181 | 0.44 [0.29, 0.61] | 91% |
Per joint group (non-flip / flip): left arm 0.60 / 0.35, right arm 0.56 / 0.50, waist 0.40 / 0.25; per chunk
step 0.54-0.58 (non-flip), 0.34-0.51 (flip); task family Close 0.46, Posttrain 0.58. Amplitude ratio to demo
0.994 (control 0.977), slope 0.984 (0.959), per-state cosine 0.987 (0.976). By control-residual quintile the
ratio is 1.40 / 1.35 / 0.75 / 0.53 / 0.44: the Bernoulli arm is 35-40% WORSE on the easiest 40% of states and
about 2x better on the hardest 40%. Hand-bit accuracy 97.7% (left) / 95.4% (right); at flip states the right
hand degrades along the chunk (0.94 -> 0.61 by step 8).
Reading: the gate-contamination hypothesis is refuted (the arm fit improved, most on flip states); the
closed-loop shortfall is not an arm-fit deficit. Candidates left: evaluation noise (benchmark run-to-run 5-6
pts vs the observed -3.5), the redistribution of precision away from the easiest states, hand flip timing
late in the chunk, the hand-representation change.

## HT hyper-parameter checks for the factorized head (GR1, 192 samples each)
1. **Initialization (sbias).** At step 0 (base model, Bernoulli config) the continuous-column residual RMS is
   0.264 vs the sigma implied by the inherited sbias -1.088 of 0.290: sigma started 11% too large; the
   continuous-only calibration is sbias = -1.197. (The hand columns' step-0 residual under the absolute
   representation is 0.96 and is irrelevant: the pretrained hand columns predicted relative offsets.)
2. **Sigma gradient.** CE -> sigma-head gradient norm exactly 0 at init and at 60k; Student-t -> sigma head
   5.5e-3 (init) / 2.8e-2 (60k). Sigma is driven by the continuous residual only. At 60k sigma tracks the
   actual continuous residual scale (ratio 0.91; control 0.96 on its 232 dims).
3. **nu.** Gate engagement at the nu used: Bernoulli nu=544 on d=136 -> 1.0% of samples with w<0.5 (0% with
   w<0.2); control nu=928 on d=232 -> 1.0% / 0%: identical engagement. Maximum-likelihood nu of the
   per-sample statistic is at the grid floor (8) for BOTH models, i.e. the arm-only residual is as
   heavy-tailed as the pooled one; removing the hand did not lighten the tails, so there is no evidence the
   arm-only head wants a different c. (As in the campaign, NLL prefers far heavier tails than the SR optimum.)
   Also: Bernoulli continuous RMS at 60k 0.039 vs control 0.056 on these samples, consistent with the fit probe.

### Correction: the "easiest 40% worse" pattern was a selection artifact
Quintiles by the control's own residual give ratios 1.40/1.35/0.75/0.53/0.44; quintiles by the Bernoulli
residual give the mirror image 0.15/0.25/0.27/0.52/0.73 (regression to the mean); quintiles by the MEAN of
both residuals give 0.50/0.58/0.54/0.51/0.56 and by demo amplitude 0.48-0.69: the gain is uniform across
difficulty. Within the control's easiest quintiles the per-state median ratio is 0.59/0.65 and the mean is
pulled above 1 by ~4 states out of 160 with a 10-30x larger Bernoulli residual on 1-3 arm joints (non-flip).
Error tails (400 states): chunk msr > 1e-2: control 29 states, Bernoulli 14; > 2e-2: 11 vs 2; max |error|
p50/p90/p99/max 0.131/0.298/0.546/0.799 (control) vs 0.097/0.236/0.443/0.502 (Bernoulli); states where one
model's max error exceeds the other's by > 0.1: Bernoulli worse on 16, control worse on 66. The Bernoulli
arm's arm-fit is better in mean, median and tail; no open-loop fit quantity explains the closed-loop gap.

## New GR1 arms (user request, 2026-09-05): tail-shape sweep with the Bernoulli hand head
Same recipe as ft_gr1bern (60k, batch 512, sbias -1.088, absolute hands, BCE hand head), only the continuous
likelihood changes:
| arm | continuous likelihood | run | node |
|---|---|---|---|
| ft_gr1bern (done) | Student-t nu=544 (c=2 on d=136) | 47.7 hard / 48.3 soft | |
| ft_gr1bern_nu5 | Student-t nu=5 (near the ML optimum of the residual; aggressive gate) | queued behind the gr1bern-r2 eval | .21 |
| ft_gr1bern_hg | heteroscedastic Gaussian (nu=inf, `--ht-hg-steps 60000`, no gate) | queued behind the fractal training | .13 |
Note: sbias kept at -1.088 for comparability with ft_gr1bern (the continuous-only calibration would be -1.197).

## Training dynamics (GR1, from the training logs; losses not comparable across d and nu)
| window | control grad-norm mean / max / frac>10 | Bernoulli grad-norm mean / max / frac>10 |
|---|---|---|
| 0-3k | 3.58 / 7.4 / 0 | 4.79 / 9.5 / 0 |
| 12-15k | 4.33 / 10.0 / 0.003 | 2.98 / 8.3 / 0 |
| 27-30k | 5.82 / 14.3 / 0.063 | 3.30 / 8.3 / 0 |
| 42-45k | 6.08 / 17.7 / 0.047 | 3.28 / 8.0 / 0 |
| 57-60k | 5.19 / 15.8 / 0.007 | 2.56 / 4.3 / 0 |
The Bernoulli arm optimizes more smoothly: gradient norms about half the control's from 12k on, no spikes
above 10 (the control has them in 0.3-6% of steps mid-run, consistent with hand-flip residuals hitting the
pooled statistic). Together with the 45% lower endpoint residual, the arm channels are better optimized
under the factorized head by every training-side measure.

## GR1 second evaluation runs (same checkpoints, fresh scene draws; 24 tasks x 20 eps each)
| arm | run 1 | run 2 | two-run mean |
|---|---|---|---|
| Bernoulli hand head (ft_gr1bern, hard readout) | 47.7 | 51.9 | 49.8 |
| control HT c=2 (ft_gr1c2) | 51.5 | 49.3 | 50.4 |
Difference of two-run means -0.6 (paired SE across tasks 2.0). Run-to-run swings of +4.2 and -2.2 points
for identical checkpoints match NVIDIA's stated 5-6 point variance on this benchmark. The earlier -3.8 was
evaluation noise; the Bernoulli head is level with the control on GR1 closed loop while fitting the arms
45% better open loop. (Per-task run-to-run mean |diff| 0.12-0.14, i.e. 2-3 episodes of 20.)

## GR1 closed-loop rollouts with dumps (3 tasks x 20 scenes per policy, ego-view videos + per-step actions)
Seeding does NOT pair scenes across policies on this harness (same seed+env slot first-frame |diff| 52 vs 55
between unrelated scenes), so these are unpaired same-task samples.
| task | Bernoulli | control |
|---|---|---|
| PlacematToBowl | 4/20 | 13/20 |
| TrayToTieredshelf | 4/20 | 4/20 |
| TrayToPot | 14/20 | 9/20 |
Opposite-sign task effects, level on average, as in the 24-task evals (two-run means 49.8 vs 50.4).
Hand-command signature in the 8 rendered PlacematToBowl failures (Bernoulli) vs control successes: right hand
first close at chunk 7-11 (Bernoulli) vs 12-15 (control) in 6/8, more close events (up to 7 vs 2-3 = premature
closure and retries), and spurious LEFT-hand closes late in 4/8 failing episodes (never in the control). Under the
control's relative hand parametrization the executed target is state + predicted offset, so a hedged offset
does not close the hand until the offset exceeds the finger travel: an implicit dead-band the MAP readout at
p=0.5 lacks (and tanh(logit/2) cannot restore because the logits are saturated). Test: readout threshold
p>0.8 / p>0.95 (`GROOT_BERN_THRESH`), same checkpoint, same 3 tasks.

### Closing-threshold readout test (same Bernoulli checkpoint, same 3 tasks, 20 scenes each, unpaired)
| readout | PlacematToBowl | TrayToTieredshelf | TrayToPot | total |
|---|---|---|---|---|
| control (HT c=2, relative hands) | 13 | 4 | 9 | 26/60 |
| Bernoulli, close at p>0.5 | 4 | 4 | 14 | 22/60 |
| Bernoulli, close at p>0.8 | 6 | 3 | 12 | 21/60 |
| Bernoulli, close at p>0.95 | 4 | 1 | 8 | 13/60 |
Raising the closing threshold does not recover PlacematToBowl and hurts at 0.95; with saturated logits the
threshold barely moves the switch time. The premature-closure reading is not fixable at the readout, and the
readout knob is exhausted (soft, hard, thresholds). The opposite-sign per-task effects with an equal mean are
most consistent with per-task variance between single training runs (plus the hand-representation change);
only 24-task means with >= 2 eval runs (+/-2) are readable, per-task comparisons of single runs are not.

## Fractal result (Bernoulli gripper head, ft_fr_bern_20k @20k, seeded 100 scenes/task, paired)
| task | flow | HT control (nu=224) | Bernoulli | control-only / Bernoulli-only, p |
|---|---|---|---|---|
| coke | 0.88 | 0.94 | 1.00 | 0/6, p=0.03 |
| object | 0.76 | 0.82 | 0.84 | 11/13 |
| move-near | 0.99 | 0.94 | 0.88 | 12/6 |
| open-drawer | 0.59 | 0.56 | 0.41 | 31/16, p=0.04 |
| close-drawer | 0.75 | 0.50 | 0.62 | 18/30, p=0.11 |
| place-in-drawer | 0.07 | 0.05 | 0.04 | 5/4 |
| **mean** | 67.3 | 63.5 | **63.2** | pooled 77/75, p=0.94 |
Level with the HT control (redistribution: coke/close-drawer up, open-drawer/move-near down); vs flow pooled
92/67 (p=0.057). Soft-readout eval of the same checkpoint follows (label sbernsoft-20k).
Next arm (user): Bernoulli + large nu on the continuous channels, ft_fr_bern_nu896 (nu=896, c~4.3 on d=48;
joint-head reference c=4 = 58.0), node .20; eval chain sbernnu896-20k.

## Hand-only replanning (GR1; arm executes the 8-step anchor chunk, hand refreshed every 2 env steps; 20 scenes/task, unpaired)
| policy | PlacematToBowl | TrayToTieredshelf | TrayToPot | total |
|---|---|---|---|---|
| control, horizon 8 | 13 | 4 | 9 | 26/60 |
| control, hand every 2 | 5 | 4 | 11 | 20/60 |
| Bernoulli, horizon 8 | 4 | 4 | 14 | 22/60 |
| Bernoulli, hand every 2 | 10 | 5 | 10 | 25/60 |
No consistent effect: PlacematToBowl moves in opposite directions for the two policies and the totals change by
-6 / +3 on 60 scenes (SE ~4). Denser hand decisions do not resolve the grasp-timing entropy; the per-task
swings are the same 20-scene noise seen everywhere on GR1. Closed.

## Fractal readout ablation on the Bernoulli checkpoint (seeded, same 100 scenes)
| readout | coke | object | move-near | open-drawer | close-drawer | place | mean |
|---|---|---|---|---|---|---|---|
| hard (MAP) | 1.00 | 0.84 | 0.88 | 0.41 | 0.62 | 0.04 | 63.2 |
| soft (posterior mean; the latch fires only for |2a-1|>0.5) | 0.98 | 0.83 | 0.89 | 0.43 | 0.64 | 0.01 | 63.0 |
Identical within noise on every task: the CE logits are saturated, so even where the environment's latch
semantics differ (Fractal), the readout is not a lever. Bernoulli line closed on all three stacks.
