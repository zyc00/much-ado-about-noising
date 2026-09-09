# Fractal (Google Robot) with the HT head: what actually happens (working note, 2026-09-04)

Question: why is Fractal the one stack where HT (single-pass heteroscedastic Student-t) loses to
flow (67.3 vs 63.5), and can HT reach 71+ by legitimate training-side optimization?

## 1. The deficit is one task, not the benchmark

Seeded protocol (`--seed 1234`: 100 distinct scenes, identical across arms, so per-scene paired):

| task | flow | HT c=2 | flow-only wins | HT-only wins | McNemar p |
|---|---|---|---|---|---|
| pick coke can | 0.88 | 0.94 | 6 | 12 | 0.24 |
| pick object | 0.76 | 0.82 | 12 | 18 | 0.36 |
| move near | 0.99 | 0.94 | 6 | 1 | 0.13 |
| open drawer | 0.59 | 0.56 | 24 | 21 | 0.77 |
| **close drawer** | **0.75** | **0.50** | **36** | **11** | **<0.001** |
| place in closed drawer | 0.07 | 0.05 | 7 | 5 | 0.77 |
| pooled, 600 paired episodes | 67.3 | 63.5 | 91 | 68 | 0.081 |

Only close-drawer is significant. Pooled over all six tasks the HT-vs-flow difference is p = 0.08.
HT is (non-significantly) ahead on both pick tasks. Restoring close-drawer to flow's level would
put HT at 67.7, not 71: 71 requires HT at roughly the released model's level on every task.

## 2. Evaluation-protocol confound (found and corrected)

`rollout_policy.py --n-envs 5` without `--seed` gives the 5 envs the same initial state: a
deterministic policy yields blocks of 5 identical outcomes, so "100 episodes" = 20 scenes
(SE ~11 points per task). `eval_fractal.sh` was changed on 09-03 to make the seed optional, and
the c=1.5 arm and the first beta=0.5 pass ran on that 20-scene protocol; they are not comparable
to the seeded numbers above and were re-run seeded. The unseeded protocol is also what NVIDIA's
README example uses.

## 3. What is ruled out

- **nu (tail weight).** c=4 (nu=896) is significantly worse than c=2 on object (p=0.02) and
  open-drawer (p=0.003). c=1.5 (nu=126) seeded: 0.97 / 0.95 / 0.73 / 0.51 / 0.54 / 0.06 = **62.7**
  (its unseeded 20-scene read of ~61 had object at 0.90 and open-drawer at 0.35 — both noise).
  On one protocol the nu row is 62.7 / 63.5 / ~58 for c = 1.5 / 2 / 4: c=1.5 and c=2 are the same
  policy within noise, c=4 is worse, and all sit 4-9 points under flow. Each value is a single
  training run. Consistent with the probe: the gate is inert at the
  operating point (median weight 1.011, 0.8% of samples below 0.5, family spread 1.03x), and nu
  only moves the gate.
- **Aggregate under-fitting.** Open-loop residual on 2000 training samples, HT/flow RMS ratio: pick 0.75,
  move-near 0.74, open-drawer 0.77, close-drawer 0.81, place 0.78; arm dims 0.83-0.90, gripper
  0.65-0.74; HT's margin grows along the chunk (0.83 at step 1 to 0.75 at step 8). HT fits
  close-drawer better than flow in aggregate and loses it closed-loop. This rules out aggregate
  under-fitting only; a deficit on a specific subset of samples remains possible (see 4b). Caveat
  (Codex): flow's residual uses a sampled action, which inflates it relative to a conditional
  mean; the K-sample-mean re-measurement is in progress.
- **Gross data volume.** Per-eval-task share of the 87,212 episodes: coke 1.1%, object 22.0%,
  move-near 23.2%, open 8.5%, close 8.0%, place 13.8%. Spearman(share, SR) = +0.2; coke has the
  least data and the best score; place-in-drawer has 12x more data than coke and scores 0.05.
  Close-drawer has 6,966 episodes. Open/close have only 3 distinct instructions (top/middle/bottom).
  This rules out volume as the explanation, not coverage of specific states.
- **Execution horizon.** n_action_steps = 1 is best (4 and 8 worse).
- **Code / config.** Same recipe, data and harness across arms; nu and beta verified live at the
  loss site (`[HT]` banner); NFE and sampler defaults verified in the model config.

## 3b. Three more things ruled out or measured (added 2026-09-04, later)

- **Flow's per-scene outcome is itself half luck.** A second flow run on the same 100 seeded
  close-drawer scenes with fresh sampling noise reproduces the aggregate exactly (0.75 -> 0.75) but
  agrees with the first run on only 70/100 scenes: 15 flip each way. Flow succeeds at least once on
  90 scenes and twice on 60. The HT deficit replicates against the second run (flow-only 39, HT-only
  14, p=0.0008), so it is not an artifact of one draw.
- **HT's failures do not track flow's difficulty classes.** HT succeeds on 30/60 of the scenes
  flow solves twice, 15/30 of the scenes flow solves once, and 5/10 of the scenes flow never
  solves: 50% in every class. Two flow draws give only a coarse difficulty estimate, so this shows
  a lack of association, not independence from latent scene difficulty.
- **Drawer type (top/middle/bottom) is not it.** Replaying the env RNG chain per seeded scene
  (env i: s0 = 1234+i; drawer = RandomState(s_k).choice; s_{k+1} = RandomState(s_k).randint(2^32);
  scene j -> env j%5, round j//5; validated against the clip overlays) gives 34/32/34 scenes.
  Close-drawer success by drawer: flow 68/81/76%, flow run 2 79/72/74%, HT c=2 59/53/38%,
  HT c=4 32/62/53%. HT c=2 is below flow on all three, and c=4's worst drawer differs from c=2's,
  so each deterministic policy carries its own idiosyncratic failure set. This rules out drawer
  identity as the sole cause; it does not test language conditioning more generally. Open-drawer
  shows no drawer-type effect either.
- Correction to the video note: the recording wrapper's own cap is 720 steps, so 300-step timeouts
  are saved; clips missing from a run are the last episode per env (ffmpeg not flushed) and a few
  truncated files.

## 4. What is still live

- **sigma-based family down-weighting.** All between-family reweighting comes from 1/sigma^2
  (spread 2.37x); the drawer families have the highest sigma (open 0.248, close 0.237 vs move-near
  0.182) and are down-weighted most. sigma tracks per-family RMS with log-log slope +1.07, so
  beta-NLL is the correctly specified counter. **Seeded beta=0.5 result: a reshuffle, not a
  fix.** coke 0.90, object 0.74, move-near 0.87, open-drawer **0.36**, close-drawer 0.59 (place
  pending); five-task sum 3.46 vs c=2's 3.76 and flow's 3.97, i.e. ~58-59 overall. On
  close-drawer it rescues 28 scenes c=2 failed and loses 19 that c=2 solved (p=0.24 vs c=2), and
  stays significantly below flow (29/13, p=0.02). Over the four finished tasks it is
  indistinguishable from c=2 (62/52, p=0.40) and worse than flow (69/41, p=0.01). Its drawer-type
  pattern (top/middle/bottom 47/72/59%) differs from c=2's (59/53/38%) and c=4's (32/62/53%):
  each deterministic HT variant fails a different, idiosyncratic set of scenes. Flattening the
  family weights moved gradient toward the drawers at the cost of the free-space tasks without a
  net gain. beta=1.0 (seeded eval pending) is the last loss-side candidate.
- **Deterministic mean vs stochastic sampling at contact — RESOLVED: mostly the mean field.**
  Failure videos (flow and HT alike): the arm reaches the open drawer's front, then holds or
  retracts without pushing through (300-step timeout); successes push it shut in 24-48 steps.
  Discriminator (flow with initial-noise scale 0, same 100 scenes): **0.68**, vs stochastic flow
  0.75 / 0.75 and HT 0.50. Deterministic flow vs stochastic flow: 18/11 and 23/16 discordant,
  p = 0.27 / 0.34 (not significant); deterministic flow vs HT: 34/16, **p = 0.015**. Gap
  decomposition on the mean: sampling noise +7 points (n.s.), learned policy +18 (significant).
  Deterministic flow also tracks flow's difficulty classes (46/60, 17/30, 5/10) while HT does not
  (30/60, 15/30, 5/10). So determinism per se is a minor part of the deficit; the HT head has
  learned a different, worse mean action at the contact state.

## 4b. What HT does in the final phase of failed episodes (trajectory dumps, first 25 seeded scenes)

Per-step end-effector pose and commanded action, HT vs flow, close-drawer. Over the last 60
steps of an episode (for failures this is the stall; for successes the final approach and push,
so the comparison is phase- and outcome-confounded and is descriptive):

| | \|commanded xyz\| | actual eef step (m) | gripper cmd |
|---|---|---|---|
| HT failures (n=10) | **0.023** | **0.0007** | 0.67 |
| HT successes (n=11) | 0.070 | 0.033 | 0.53 |
| flow failures (n=6) | 0.047 | 0.0055 | 0.42 |
| flow successes (n=16) | 0.061 | 0.024 | 0.49 |

HT's failures are stalls: it commands 3x less translation than in its own successes and the arm
does not move. Flow's failures keep commanding ~77% of success-level motion while blocked. On
the 5 paired scenes (HT fail, flow success) HT's stall position sits at mean dy = -0.099 m,
dz = +0.045 m from flow's push position (dx ~ 0): ~10 cm lateral and ~5 cm above the push line,
where it goes quiet. Flow's successful push is +x (0.059 m over the final 10 steps).

**Loss-level counterpart (training-set fit, 193 close-drawer samples, executed step's xyz, by
quartile of the commanded magnitude):** HT/flow residual ratio Q1 0.93, Q2 0.92, Q3 0.87,
**Q4 1.02**. HT is better on the three smaller quartiles and worse on the largest actions (mean
|a| 0.98): it shrinks them to 0.64 of scale vs flow's 0.73 (bias along the command -0.40 vs
-0.33). This is specific to close-drawer: open-drawer Q4 ratio 0.86, all other families 0.95.
**RETRACTED by the stronger test (4c).** The quartile result above compared HT against a single
flow SAMPLE, whose sampling variance is largest on large actions.

## 4c. Stronger loss-level test: HT equals flow's mean on the demonstrations, push phase included

600 close-drawer samples from 175 episodes (and 300 open-drawer, 300 other), flow's 8-sample mean
as comparator, push phase defined independently of the residual (last 40% of the episode by step
index and forward-dominant command), episode-clustered bootstrap CIs:

| close-drawer | HT / flow-mean RMS ratio | bias along command HT / flow | \|pred\|/\|gt\| HT / flow |
|---|---|---|---|
| push phase (n=163) | 0.99 [0.92, 1.06] | -0.105 / -0.111 | 0.87 / 0.88 |
| non-push (n=437) | 1.02 [0.99, 1.05] | -0.090 / -0.092 | 0.97 / 0.96 |
| all (n=600) | 1.01 [0.99, 1.04] | -0.094 / -0.097 | 0.94 / 0.94 |

Against flow's single sample the same ratio is 0.91, which is where the earlier "HT fits
close-drawer 19% better" (section 3) came from: that was a sampled-vs-mean artifact, not a fit
difference. On the training distribution HT and flow's mean are the same policy on close-drawer,
including on the push. (Open-drawer: HT slightly worse on non-push samples, 1.07 [1.02, 1.14].)
Section 3's "HT fits close-drawer better" and section 4b's "under-fits the push" are both
withdrawn.

Consequence: the closed-loop gap (HT 0.50 vs deterministic flow 0.68, p=0.015) is not a
fitting difference on demonstrated states. It arises off the demonstration distribution: the
deterministic rollout drifts to a stall pose (~10 cm lateral, ~5 cm above the push line, 4b) that
is outside the training support, and there HT's mean extrapolates to near-zero commands while the
flow-derived deterministic policy keeps pushing. This is the same failure anatomy measured on
RoboMimic (failure = extrapolation in the annulus around the training support).

**sigma-weighting test (training forward, same 600/300/300 samples, joined on episode+step):**
close-drawer push-phase sigma 0.218 vs non-push 0.205; effective weight w/sigma^2 20.9 vs 23.9
(ratio 0.88); gate w 1.02 vs 1.02; push samples are 25% of the bottom-quartile weights (25% =
no enrichment). sigma is flat across the episode (0.18-0.24 by decile, no pre-push peak). HT
does not down-weight the push phase in any meaningful way. Together with 4c this closes both
loss-level hypotheses: on the demonstrations HT neither under-fits nor under-weights the decisive
action; its close-drawer deficit is a closed-loop, off-distribution phenomenon.
- **Recipe gap (the largest lever).** NVIDIA's released checkpoint is `global_step32000` of a
  40k-step schedule with state_dropout 0.2, 16 GPUs x 64 = batch 1024, lr 1e-4 (its own
  `experiment_cfg/config.yaml`), not the README's 20k / dropout 0.5. Its training loss is still
  falling at 20k (0.150, lr 5.4e-5) and reaches 0.129 at 32k; our 20k run ends at 0.147 with lr
  decayed to 0. Our flow's -5.2 vs released is concentrated on picks (coke -12, object -18), i.e.
  the tasks HT is already good at. Arms launched on the released recipe: `ft_fr_nv40k_ht` (HT c=2)
  and `ft_fr_nv40k_flow` (matched flow), ~24 h each; released weights being evaluated in our
  harness as the faithfulness control.

## 4d. Large-action optimization under HT, quantified (close-drawer, 600 samples, sigma from the training forward)

| quartile of \|a\| | \|a\| | \|pred\|/\|a\| | resid RMS | sigma | gate w | w/sigma^2 | pull w\|r\|/sigma^2 |
|---|---|---|---|---|---|---|---|
| Q1 | 0.15 | 2.08 | 0.243 | 0.217 | 1.026 | 22.3 | 30.9 |
| Q2 | 0.36 | 1.11 | 0.224 | 0.179 | 1.018 | 30.6 | 38.5 |
| Q3 | 0.58 | 0.89 | 0.253 | 0.205 | 1.026 | 24.4 | 34.0 |
| Q4 | 0.93 | 0.73 | 0.296 | 0.237 | 0.983 | 18.1 | 31.2 |

The gate is inert on the largest actions (Q4/Q1 0.96). sigma is 9% higher on large-action
samples, so their weight per unit residual is 19% lower (Q4/Q1 0.81); their larger residual
compensates, and the output-space pull on the largest actions equals that on the smallest (31.2
vs 30.9) where MSE would give them 22% more. That is the entire large-action property of HT here:
a mild relative under-pull through the heteroscedastic 1/sigma^2, not through the tail, and it
leaves no fit deficit on demonstrated pushes (4c). The 0.73x shrinkage of the largest quartile is
regression-to-the-mean shared with flow's mean. Normalization: q01/q99 min-max maps forward x
[-0.2245, +0.1782] to [-1, 1] (raw max +2.99), so ~7% of push-phase forward commands are clipped
at 1.0 — shared with the flow baseline and NVIDIA's recipe. Reviewer (round 5): no HT-side
training change is justified by these numbers (precision floor and Kronecker covariance not
justified; q999 normalization the only remotely plausible arm, likely small); large-action
optimization is not the bottleneck; the recipe is the lever. Cautions: output-space pull is not
the parameter gradient; sigma is endogenous.

**Shrinkage vs nu (same 600 samples; c=1.5 and c=4 checkpoints):** push-phase |pred|/|a| 0.869 /
0.875 / 0.882 for c=1.5 / 2 / 4 vs flow-mean 0.877; largest-quartile 0.729 / 0.730 / 0.741 vs
0.740; residual RMS within 1%. The tail weight does not change how large actions are fit; all
four are the same open-loop policy on close-drawer, and their closed-loop differences (c=1.5,
c=2 63.5, c=4 ~58) are entirely off-distribution behaviour.

## 5. Reviewer (Codex, gpt-5.6-sol) assessment (five rounds)

Round 1 mechanism ranking: sigma down-weighting ≈ loss-induced wrong mean field at contact > dithering >
precision-vs-robustness > instruction ambiguity. The 32k/dropout-0.2 recipe is the highest-gain
legitimate intervention; beta=1 second; inference-time sampling from the fitted t is a test-time
change and out of scope for a training-only target; rebalancing toward drawer families is
test-set targeting. Credible "HT >= 71" needs several training seeds and disjoint scene panels;
a one-seed 71 vs 67.3 is ~1.85 SE before training variance. Its single-seed prediction for HT at
32k of the released recipe: 66-73, centered ~69.5 (71 about one chance in three). NVIDIA's own
README states 5-6% run-to-run variance from augmentation.

## 6. Status
Done: deterministic-flow discriminator; seeded beta=0.5 (open/place pending); trajectory dumps;
drawer-type replay; flow reproducibility. Pending: seeded beta=1.0; seeded c=1.5; released
checkpoint in our harness; 600-sample close-drawer fit dump; HT and matched flow at 32k/40k of
the released recipe (~24 h). Reviewer round 3: finish these before launching anything new; if a
loss ablation is ever justified it should be a precision floor on 1/sigma^2 relative to an EMA
median (not a gate floor), run only if push states are shown to sit in the low-weight tail.

## 7. Tuning campaign (goal: HT >= 67 on Fractal; launched 2026-09-04)

All on the released recipe (40k-step cosine, dropout 0.2, batch 1024, lr 1e-4 unless stated),
HT c=2 (nu=224), 8 GPUs each; seeded 6-task evals chained at checkpoints 20k and 32k, paired
against `ft_fr_nv40k_ht` (same recipe, no change) and `ft_fr_nv40k_flow`.

| arm | change | rationale |
|---|---|---|
| `ft_fr_nv40k_ht` | none (baseline of the campaign) | the recipe lever alone |
| `ft_fr_nv40k_ht_clamp0` | sigma clamp a=0: sigma_eff = EMA(pooled residual RMS) for every sample | homoscedastic Student-t; removes the 19% lower weight that large-action samples receive through 1/sigma^2 (4d) |
| `ft_fr_nv40k_ht_clamp01` | sigma clamp a=0.1: sigma in [0.9, 1.1] x that reference | keeps mild heteroscedasticity |
| `ft_fr_nv40k_ht_lr2e4` | lr 2e-4 | **killed at 5k**: unstable — loss -0.47..-0.56 vs lr5e5's -0.78 at the same step, grad norm 2-3 (spike 64) vs 0.6-0.8 elsewhere |
| `ft_fr_nv60k_ht` | 60k-step cosine, lr 1e-4, save every 4k | the steps knob: 1.5x the released schedule; evals at 48k (80% of schedule, like NVIDIA's 32k/40k) and 60k, close-drawer read at 24k |
| `ft_fr_nv40k_ht_lr5e5` | lr 5e-5 | the opposite direction, in case 1e-4 is too aggressive for the sigma head |
| not run: batch 512 @ 40k | | halves the data seen at fixed steps; confounded against the more-optimization hypothesis |

Implementation: `ht_sigma_clamp` plumbed through model config, FinetuneConfig, launch_finetune
and setup (same pattern as `ht_gate_floor`); loss-site banner `[ht-sigclamp] a= sref= band=
clipped=%` printed at step 1 and every 500 steps; `--learning-rate` is a FinetuneConfig field.

## 8. Released checkpoint in our harness (calibration, seeded 100 scenes)

| | coke | object | move-near | open | close | place | mean |
|---|---|---|---|---|---|---|---|
| NVIDIA reported (their unseeded protocol) | 1.00 | 0.94 | 1.00 | 0.65 | 0.69 | 0.07 | 72.5 |
| same weights, our seeded protocol | 0.97 | 0.94 | 0.98 | 0.50 | 0.64 | 0.10 | **68.8** |
| our flow, 20k / dropout 0.5 | 0.88 | 0.76 | 0.99 | 0.59 | 0.75 | 0.07 | 67.3 |
| our HT c=2, 20k / dropout 0.5 | 0.94 | 0.82 | 0.94 | 0.56 | 0.50 | 0.05 | 63.5 |

The harness is faithful on the picks (within 3 points) and the 3.7-point headline difference is
the evaluation protocol, concentrated on the drawers (open 0.65 -> 0.50, close 0.69 -> 0.64).
Consequences: (i) the reference ceiling for "HT >= 67" is 68.8 on this protocol; (ii) our 20k
flow is within 1.5 of the released model, so the recipe gap for flow is ~1-2 points, not 5;
(iii) HT already matches the released model on open-drawer (0.56 vs 0.50) and beats our flow on
both picks; the whole remaining gap to 67 is close-drawer (0.50 vs 0.64 released / 0.75 our flow).
The earlier expectation that the released recipe alone adds ~5 points to HT was built on the
72.5 figure and is therefore too optimistic by roughly the protocol offset.

## 9. First campaign result: the sigma clamp (checkpoint 12k of the 40k recipe, LR ~8e-5)

| arm | close-drawer @12k | by drawer (top/middle/bottom) | vs base, McNemar |
|---|---|---|---|
| base HT (per-sample sigma) | 0.05 | 3/34, 1/32, 1/34 | — |
| clamp01 (sigma within +/-10% of the shared scale) | 0.24 | 8, 9, 7 | 21 vs 2, p=7e-5 |
| clamp0 (single shared sigma = EMA of pooled residual RMS) | **0.46** | 15, 14, 17 | 44 vs 3, p=2e-10 |

clamp0 vs clamp01: 33 vs 11, p=0.001. clamp0@12k vs the old HT endpoint (0.50): p=0.65 (tie);
vs the released model (0.64): 30 vs 12, p=0.008. Base@12k scores 0.92 on coke, so the base
checkpoint is not generally unready: the deficit is close-drawer-specific. Interpretation: the
per-sample heteroscedastic sigma delays close-drawer learning; removing it (homoscedastic
Student-t) learns the task far earlier. Mid-schedule caveat: the base will improve as it
anneals; the 20k/32k evals decide the endpoints. Launched `ft_fr_clamp0_20k` (clamp0 on the
standard 20k / dropout-0.5 recipe) for a fully annealed, directly comparable endpoint vs 63.5.

### Campaign additions (2026-09-05)
- `ft_fr_nv60k_ht` and `ft_fr_clamp0_20k` stopped by the user (steps knob has ~1.5 points of headroom after calibration; the standard-recipe clamp0 was on a slow node).
- `ft_fr_nv40k_ht_lr2e4` killed at 5k (unstable); `_lr5e5` stopped after its 20k checkpoint (weak lever: close-drawer 0.17 at 12k vs clamps 0.24/0.46).
- Base HT at 20k of 40k: coke 0.98, object 0.87, move-near 0.94, open 0.30, close 0.32 — picks above the old endpoint, drawers far below it (half-annealed; open-drawer's drop 0.56 -> 0.30 may be the recipe's dropout 0.2).
- Launched `ft_fr_nv40k_c0d5` (clamp0 + dropout 0.5, 40k schedule; evals: close-drawer at 20k, full at 32k) and `ft_fr_c0d5_20k` (clamp0 + dropout 0.5, standard 20k recipe; close-drawer at 12k, full at 20k). Rationale: the clamp is the lever; dropout 0.2 is the one recipe change suspected of hurting the drawers.
- Reference on this protocol: released model 68.8; target 67 needs open+close ≈ 1.10 with picks ≈ 2.8.

### 20k reads (LR ~5e-5): the clamp advantage does not persist

| arm @20k | coke | object | move-near | open | close | place | sum |
|---|---|---|---|---|---|---|---|
| base HT | 0.98 | 0.87 | 0.94 | 0.30 | 0.32 | 0.01 | 57.0 |
| clamp0 | 0.94 | 0.77 | 0.94 | 0.27 | 0.25 | (running) | |
| clamp01 | 0.96 | 0.84 | 0.96 | 0.32 | 0.18 | (running) | |
| lr5e5 | 0.98 | 0.74 | 0.91 | (running) | (running) | | |

Close-drawer 12k -> 20k, same scenes: base 0.05 -> 0.32 (lost 5, gained 32); clamp0 0.46 -> 0.25
(kept 12, lost 34, gained 13); clamp01 0.24 -> 0.18 (kept 2, lost 22, gained 16). The per-scene
success sets are nearly disjoint between checkpoints: the close-drawer policy churns during
training under every loss, and the 12k ordering was a transient (homoscedastic learns the task
earlier but does not retain it — the mid-run sag seen on other stacks). At 20k base vs clamp0
p=0.37, base vs clamp01 p=0.03 (base better). Open-drawer is 0.27-0.32 for all three arms on the
dropout-0.2 recipe vs 0.56 for the old dropout-0.5 HT: recipe-driven, not loss-driven.
Decision: the sigma clamp is not a lever (clamp0 runs to 32k as the final check); clamp01
stopped at 24k; its node goes to `ft_fr_nv40k_d5` = HT c=2 on the 40k schedule WITH dropout 0.5
(the steps knob without the dropout confound). Arithmetic: picks ~2.79 + old drawers 1.06 +
0.05 = 3.90 -> ~65; 67 needs close-drawer ~0.6 from the longer anneal.

### Bernoulli gripper head (launched 2026-09-05, `ft_fr_bern_20k`)

Motivation (MLE view): the gripper target is binary (raw 0/1, mean 0.535, std 0.497; +/-1 after
min-max normalization), and the SimplerEnv wrapper converts the prediction a in [0,1] to a relative
command -(2a-1) with a 15-step sticky latch that fires only when |2a-1| > 0.5. At a state where the
demonstrations disagree about WHEN to close (the timing multimodality the probes found), any
symmetric continuous likelihood puts its location at the mean 2p-1, which for 0.25 < p < 0.75 is
"no gripper action". In HT the +/-2 gripper residual at every ambiguous chunk step also inflates
the shared sigma and S for the whole 56-dim sample, contaminating the gate and the weighting of
the six continuous channels (cf. GR1: HT's gain was on continuous channels only).

Change (`ht_gripper_bce`, default off): the decoder's gripper column is treated as a logit and
trained with BCE against the 0/1 target over the valid chunk steps (weight 1.0 = the joint
factorized NLL); that column is removed from the Student-t residual, sigma pooling and gate; at
inference the column is replaced by sign(logit) so the wrapper latches a decisive open/close.
Banner `[ht-gripper-bce] dim=6 weight= bce/elem= acc= student-t dims/sample=` at step 1 and every
500 steps. Config plumbed through model config / FinetuneConfig / launch_finetune / setup.

Arm: `ft_fr_bern_20k` = original HT (c=2, per-sample sigma) + Bernoulli gripper on the standard
20k / dropout-0.5 recipe, so `frnu224` (63.5) is the exact control. clamp0 stopped at 24k to free
the node (its 20k eval 53.5 was the worst of the set; clamp effect transient). Evals: open/close
drawer at 12k, full 6-task at 20k. Expected effect: open-drawer and place-in-drawer (grasp timing)
rather than close-drawer (a push).

## HT-only tuning, round 2 (2026-09-06)
User decision: Bernoulli training stopped (all three stacks level with the joint head); goal remains HT >= 67.
| arm | recipe | node | eval |
|---|---|---|---|
| ft_fr_d5_30k | HT c=2 nu=224, dropout 0.5, 30k cosine (steps without the dropout confound) | .13 | seeded 6 tasks @30k (sd530k-30k) |
| ft_fr_lr5e5_20k | control recipe (20k, dropout 0.5) with lr 5e-5 | .20 | @20k (slr5e5d5-20k) |
| ft_fr_nv40k_ht (resume 28k->40k) | released recipe (40k, dropout 0.2) | .21 when free | @28k now (sbase-28k), @32k, @40k |
Round-2 additions (2026-09-06, reviewed design): A4 `ft_fr_bs512_40k` = batch 512 x 40k at dropout 0.5 (same
sample budget as the control, 2x optimizer updates; node .22, eval sbs512-40k); D1 = uniform average of the
last five checkpoints (16k..20k) of BOTH heads (`swa_ckpt.py`; evals sswa-nu224 / sswa-flow), kept as a paper
row only if applied to both. Execution-horizon diagnostic dropped (horizon 1 already recorded as best).
Collector `fr_collect.py` now requires seed 1234 and 100 scenes per task before a task enters a mean or a
paired test (an earlier version accepted the literal "--seed" token and any scene count).
Base recipe (40k/dropout 0.2) HT @28k seeded: coke 0.99, object 0.89, move-near 0.99, open-drawer 0.13 (0.30 at
20k), close-drawer 0.38 (0.32 at 20k): the picks match the released flow checkpoint but the drawers collapse under
dropout 0.2; the 28k->40k resume (A3) is cancelled and node .21 goes to `ft_fr_d5_40k` = HT c=2, dropout 0.5, 40k
schedule (the steps knob beyond A1's 30k; eval sd540k-40k).
Base recipe (40k/0.2) HT @28k complete: 0.99/0.89/0.99/0.13/0.38/0.02 = 56.7 (20k read 57.0): flat, drawers dead.
D1 checkpoint averaging (uniform, 16k..20k), seeded: HT 0.99/0.77/0.92/0.57/0.52/0.07 = 64.0 (endpoint 63.5);
flow 0.84/0.79/0.98/0.50/0.69/0.10 = 65.0 (endpoint 67.3). Averaging redistributes for HT and hurts flow; not a
lever, no paper row. `ft_fr_d5_40k` (dropout 0.5, 40k) launched on .21 (joint head, nu 224, batch 1024 confirmed).
Round-2 closing reads (seeded, 100 scenes/task): `slr5e5d5-20k` (lr 5e-5, dropout 0.5, 20k) coke 0.93 / object 0.75
/ move-near 0.95 / open-drawer 0.50 / close-drawer 0.57 (place pending) = 74.0 over 5 tasks vs flow 79.4 and HT c=2 75.2
on the same 5; `sbs512-40k` (batch 512, 40k) coke 1.00 / object 0.68 / move-near 0.95 (drawers pending). Neither arm
moves toward flow; the paper keeps HT 63.5 vs flow 67.3 with the drawer mechanism section. Tuning stopped on the
user's instruction (2026-09-05); remaining evals finish on their own.
`slr5e5d5-20k` complete: place-in-drawer 0.08 (8/101 episodes; the collector excludes a 101-episode task from the paired
test), six-task mean 63.0 = HT c=2 endpoint (63.5). Closed.
`sbs512-40k` complete (batch 512, 40k, seeded): coke 1.00 / object 0.68 / move-near 0.95 (101 eps) / open-drawer 0.53 /
close-drawer 0.35 / place 0.04 = 59.2, below the HT endpoint 63.5 and flow 67.3. Batch size is not a lever either.
Round 2 is closed: no arm beats the incumbent HT recipe; paper keeps HT 63.5 vs flow 67.3.

## Round 3 (2026-09-06): MSE warm phase, then HT (`ht_mse_steps`)

Hypothesis (user): the fractal HT arm misses an MSE warm-up. Two arms on the HT c=2 recipe (nu=224, 20k, batch 1024,
dropout 0.5), MSE for the first `ht_mse_steps`, then HT with sbias recalibrated at the switch to the MSE residual RMS
(`ft_fr_ms10ht`: switch at 10k; `ft_fr_ms15ht`: switch at 15k, sbias -0.509 -> -1.42, residual RMS 0.215). Seeded
protocol (SEED=1234, 100 scenes/task), paired McNemar from `fr_collect.py` (flow-only / HT-only scene counts).

| arm | coke | object | move-near | open-dr | close-dr | place-dr | mean | vs flow | vs HT c=2 |
|---|---|---|---|---|---|---|---|---|---|
| flow (endpoint) | 0.88 | 0.76 | 0.99 | 0.59 | 0.75 | 0.07 | 67.3 | | 68/91 p=0.081 |
| HT c=2 (endpoint) | 0.94 | 0.82 | 0.94 | 0.56 | 0.50 | 0.05 | 63.5 | 91/68 p=0.081 | |
| MSE | 0.98 | 0.69 | 0.97 | 0.46 | 0.60 | 0.07 | 62.8 | 95/68 p=0.041 | 81/77 p=0.811 |
| MSE 10k -> HT 10k (`sms10ht-20k`) | 0.97 | 0.73 | 0.85 | 0.50 | 0.61 | 0.05 | **61.8** | 99/66 p=0.012 | 89/79 p=0.488 |
| MSE 15k -> HT 5k (`sms15ht-20k`) | 0.98 | 0.52 | 0.94 | 0.57 | 0.56 | 0.03 | **60.0** | 104/60 p=0.001 | 98/77 p=0.130 |

MSE 10k -> HT 10k: 61.8, a tie with HT (p=0.49) and with MSE, and a significant loss to flow (p=0.012). Move-near
drops to 0.85 (every other arm 0.92-0.99); close-drawer sits at the MSE level (0.61), not the HT level (0.50), so the
warm phase removes the close-drawer deficit but pays for it elsewhere. MSE 15k -> HT 5k: 60.0, the lowest of the
four arms (pick-object 0.52 vs 0.69-0.82 elsewhere; place-in-drawer 0.03), a significant loss to flow (p=0.001) and
a numerical loss to HT (p=0.13). The longer the MSE phase, the worse the endpoint (63.5 -> 61.8 -> 60.0). An MSE
warm-up is not the missing ingredient. Round 3 closed 2026-09-07; the paper keeps HT 63.5 vs flow 67.3.

## Round 4 (2026-09-07): Muon optimizer, stopped

User request: HT c=2 recipe with Muon. Implemented in the GR00T trainer behind `GROOT_MUON=1` (Newton-Schulz
orthogonalized momentum on the 252 two-dimensional weights of the DiT and VL self-attention, 1.29B params; AdamW on the
rest; update RMS matched to AdamW so the recipe lr/schedule/clip are unchanged; runs under plain DDP with fp32 trainable
weights because ZeRO-2 hands the optimizer flattened partitions). Startup validated: step-10 loss -0.1812 identical to
four fresh DeepSpeed HT runs (flow+Muon 1.2332 vs fresh flow 1.2341); 57-62 GB per GPU; 2.05 s/it. `ft_fr_muon_ht`
launched 10:39 on .20 and was stopped by the user at step 910/20000 before its first checkpoint; the flow+Muon control
(`ft_fr_muon_flow`, .21) was deleted after 10 minutes; the AdamW-under-DDP bridge control never launched. No result.
Scripts: `run_fr_muon_ht.sh`, `run_fr_muon_flow.sh`, `run_fr_ddp_ht.sh`; patch `muon_patch.py`, test `muon_test.py`.
