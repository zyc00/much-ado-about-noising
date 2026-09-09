# GR1: what the HT head learns differently from MSE / flow (2026-08-28)

Question: does the HT head help policy learning by fitting small actions better, or by committing
to actions (larger amplitude, sharper hand transitions)? Study on GR00T N1.7 finetuned on the 24-task
GR1 RoboCasa mixture (60k steps each): `mse`, `l1`, `flow` (4 NFE), `ht464` (incumbent, nu=2d),
`ht928` (fixed c=2, nu=4d).

Two measurements:

1. Training-set fit on identical states (open loop). The same 400 training samples (identity-tagged,
   47 episodes of `PosttrainPnPNovelFromPlateToPlate`, all 24 datasets in the loader so the
   normalization statistics equal the training run's) are pushed through every checkpoint; predictions
   are compared with the demonstration chunk (8 steps x 29 dims, relative joint offsets) in the model's
   normalized units and in radians / centimetres (MuJoCo forward kinematics of the right wrist).
   Sanity check: the training-path forward loss on these batches equals the logged training loss
   (MSE 0.0084 vs 0.0083 at step 60k).
2. Closed-loop rollouts with trajectory dumps: 4 tasks x 10 episodes x {ht464, mse, ht928}, seed 1234,
   720 env steps max (90 chunks of 8), official harness. Scenes are re-randomized per run, so the
   comparison across models is unpaired.

Files: `fig_gr1_mechanism.png`, `fig_gr1_example_ep175_f48.png`, `fig_gr1_example_ep814_f178.png`,
`fig_gr1_closedloop_basket.png`,
scripts in `scripts_gr1_mechanism/` (`fit_dump_gr1_v2.py` on the cluster, `analyze_fit_v2.py`,
`analyze_rollouts_gr1.py`, `make_gr1_example_fig.py`, `make_gr1_mechanism_fig.py`).

## 1. Training-set fit on identical states (N=400)

Arm = 14 arm joints (dims 0-13); hand = 12 hand joints (dims 14-25); waist = 3.

| head | arm residual rms (rad) | arm residual, normalized | R^2 step 0 / steps 1-7 | amplitude slope pred~GT | cos(pred, GT) median | right-wrist endpoint error, median (cm) | hand residual (normalized) |
|---|---|---|---|---|---|---|---|
| MSE | 0.0291 | 0.0771 | 0.70 / 0.88-0.93 | 0.926 | 0.969 | 1.38 | 0.114 |
| L1 | 0.0325 | 0.0856 | 0.63 / 0.85-0.92 | 0.888 | 0.961 | 1.47 | 0.173 |
| flow (mean of 4 draws) | 0.0252 | 0.0668 | 0.79 / 0.91-0.95 | 0.943 | 0.979 | 1.14 | 0.124 |
| HT nu=464 | 0.0208 | 0.0554 | 0.85 / 0.94-0.96 | 0.963 | 0.987 | 0.86 | 0.115 |
| HT nu=928 | 0.0211 | 0.0563 | 0.84 / 0.93-0.96 | 0.961 | 0.986 | 0.92 | 0.106 |

Ordering of the arm residual (HT < flow < MSE < L1) is the ordering of the 24-task success rates
(47.5 / 44.1 / 37.8 / 17.2).

Arm residual by demonstration-offset magnitude (normalized units; bin population in % of elements):

| head | [0, 0.05) 24% | [0.05, 0.1) 20% | [0.1, 0.2) 25% | [0.2, 0.4) 18% | [0.4, 0.8) 12% |
|---|---|---|---|---|---|
| MSE | 0.062 | 0.064 | 0.073 | 0.096 | 0.095 |
| flow | 0.055 | 0.058 | 0.064 | 0.083 | 0.076 |
| HT 464 | 0.043 | 0.049 | 0.054 | 0.072 | 0.058 |
| HT/MSE | 0.69 | 0.77 | 0.74 | 0.75 | 0.61 |

The HT gain is present in every magnitude bin (25-40 % lower residual) and is largest in the
largest bin. Small actions are fitted better, but no more than large ones.

Per joint group (residual rms, normalized): left arm MSE 0.050 / HT 0.031, right arm 0.097 / 0.072,
waist 0.046 / 0.034, hands 0.114 / 0.115. The gain is confined to the continuous arm/waist channels;
the hand channels (discrete open/close levels) are fitted equally by MSE and HT.

Per state (arm chunk residual): HT is lower than MSE on 88 % of the 400 states; median ratio
MSE/HT = 1.54. The gain grows with how badly MSE fits the state (MSE-residual deciles 0 to 9:
HT-MSE = -0.005, -0.011, -0.018, -0.015, -0.020, -0.021, -0.027, -0.030, -0.025, -0.051).
Projection of (pred_HT - pred_MSE) onto MSE's residual: 0.55-0.59 at every chunk step (HT's
prediction moves 56 % of the way from MSE's prediction to the demonstration; flow 0.31-0.36; L1 0.06).

Amplitude and commitment:

- Arm amplitude: all heads predict demonstration-speed offsets on training states (arm offset rms
  over the chunk: demo 0.107 rad, MSE 0.105, HT 0.106, flow 0.104, L1 0.103). Regression slope
  pred~GT: MSE 0.926, HT 0.963 (shrinkage 7 % vs 4 %); median |pred|/|GT| 0.99 vs 1.00. The
  difference is in direction and per-joint accuracy (cos 0.969 vs 0.987), with 0.5 cm lower wrist
  endpoint error at 8 cm typical displacement.
- Hand transitions inside the chunk (46 of 400 states contain a hand flip of size 1.39 in
  normalized units): every head predicts a smeared flip. Predicted / demonstrated flip size: MSE 0.11,
  HT 0.13, HT928 0.16, flow 0.25, L1 0.06; timing error median 2 chunk steps for all. Hand residual on
  these states: MSE 0.247, HT 0.276, flow 0.261. HT does not commit to hand flips more than MSE.

Conclusion for the question: the HT advantage on GR1 is a uniform precision gain on the continuous
arm channels (28 % lower residual, 0.5 cm lower end-effector error, largest on the states MSE fits
worst), with no gain on the hand channels and no difference in commanded amplitude or hand-flip
commitment. Small actions are not favoured over large ones.

## 2. Closed-loop rollouts (4 tasks x 10 episodes, seed 1234)

| task | HT 464 | MSE | HT 928 |
|---|---|---|---|
| CuttingboardToBasket | 0.90 | 0.50 | 0.73 |
| CuttingboardToPot | 1.00 | 0.90 | 1.00 |
| PlateToBowl | 0.60 | 0.36 | 0.30 |
| CuttingboardToCardboardbox | 0.40 | 0.60 | 0.45 |
| pooled (n=40-42) | 0.72 | 0.59 | 0.62 |

Tasks were chosen from the 24-task eval as the three largest HT-over-MSE gains and the largest
MSE-over-HT gain (Cardboardbox); the rollouts reproduce those signs at n=10.

Execution statistics (all episodes):

- Commanded arm offset rms by chunk step (rad): HT464 0.049 -> 0.105, HT928 0.045 -> 0.099,
  MSE 0.041 -> 0.087; demonstration (PlateToPlate) 0.049 -> 0.129. Measured arm motion per chunk:
  HT464 0.288 rad (successes) vs MSE 0.241. In closed loop HT drives the arm 20 % faster than MSE,
  although on training states the two heads predict the same amplitude (section 1). MSE's slower
  commands come from the states it drifts into.
- Failures are all timeouts (90 chunks); no head stalls (fraction of chunks with < 0.02 rad motion: 0).
  Time to first hand closure is the same (median 11-13 chunks).
- Failure taxonomy (never grasped / grasped and released without completing / >= 3 grasp attempts):
  HT464 0 / 5 / 6 of 11 failures; MSE 0 / 3 / 14 of 17; HT928 0 / 6 / 10 of 16. Closures per
  episode: successes 1.2-1.6, failures 3.4-3.9. MSE's failure mode is a drawn-out sequence of grasp
  attempts. The first closure is held for 5 chunks more often by MSE (80 %) than by HT464 (62 %;
  HT928 79 %), i.e. HT464 re-opens and re-grasps sooner rather than grasping more reliably on the
  first try; time from first closure to success is the same (median 9-10 chunks).
- Commanded arm amplitude by phase (reach / hold / after release): HT464 successes 0.083 / 0.099 /
  0.069, failures 0.071 / 0.087 / 0.059; MSE successes 0.063 / 0.079 / 0.066, failures 0.061 / 0.077
  / 0.052. MSE reaches more slowly in both outcomes.

Cause of the closed-loop amplitude gap is unidentified beyond the correlation with fit precision:
with unpaired scenes the rollouts cannot show which specific mis-prediction produces the missed grasp.

`fig_gr1_closedloop_basket.png` shows one episode per head (env 2, first episode; different scenes):
HT464 reaches, closes at chunk 20, lifts and places, done at chunk 26 (10 s). MSE closes at chunk 13,
holds the object for 30 chunks with the wrist rising only ~5 cm and commanded arm offsets of ~0.06 rad,
loses it at chunk 42, and re-grasps at chunks 55, 79, 86 and 88 until the 90-chunk timeout; its mean
commanded arm offset over the episode is 0.059 rad vs 0.091 for HT.

## 3. Concrete example

`fig_gr1_example_ep175_f48.png` (PlateToPlate episode 175, frame 48; the right hand is closing on the
object, next 8 steps = 0.4 s). Demonstration: wrist roll 0.14 -> 0.25 rad, wrist pitch 0.10 -> 0.15,
shoulder pitch -0.07 -> -0.17. MSE head predicts wrist roll ~0.01 and wrist pitch ~0.00 for all 8
steps (the wrist rotation is absent), shoulder pitch -0.06 -> -0.09; HT follows every joint
(right-arm residual 0.039 rad vs 0.096; wrist endpoint error 0.91 cm vs 3.95 cm at 11.4 cm demo
displacement).

`fig_gr1_example_ep814_f178.png` (episode 814, frame 178, lifting the object): demonstration shoulder
pitch 0.05 -> 0.16 rad, wrist pitch 0.10 -> 0.37, elbow -0.22 -> -0.68. MSE predicts 0.00 -> 0.02
shoulder pitch and 0.01 -> 0.18 wrist pitch (endpoint error 4.96 cm vs HT 1.38 cm; demo
displacement 12.8 cm).

Both examples are states that MSE fits poorly (top decile) and that involve a multi-joint motion of
10-13 cm; the MSE prediction keeps the large joints (elbow, shoulder pitch) and drops the wrist
rotation. Candidate list with residuals: `analyze_fit_v2.py` section [G].

## 4. Pitfall found on the way (affects earlier GR1 figures)

The GR00T mixture dataset recomputes the q01/q99 normalization statistics from the datasets it is
given and writes them into the processor (`ShardedMixtureDataset ... processor.set_statistics(...,
override=True)`). A loader built on a 4-dataset subset therefore normalizes states with different
statistics than the 24-dataset training run (right shoulder joints shifted by up to 0.34 rad after
un-normalization), and every checkpoint then predicts a constant elbow offset of about -0.17 rad at
chunk steps >= 1 (forward loss 0.041 instead of 0.0083). This is the source of the "flow commands
elbows ~0.3 rad lower from step 1" observation in the GR1 chunk-path figures
(`fig_chunk_paths_3d.png`, `fig_gr1_ee_clarity.png`); those figures' GR1 offsets were computed with a
subset loader and should be regenerated with all 24 datasets before use. Rule: build any GR1 probe
loader with all 24 dataset paths (or check `processor.state_action_processor.norm_params` against
the checkpoint's `statistics.json`) and confirm the forward loss matches the training log.

## 5. Generality check across tasks (16 shards, 14 datasets, 303 episodes, N=400)

Same procedure with 25 states from each of 16 shards (`fit_dump_gr1_v3.npz`; forward loss MSE 0.0075).
`fig_gr1_mechanism_crosstask.png`.

| head | arm residual rms (rad-equiv., normalized) | R^2 steps 1-7 | amplitude slope | cos median | wrist endpoint error median (cm) | hand residual | better than MSE (states) |
|---|---|---|---|---|---|---|---|
| MSE | 0.0843 | 0.84-0.92 | 0.894 | 0.960 | 1.54 | 0.097 | - |
| L1 | 0.0927 | 0.81-0.91 | 0.854 | 0.954 | 1.60 | 0.160 | 41 % |
| flow (mean of 4) | 0.0705 | 0.89-0.94 | 0.926 | 0.975 | (n/a) | 0.112 | 73 % |
| HT 464 | 0.0636 | 0.91-0.95 | 0.940 | 0.986 | 0.88 | 0.090 | 85 % |
| HT 928 | 0.0644 | 0.91-0.95 | 0.938 | 0.983 | 1.06 | 0.089 | 84 % |

Arm residual by magnitude bin, HT464/MSE: 0.75, 0.77, 0.79, 0.73, 0.75 (bins [0,0.05) ... [0.4,0.8));
gain by MSE-residual decile 0..9: -0.007, -0.010, -0.019, -0.018, -0.027, -0.024, -0.018, -0.029,
-0.041, -0.044. Projection of (pred_HT - pred_MSE) onto MSE's residual 0.51-0.52 at every step
(flow 0.34-0.37, L1 0.05-0.07). Joint groups: left arm 0.043 vs 0.057, right arm 0.079 vs 0.105,
waist 0.041 vs 0.044, hands 0.090 vs 0.097. In-chunk hand flips (33 states): predicted / demonstrated
flip size MSE 0.16, HT464 0.30, HT928 0.28, flow 0.35, L1 0.10; residual on those states
0.235 / 0.242 / 0.230 / 0.279 / 0.316.

All section-1 conclusions hold on the cross-task sample: uniform 21-27 % arm-residual reduction in
every magnitude bin, 85 % of states, larger on hard states, 0.66 cm lower wrist endpoint error,
demonstration-speed amplitude for every head. Two small differences from the single-task sample:
the hand residual is 7 % lower for HT here (0.090 vs 0.097), and HT's predicted hand flips are
larger (0.30 vs 0.16 of the demonstrated size, still smeared); neither changes the ranking.

## 6. Gradient budget: what MSE spends on aleatoric residual

Per-state squared residual (= MSE gradient weight) on the cross-task sample: the 8 % of states with
an in-chunk hand flip carry 28 % of MSE's squared residual (single-task sample: 12 % of states, 38 %);
the top-10 % states carry 46 %; the hand channels carry 52 % of the squared residual on 41 % of the
dims. On the flip states every head lands on the same residual (MSE 0.161, HT 0.167, flow 0.185 vs
0.06-0.08 elsewhere): the component is flip-timing jitter, which is aleatoric given the observation.
An HT gate with nu=464 and pooled sigma weights those states at 0.34 vs 0.98 for the rest, cutting
their gradient share from 28 % to 17 %; HT then fits the flip states slightly worse than MSE and the
remaining 92 % better (0.059 vs 0.077). This is the supported reading of "why": MSE's shared trunk
keeps spending a quarter to a third of its gradient on residual that cannot be reduced, and
under-fits the bulk; HT reallocates it. Caveat: the gate selects by residual size, so learnable hard
states are partly traded away as well (the OFT/pi0.5 decile probe: HT 26 % worse on the samples it
suppresses most); this is correlational, and the intervention test (MSE trained with flip states
down-weighted) has not been run.

## 7. Predicted sigma along successful rollouts (2026-09-07, `sigma_videos/`)

Closed-loop rollouts of the HT c=2 checkpoint (`ft_gr1c2/checkpoint-60000`) with the sigma decoder evaluated at every
policy call (server patch `GROOT_SIGMA_DUMP`, training definition of sigma: mean over the 8 executed steps x 29 channels
of softplus(s_raw + sbias), normalized units). One successful episode per task, five tasks; videos and six-keyframe
sheets in `sigma_videos/`, protocol in `sigma_videos/README.md`, renderer `scripts_gr1_mechanism/make_sigma_videos.py`.

| task (episode) | calls | sigma, first call | median after | peak | peak / median | sigma peaks | hand open/close transitions | peak leads transition by (env steps) |
|---|---|---|---|---|---|---|---|---|
| tray -> pot | 45 | 0.075 | 0.023 | 0.150 | 6.5 | 3 | 2 | 8, 16 |
| can -> drawer, close | 33 | 0.088 | 0.021 | 0.124 | 5.8 | 3 | 3 | 8, 24, 0 |
| placemat -> basket | 34 | 0.102 | 0.026 | 0.132 | 5.0 | 2 | 2 | 16, 16 |
| bottle -> cabinet, close | 51 | 0.090 | 0.027 | 0.211 | 7.9 | 8 | 3 | 8-32 (door phase adds peaks without hand transitions) |
| cutting board -> pan | 52 | 0.093 | 0.035 | 0.142 | 4.1 | 9 | 9 | 8-16 (nine grasp attempts, one peak each) |

Pattern, identical across the five tasks: sigma is small and flat while the arm reaches or transports (median 0.02-0.035),
rises 4-8x in the chunk that contains a contact transition (hand closing on the object, hand opening to release, hand
switching to push a door or drawer), leading the observed hand-state change by 8-24 env steps (0.4-1.2 s at 20 Hz, i.e.
the transition falls inside the predicted chunk), and returns to the floor within one or two calls. The first call of
every episode carries an elevated sigma (0.075-0.10) before the robot has moved. The waist channels carry a large sigma
throughout (mean 0.15-0.20, up to 0.45 at the first call), so the scalar sigma is a weighted mix in which the 3 waist
channels contribute about as much as the 26 arm and hand channels; arm and hand channel means track each other closely.
Reading: the head down-weights exactly the contact-transition chunks, where demonstration timing varies most, and keeps
full weight on the free-space motion. Rollout successes in these single-env runs: 1/4, 4/4, 4/8, 3/4, 2/4 (harness
numbers 0.90 / 0.85 / 0.85 / 0.65 / 0.55 at n = 20).
