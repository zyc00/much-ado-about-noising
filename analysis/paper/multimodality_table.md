# Chunk-level action multimodality: same-state K-sample probe

Protocol: for each training-set state, the policy receives K identical
observations (matching the evaluation configuration: observation history,
action-chunk length, sampler entry point, number of inference steps) and K
action chunks are sampled. Per state, the K x (chunk x action-dim) matrix is
centered and analyzed: total spread, effective rank (participation ratio of
singular values), and a 2-means clustering test on the first principal
component. Cluster separation is compared against a Monte-Carlo Gaussian null
at the same K and dimension (K=16: median 2.90, q95 3.97, max over 2000 draws
7.90); a pure single-mode Gaussian cloud produces separations up to ~8, so
only separations beyond that range indicate genuine structure.

WidowX extension (2026-09-03): the same checkpoint was additionally probed on
3000 Bridge training states and 1000 observations from its own saved rollouts.
The screen uses K=8. Confirmation fixes PC1 on 32 fresh samples and tests a
one- versus two-Gaussian fit on 64 further, independent samples, with BH
correction, delta-BIC >= 10, separation >= 2 pooled SD, and at least 20% mass
per component. One hundred on-policy states are confirmed from a uniform
random sample so that the rate is interpretable; the original extreme-state
confirmation is retained only as a discovery audit. Because evaluation
executes four of the eight predicted actions, a post-hoc check repeats the
same independent test on the executed four-step prefix. A subsequent causal
audit fits modes on that executed prefix, forks exact simulator states with 12
representative chunks per component, and follows each branch for 48 closed-loop
steps.

| policy | inference | SR | per-state sample spread | eff. rank | 2-means separation, median | states beyond null q95 (chance = 5%) | content of the variance |
|---|---|---|---|---|---|---|---|
| push-T MIP (vision) | 1 pass | 0.947 | 3e-4 px | 1.0 | — | — | none (float jitter) |
| push-T HT (vision) | 1 pass | 0.927 | 3e-4 px | 1.0 | — | — | none (float jitter) |
| GR1 HT | 1 pass | **0.475** | 0 (bitwise identical) | — | — | — | none (deterministic) |
| GR1 flow | 4 steps | 0.441 | 0.299 | 3.1 | 2.85 (= null median) | 5/40, all within null max | single mode + undirected sampling noise |
| GR00T WidowX flow | 4 steps | 0.571 | 1.258 | 2.9 | 3.55 (K=8 null median 3.09) | 82/1000 arm; 184/1000 gripper at screen | gripper timing + sparse arm-phase modes |
| pi0.5 HT | 1 pass | **0.976** | 0 (bitwise identical) | — | — | — | none (deterministic) |
| pi0.5 flow | 10 steps | 0.969 | 0.924 | 4.6 | 3.23 | 17/60; 14 beyond null max | single trajectory; timing of the gripper transition |

Notes.
- WidowX flow, unbiased deep confirmation: among 100 uniformly sampled
  on-policy states, 27 pass the full-chunk test, 22 the gripper test and 6 the
  arm test. Every full-chunk positive is accounted for by a gripper or arm
  positive (22 and 5, respectively; the sixth arm split is diluted in full
  space). A density-shape and raw-trajectory audit leaves three clearly
  resolved, arm-dominated profiles: two drawer states choose between executing
  a wrist rotation within this chunk and deferring it, and one eggplant state
  shifts the same lateral-motion pulse within the 8-step horizon. The largest
  examples differ by 12.4 degrees of yaw or 1.5 cm laterally. An independent
  adjacent-frame probe localizes these splits to short windows: the two
  replicated drawer cases pass at offsets {-2,0,+1} and {-1,0}, and the
  eggplant case at {0,+2}; all disappear by offset +4. This is real local
  arm-phase multimodality. In the executed-prefix reanalysis, 10/100 arm tests
  are positive and 6/100 have two resolved peaks (three drawer, two
  arm-dominated basket, and one gripper-correlated basket state), ruling out
  the explanation that all modes occur only in discarded future actions.
  A direct closed-loop fork screened 165 additional states, deep-tested 66,
  and found four independently confirmed prefix modes (Bonferroni p=0.026;
  two episodes per task). Three modes induce a measurable initial pose split;
  their label is decoded from pose at 65/72 during the forced four-step prefix
  (90.3%, permutation p=0.0002), but only 40/72 in the middle (55.6%, p=0.253)
  and 31/72 late (43.1%, p=0.808). Late centroid separation is 0.24--0.35 of
  radial within-mode RMS, and no case has a significant late path, endpoint
  metric, or success split. The fourth confirmed action mode produces no
  measurable arm actuation. Thus these modes causally change the immediate
  maneuver but do not behave as persistent route variables.
  Full protocol, raw outputs and figures are in
  `widowx_mm/widowx_experiment_report.md`.
- pi0.5 flow: the 14 states beyond the null's full range are all
  gripper-timing splits (share of the cluster-difference energy in the gripper
  channel 0.53-0.95, median 0.66). At the most separated state
  (separation 128), 12 of 16 samples issue the gripper-close command at chunk
  step 46 and 4 at step 47; the arm channels differ between the clusters by
  0.013 RMS, less than half of one control step of arm motion. The remaining
  3 states above q95 (separations 4.2-4.5) match the ~3 expected by chance.
- pi0.5 extreme-state search, 3000 screened states (K=8; 26.8% exceed the
  K=8 null q95), top 36 deep-verified with fresh K=32 samples: 20 states
  beyond the K=32 null q95 with separations up to 257 - every one
  gripper-dominated (gripper share 0.82-0.98; the two sub-4 exceptions match
  the chance-expected count). All three low-speed-alignment "mode
  candidates" across both searches (idxs 267061, 28723, 84346) differ
  between clusters only at the final chunk step - the gripper command
  issued at t=49 in one cluster and not yet in the other. Zero
  alternative-action splits in 3000 states per stack (6000 pooled): 95%
  upper bound on mode-state prevalence 0.1% per stack.
- GR1 flow: separation distribution coincides with the Gaussian null
  (median 2.85 vs 2.90); the dominant variance direction is unrelated to the
  temporal derivative of the mean chunk (|cos| median 0.07, equal to the random-direction
  baseline 1/sqrt(232) = 0.066; the probe's own printed 0.00 was a mask-shape fallback). The 5 states
  above q95 (4.2-5.7) are within the null's own maximum (6.7) and their
  cluster differences are spread across unrelated joint dimensions.
- GR1 extreme-state search, 3000 screened states (K=8; 10.1% exceed the K=8
  null q95 vs 5% chance), top 30 deep-verified with fresh K=32 samples: 15
  states beyond the K=32 null maximum (separations 4.7-8.7). Every one
  splits on one of two articulated-hand joint groups (dims 20-25 or 14-19).
  Although these are continuous-valued joint targets, the fingers move
  together between two narrow open/close configurations; the dominant
  channels are dims 24 or 18. Per-sample
  classification at those 15 states: 348/480 hold a constant level across
  the whole 8-step chunk, 118/480 contain exactly one coherent transition,
  14/480 (3%) flicker. This is the same event-timing phenomenon as pi0.5's
  gripper, censored by GR1's shorter chunk: the sampled hand-transition time
  straddles the 8-step window, so "flip at step k" appears as constant-old,
  constant-new, or one-flip depending on where k falls.
- Data side: at recurrent states of the push-T demonstrations
  (epsilon-ball recurrence, >=24 visits from >=5 episodes) the conditional
  action distribution is unimodal; the training signal does not contain
  chunk-level modes to learn.
- Cosmos3 is excluded as a witness: it generates future video jointly with
  actions and all its sample variance is seeded latent noise of that process
  (identical-seed control collapses spread 48x).

### Contact-state channel ablation

The held-out density audit evaluates the same 30 adversarial GR1 states three
ways, so only channel inclusion changes. PC1 is discovered on samples disjoint
from all evaluated draws; a one-Gaussian density, Student-t density, and
two-Gaussian mixture are compared by four-fold held-out likelihood. Gripper is
excluded for WidowX and pi0.5. `1G > 2G` counts states whose state-averaged
held-out score favors the single Gaussian; delta LL is the median 2G-minus-1G
score in nats per projected draw.

| policy / subspace | states | 1G > 2G | median delta LL | exact-normality BH rejects |
|---|---:|---:|---:|---:|
| WidowX arm, no detected phase split | 90 uniform | 53/90 | -0.018 | — |
| WidowX arm, detected local phase | 10 uniform | 0/10 | +0.170 | — |
| GR1 arm + waist | 30 adversarial | **29/30** | **-0.279** | **1/30** |
| GR1 articulated hands | 30 adversarial | 14/30 | +0.256 | 26/30 |
| GR1 all 29 joints | 30 adversarial | 14/30 | +0.246 | 26/30 |
| pi0.5 arm | 36 adversarial | **32/36** | **-0.266** | **3/36** |

The ablation localizes GR1's full-action non-Gaussianity to the articulated
hands: adding the hand channels changes both predictive mixture preference and
exact-normality rejection, whereas arm+waist remains single-component. Exact
normality is also rejected on the independent arm PC1 in 45/100 WidowX states,
mostly because of skew and the sparse phase cases; the supported wording is
therefore "single Gaussian component" or "single continuous basin," not
"mathematically Gaussian everywhere." Full protocol and raw results:
`gaussianity/action_basin_report.md`.

An exact-state WidowX intervention separately tests whether the continuous
mean is executable. At 40 on-policy states, the arm mean estimated from 64
same-observation draws is executed against eight fresh reference chunks while
holding the gripper sequence matched within each comparison. The mean is
outside the empirical 95% reference-path envelope in 0/40 states. Its median
path-centroid distance is 0.415 reference RMS in position and 0.382 in
orientation, substantially below the nearest sampled arm medoid (0.770 and
0.575; paired p=6.5e-8 and 2.0e-5). This supports mean-action validity, not
exact Gaussianity. Full protocol: `widowx_mm/widowx_arm_mean_report.md`.

Conclusion: the iterative heads do not show broad alternative-route sampling.
GR1 body motion is a single noisy component while its full-vector structure is
articulated-hand event timing; pi0.5's resolved structure is likewise
gripper-event timing. WidowX makes the literal "unimodal" claim untenable: it has frequent
gripper modes (22/100 on-policy states) and sparse but real arm-phase modes
(6/100 statistical arm splits; 3/100 clearly resolved and arm-dominated after
the geometric audit). Those arm modes are short-lived execute/defer or phase
shifts along one local maneuver, not different targets or routes. The
closed-loop intervention strengthens that interpretation for the tested
WidowX modes: a strong immediate mode signal disappears under replanning. The
reviewer-safe claim is therefore that learned stochasticity is dominated by
local event timing and maneuver phase, while persistent semantic-plan
multimodality is not observed in these probes.

## Figure: WidowX local modes (`widowx_mm/widowx_main_modes.png/.pdf`)

Three uniformly sampled on-policy states, each with 64 independent
confirmation chunks. Left: the exact observation; middle: the confirmation
samples projected onto PC1 fixed from a separate 32-sample discovery batch,
with the two fitted components; right: every sampled chunk and the two cluster
means in physical units. The rows show a drawer wrist execute/defer split, two
phase-shifted lateral pulses for eggplant transport, and gripper-close timing.
The first two rows are genuine arm distributions, but neither is a different
route. Script: `widowx_mm/make_widowx_mm_audit.py`.

## Figure: WidowX same-state closed-loop forks (`widowx_mm/widowx_branching_audit.png/.pdf`)

Modes are fit only over the four actions executed before replanning. Panels a
and b show all 24 exact-state branches for one independently confirmed drawer
and basket case (12 representative chunks per action component); thick lines
are component means, circles mark the end of the forced prefix, and crosses
mark the final state after 48 simulator steps. Panel c gives
leave-one-branch-out mode decoding from full end-effector pose with Wilson 95%
intervals. Panel d plots Euclidean mode-centroid distance divided by radial
within-mode RMS; panel e relates independent action-space evidence to late
spatial separation. Decoding is strong during intervention and at chance
thereafter, while every late between/within ratio is below one. The fourth
accepted case is omitted from the route panels because its translation-space
mode causes no measurable end-effector motion. Script:
`widowx_mm/make_widowx_branching_audit.py`.

## LaTeX

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{llccccc}
\toprule
policy & inference & SR & spread & \begin{tabular}{@{}c@{}}2-means sep.\\(null med.\ 2.9)\end{tabular} & \begin{tabular}{@{}c@{}}states $>$ null q95\\(chance $=5\%$)\end{tabular} & content of the variance \\
\midrule
push-T MIP & 1 pass & 0.947 & $3{\times}10^{-4}$\,px & --- & --- & none (float jitter) \\
push-T HT  & 1 pass & 0.927 & $3{\times}10^{-4}$\,px & --- & --- & none (float jitter) \\
GR1 HT     & 1 pass & \textbf{0.475} & 0 & --- & --- & none (deterministic) \\
GR1 flow   & 4 steps & 0.441 & 0.299 & 2.85 & 5/40, all $\le$ null max & single mode $+$ sampling noise \\
GR00T WidowX flow & 4 steps & 0.571 & 1.258 & 3.55 & 82/1000 arm; 184/1000 gripper & gripper timing $+$ sparse arm phase \\
$\pi_{0.5}$ HT   & 1 pass & \textbf{0.976} & 0 & --- & --- & none (deterministic) \\
$\pi_{0.5}$ flow & 10 steps & 0.969 & 0.924 & 3.23 & 17/60; 14 $>$ null max & gripper-transition timing \\
\bottomrule
\end{tabular}
\caption{Same-state sample probes at exact evaluation settings. For GR1 and
$\pi_{0.5}$, separation is the 2-means cluster separation on the first principal
component of the $K{=}16$ sampled chunks, compared to a Monte-Carlo Gaussian null (median
2.90, q95 3.97, max 7.90 over 2000 draws). WidowX uses a K=8 on-policy screen
with independent K=32/K=64 discovery/confirmation. Its unbiased 100-state
confirmation finds 22 gripper and 6 arm positives; only three are clearly
resolved arm-dominated profiles, all short-lived maneuver-phase splits. GR1
matches a single noisy mode and $\pi_{0.5}$ differs only in gripper-transition
timing (Fig.~\ref{fig:flow_unimodal}).}
\label{tab:multimodality}
\end{table}
```

## Figure: sampling spectrum (`fig_flow_spectrum.png/.pdf`)

Companion to `fig_flow_unimodal`: (a)/(b) per-state PC scale spectra of the K=16 sampled chunks
(GR1 flow: eff. rank median 3.1 of 232 dims, 5 PCs for 90% of variance; pi0.5 flow: 4.6 of 350,
9 PCs for 90%); HT's spectrum is identically zero (all K samples bitwise equal). (c) cumulative
variance vs leading PCs (median, IQR). (d) |cos(PC1, trajectory speed direction)| per state: pi0.5
median 0.64 (the dominant sampling axis is gripper-transition timing); GR1 median 0.07 = the
random-direction baseline 1/sqrt(232) = 0.066 (undirected noise). Built by
`scratchpad/make_spectrum_fig.py` from `gr1mm_flow.npz` / `pi05_flow_mm.npz`.

## Figure: sampled chunks as paths (`fig_chunk_paths.png/.pdf`)

Each panel: the K sampled action chunks of ONE training state drawn as integrated paths (pi0.5:
xyz translation; GR1: the three highest-motion ARM joints, hand channels excluded), projected onto
the plane of the mean path's top-2 PCs; thin = samples, black = mean (o start, square end). Three
states per stack: typical (spread at median), widest, and the most separated state found in the
3000-state search (K=32; pi0.5 colored by gripper-close step, GR1 by the level of the dominant hand
channel). pi0.5's paths form thin tubes at every state, including the sep-257 state where the two
"clusters" (close at step 46 vs 47) lie on the same path. GR1's arm paths are tubes as well; the only
fan-out seen in an earlier draft came from integrating the binary hand-command channels, i.e., the
discrete-level flips, not arm-trajectory alternatives. GR1 action layout (identified empirically
from per-dimension level counts and zero-velocity fractions): left arm 0-6, right arm 7-13, left hand
14-19, right hand 20-25, waist 26-28; dims 18 and 24 are the two-level channels. Tube width vs path
length: pi0.5 0.57/90.8, 0.87/69.2, 0.43/76.9; GR1 0.03/1.6, 0.04/1.6, 0.11/1.4 (offset-based, see EE-space note) (typical / widest /
most separated). Script `scratchpad/make_chunk_paths_fig.py`.

## Figure: sampled chunks as 3-D paths (`fig_chunk_paths_3d.png/.pdf`)

Same six states and same integrated paths as `fig_chunk_paths`, drawn in their own three coordinates
(pi0.5: x, y, z translation; GR1: the three highest-motion arm joints) with equal axis scales and
the camera tilted 25-30 degrees off the plane of the mean path, instead of the 2-D PCA projection.
"Typical" is now defined as the median-spread state among states below the null q95 (the previous
argmin tie between states 50 and 14 on pi0.5 included a sep-128 gripper-timing state). Reading is
unchanged: one bundle per state. At GR1's most separated state the two hand-level groups overlap at
the chunk start and drift apart by ~0.5 units (of a 4.8-unit path) by the last steps — a small
arm offset correlated with the hand level, not a second destination. Script
`scratchpad/make_chunk_paths_3d.py`.

## State identities behind `fig_chunk_paths` (`fig_chunk_states_pi05.png`, `fig_chunk_states_gr1.png`)

pi0.5 (LeRobot HuggingFaceVLA/libero, 273,465 frames, 10 fps; random probe = 60 frames evenly
spaced with np.linspace, search = 3000 frames evenly spaced):
- typical: dataset index 231749 = episode 1352, frame 19/96, "pick up the black bowl between the
  plate and the ramekin and place it on the plate" (LIBERO-Spatial), approach with open gripper.
- widest: index 125144 = episode 571, frame 46/155, "put the wine bottle on the rack" (LIBERO-Goal),
  bottle in hand, mid-transport.
- most separated (sep 257): index 47780 = episode 176, frame 176/275, "put the white mug on the
  plate and put the chocolate pudding to the right of the plate" (LIBERO-10), descending onto the
  mug; the chunk contains the grasp, hence the step-46/47 gripper-close split.

GR1 (GR00T seeded shard-mixture loader over the four RoboCasa GR1 tabletop datasets; the probe
consumed items sequentially and did not log identities; recovered by replaying the loader with
identity tagging, `groot/replay_gr1_states.py`): all three are PnPMilkToMicrowaveClose (the
loader's first shards came from that dataset) — typical = item 24 = episode 53 step 239/290;
widest = item 8 = episode 706 step 39/241; most separated (sep 8.7) = item 1974 = episode 801
step 1/267 (the very start of an episode).

## Figure: GR1 in end-effector space (`fig_chunk_paths_3d_ee.png/.pdf`; `fig_chunk_paths_3d.*` is the JOINT-space version, per the user's choice 2026-08-25)

GR1 row = right-wrist position (cm) from MuJoCo forward kinematics (robosuite 1.5.1 GR1 model,
`r_wrist_site`) of absolute joint targets = dataset state + un-normalized relative chunk, using the
processor's exact normalization parameters (dumped by `groot/dump_gr1_norm.py`; my inverse matches
`processor.unapply` to 5e-8 rad). Waist uses each path's own commanded waist. Dashed gray = the
demonstration's own chunk (dataset actions through the same FK). Script
`scratchpad/make_chunk_paths_3d_ee.py`.
Reading: one bundle per state (tube width mean 0.4-1.3 cm, max 1.6-3.7 cm on 10-15 cm paths).
OBSERVATION (cause unidentified): at all three states the model's chunk agrees with the demo at
step 0 (1.4-4.6 cm) but commands both elbows ~0.3 rad lower from step 1 onward (normalized
offset -0.6 vs the batch's own target; every other joint tracks the target, corr 0.96-0.99 for
wrists/shoulder), i.e., a 7-9 cm wrist jump between steps 0 and 1 and a 10-13 cm gap from the demo
by step 7. Verified to be genuine model output: fresh re-draws reproduce the probe samples (corr
0.97-1.00) and GR00T's own `unapply` gives the same absolute joints. This concerns the GR1 flow
checkpoint's chunk shape, not the multimodality question (all samples share it).
CAUSE IDENTIFIED (2026-08-28): the elbow offset is a probe-side normalization artifact. The
GR00T mixture dataset recomputes q01/q99 statistics from the datasets it is given and overrides the
processor's; the GR1 probe loader used a 4-dataset subset, so its STATE normalization differed from
the 24-dataset training run (right-shoulder joints shifted by up to 0.34 rad after un-normalization),
and every checkpoint (flow, MSE, L1, HT) then predicted the same constant elbow offset at chunk steps
>= 1 (forward loss on such batches 0.041 vs the logged 0.0083). With all 24 datasets in the loader
the offset disappears and the forward loss matches the training log. The GR1 chunk-path panels
(`fig_chunk_paths_3d.png`, `fig_gr1_ee_clarity.png`) should be regenerated with a full-mixture
loader before use; the multimodality reading (one bundle per state) is unaffected because all
draws share the offset. Details: `gr1_ht_mechanism.md` section 4.
Correction to the joint-space panels: GR00T relative actions are offsets from the current state,
not per-step deltas, so the earlier cumulative-sum integration was wrong; both joint-space figures
now plot the offset sequence directly (tube/fan reading unchanged; path lengths differ from before).
pi0.5 samples are in the policy's normalized action space; the demo overlay is therefore omitted
on the pi0.5 row.

EE-space two-bundle test on the GR1 panels (right-wrist paths, 8 steps x 3, vs Gaussian null at the same K):
- typical (K=16): 2-means sep 2.46 (null median 2.87, q95 3.94); hand-level groups intermixed
  (between-mean 0.6 cm vs within-rms 1.5 cm; same-color nearest neighbour 0.31, chance 0.5).
- widest (K=16): sep 4.04 (null q95 4.00, max 6.49 - chance level); groups intermixed (1.5 vs 2.6 cm;
  same-color NN 0.38); per-step gap between color means grows 0.2 -> 0.85 cm.
- most separated (K=32): sep 2.79 = null median (no split on the dominant axis); the hand-level
  groups are partially separated: between-mean 4.2 cm vs within-rms 3.5 cm (ratio 1.2), same-color
  NN 0.84, per-step gap 1.1 -> 2.0 cm. A correlated shift of the arm with the hand level, smaller
  than the within-group scatter - overlapping sub-clouds on one route, not two trajectories.

## Figure: GR1 plain view (`fig_gr1_ee_clarity.png/.pdf`, script `scratchpad/make_gr1_ee_clarity.py`)

Made because the 3-D GR1 panels read as "large spread" (their axis boxes are only 10-15 cm wide).
Top row: right-wrist paths, top view, fixed 16 cm window for all three states (x = wrist now,
o/square = mean chunk start/end, dashed = demonstration chunk). Bottom row: the K end-of-chunk
positions projected on the end-point cloud's main axis, with a single-Gaussian fit and the 2-means
split score. Scatter around the mean path: 0.4 / 0.9 / 1.2 cm mean (max 1.6 / 1.8 / 3.7 cm);
end-point sd 0.6 / 0.9 / 1.6 cm; split scores 2.53 / 3.46 / 2.83 vs pure-noise median 2.9. In the
most-separated state the hand-high (blue) and hand-low (orange) chunks sort to opposite sides of
the axis but under one hump (group means 1-2 cm apart, each group ~1.5 cm wide).

CORRECTION (2026-08-25, coloring): earlier versions of the GR1 panels colored samples "hand high /
hand low" by thresholding the dominant hand channel at its mean, which splits even states where
all samples agree (typical state: all 16 samples at 0.63-0.65 on channel 18, gap 0.008). All four
figures now use `hand_colors`: one color when samples agree (gap < 0.3 on channels whose two levels
are ~1.2 apart); color by hand-closing step when the hand switches inside the chunk (widest state:
right-hand channel 24 goes -0.61 -> +0.78, closing on the milk carton; samples differ in the closing
step); two colors only when samples hold two different levels for the whole chunk (most separated
state: LEFT-hand channel 19, +0.3 in 18 samples vs -0.8 in 14, at the first frame of the episode -
the idle hand in this task). Start points: GR1 paths are commanded joint targets, so (a) each
sample's step-0 target differs slightly (0.3-1.0 cm), and (b) targets lead the measured wrist by
1.7-6.2 cm (the demonstration's own step-0 target is 2-8 cm from the measured wrist: controller
tracking lag in the RoboCasa GR1 data). pi0.5 paths are cumulative displacements from a common origin.

Start-point offset explained by the data (`fig_gr1_command_lag.png`): in the RoboCasa GR1
demonstrations the commanded joint target a(t) matches the measured pose q(t+k) best at k = 4
steps (0.2 s at 20 Hz; mean gap 0.013-0.028 rad vs 0.045-0.055 rad at k = 0), so commands lead the
measured pose. At the probe frames the demonstration's own step-0 target is 7.8 / 2.0 / 2.8 cm
from the measured wrist and the wrist then moves toward it (7.8 -> 2.8 cm within 9 steps). The
policy's step-0 targets lead the measured wrist by the same kind of margin (1.7-6.2 cm).

## Figure: pi0.5 boundary test (`fig_pi05_paths_boundary.png/.pdf`, script `scratchpad/make_pi05_boundary_fig.py`)

Top row = the three pi0.5 3-D path panels. Bottom row = the most separated state (episode 176,
frame 176, dataset index 47780) re-sampled with the observation shifted by -10,-5,-3,-2,-1,0,+1,+2,
+3,+5,+10 frames (K=32 each; probe `pi05/probe_pi05_offsets.py`, data `pi05_offsets.npz`).
Gripper-close step per sample: at every frame one majority step (= demo's close step + 1) with
20-35% of samples one step early/late; the majority step tracks the frame shift exactly (46 at
0, 44 at +2, 41 at +5, 37 at +10; at -5/-10 the close falls beyond the 50-step chunk, all 32 agree).
2-means split score (all channels): 2.8, 3.1, 6.7, 0, 13.7, 25.5, 0, 75.3, 0, 83.8, 0 - it flips
on/off from frame to frame because it is the timing jitter seen through the test's minority-size
rule (0 = minority < 20% of samples), not a property of the state; arm-only score 2.5-3.8 at every
frame (null q95 3.5, max 4.2); arm tube width 0.32-0.45 throughout. There is no boundary: the
"two clouds" exist at no frame and the ±1-step gripper jitter exists at every frame.

Update (same figure, final bottom row): the derived-statistics panels were replaced by the raw
signal. (4) gripper channel of the 32 chunks at the most separated state vs chunk step: identical
open->closed step functions that differ only in the flip step (45: 2, 46: 22, 47: 8; demonstration
45). (5) arm channels (dx, dy, dz, each centred) of the same 32 chunks in the same colors: the
early-flip and late-flip groups differ by 0.020 RMS on the arm (within-group scatter 0.015) - the
arm does not carry the split. (6) the gripper channel at frames -1/0/+1/+2: the flip moves one step
per frame, the +-1-step jitter is present at every frame. The demonstration overlay is omitted on the
arm channels because the probe samples are in the policy's normalized action space.

Coloring convention change (2026-08-25, all pi0.5 "most separated" panels in fig_chunk_paths,
fig_chunk_paths_3d, fig_pi05_paths_boundary): paths are colored by the gripper state at each step -
orange while open, blue after the close command, with a diamond at the close step - instead of one
color per sample. The two sample groups therefore show as the same path with the diamond one step
apart (step 46 in 22 samples, 47 in 10).

## Long-tail videos: which state-action pairs HT down-weights (`videos/ht_longtail_ep*.mp4`, frame `fig_ht_longtail_frame.png`)

pi0.5 / LIBERO, HT checkpoint run_ht 30k (legacy form nu'=2, i.e. nu_eff = 700) and flow checkpoint
libero_ft. Probe `pi05/probe_ht_video.py`: 60 episodes scanned at stride 5 with the HT policy
(S = chunk residual, sigma from the head, gate w = (nu+d)/(nu+S/sigma^2), "gated" = S/sigma^2 > nu);
then every frame of 4 episodes (two most gated, median, least) with HT and flow (8 (t,noise) draws
of the flow loss, flow's per-sample gradient is proportional to it). Renderer `videos/make_ht_video.py`.
- Gating is rare and concentrated: median episode 0% of frames past the knee, q90 7%, max 40%.
- ep 1634 (bowl on stove -> plate, 40% gated): frames 69-118 = exactly the 50-frame windows that
  contain one demonstrator GRIPPER FLICKER (release at chunk step ~13, re-close spike at ~17-19,
  release again). Both HT and flow predict one clean release; the residual is 77% in the gripper
  channel; S/sigma^2 up to 819 vs nu 700, w down to 0.28. A single non-reproducible event in the
  demonstration gates every chunk that straddles it.
- ep 1663 (median): the one gated frame is the same phenomenon (gripper share 0.85).
- ep 1491 (13% gated): frames 84-96 at the END of the episode with a SMALL residual (rms 0.044 vs
  0.152 elsewhere) but a collapsed sigma (0.027): the head is over-confident on the idle terminal
  frames, so the gate fires on sigma, not on a large error.
- Flow's smallest gradients (bottom 10% of per-sample loss) sit almost entirely on the idle
  end-of-episode frames (ep 1634: 110-124; ep 1491: 85-99; ep 1663: 114-137; ep 487: 78-87) plus a
  few early static frames - where the action is nearly constant. Overlap with HT-gated frames is
  small except on the terminal idle frames (ep 1491: 7/13).
Reading: on pi0.5 the samples HT treats as long-tail are demonstrator gripper flickers (a discrete
event with no visual cause the policy can learn) and over-confident terminal frames; flow's
gradient is smallest where nothing happens. Neither set is a second behavior mode.
