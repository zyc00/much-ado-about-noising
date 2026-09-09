# HT sigma: action amplitude, hand transitions, and contact-related interpretation

## Question and result

High sigma is better understood as the model's shared **action-chunk residual scale**, not a movement-size meter or a physical precision tolerance. On these checkpoints it is associated with actual fitting error, and hand/gripper transitions are an important contributor independent of arm amplitude. The contribution is strongest in GR1 and is not the entire explanation in Bridge or Fractal.

This study does not establish that contact causes irreducible noise. There are no force/contact labels, repeated demonstrations at identical states, or independent epistemic/aleatoric decomposition. Contact, alignment, and occlusion are visual interpretations of selected frames, not automatically verified phase labels.

## Frozen probe and provenance

- GR1: `ft_gr1c2/checkpoint-60000`, nu=928, all 24 task datasets; 144 episodes, 2,880 states.
- Bridge: `ft_wxnu224/checkpoint-20000`, nu=224; 24 most frequent eligible nonempty instruction groups, 144 episodes, 2,867 states.
- Fractal: `ft_fr_nu224/checkpoint-20000`, nu=224; same frequency rule, 144 episodes, 2,862 states.
- Six random episodes per task, fixed seed 20260908; up to twenty evenly spaced valid eight-step chunks per episode. No padding. 8,609 states total.
- Original checkpoint processor/normalization, eval mode, bf16 autocast, no fitting or parameter updates. Forward zero-input predictions match deployment predictions exactly on the first batch of each dataset (max difference 0).
- All are training-demonstration diagnostics, not policy-held-out validation. This probe uses a different sample and model objective from the previous general-MSE correlation figure.
- Cluster raw data: `/mnt/pfs/yuchen/ht_sigma_context_20260908/{gr1,bridge,fractal}/probe.npz` and `thumbnails.npz`. Protocol JSON contains task names, dataset paths, episode IDs, seed, checkpoint, action channel ordering, and inference check.
- Local analysis copies: `{dataset}/analysis.json`, `physical_analysis.json`, `rollout_analysis.json`. Image audits: `{dataset}/visual_pairs.jpg`. A failed local GR1 raw download is explicitly named `probe.npz.partial`; use the complete cluster source, not this partial file.
- Scripts: `scripts/probe_ht_sigma_context.py`, `scripts/analyze_ht_sigma_context.py`, `scripts/enrich_ht_sigma_context.py`; cluster job `scripts/cluster/ht_sigma_context.yaml` completed on one GPU. No training jobs interrupted.

## Main measurements

All correlations below are pooled Spearman coefficients in the checkpoint's normalized action space. Arm excludes hands/gripper; GR1 arm means its 14 arm joints. High/low sigma groups are defined separately within each task using the top/bottom quintiles.

| Dataset | sigma vs arm action RMS | sigma vs total residual RMS | Hand/gripper share of error energy, high sigma | Same share, low sigma |
|---|---:|---:|---:|---:|
| GR1 | 0.273 | 0.908 | 75.5% | 2.9% |
| Bridge | 0.324 | 0.734 | 30.1% | 7.9% |
| Fractal | 0.121 | 0.766 | 55.8% | 14.1% |

Within-task rank correlations with total residual RMS are respectively 0.885, 0.709, 0.729. These are not merely a difference between task averages.

Normalized action RMS is not physical motion magnitude: normalization offsets and per-channel scales matter. Physical checks invert the saved normalization (clipped labels remain clipped). GR1 uses relative arm joint offsets in radians; Bridge/Fractal use translational action components in meters, without mixing rotations into a physical norm. Sigma correlations with those physical amplitude measures are 0.313, 0.192, 0.222 respectively. They remain modest.

## Hand-change comparison at similar arm amplitude

Hand-change proxy: GR1 maximum in-chunk hand-joint command range >0.2 rad; Bridge/Fractal in-chunk gripper command range >0.5. This is a command transition, **not ground-truth contact**. In particular, maintained contact with a steady hand is not an event by this definition.

Partition each task into four physical arm-amplitude bins. Retain bins with at least three event and three non-event states. In each bin compute median sigma(event)/median sigma(non-event), then summarize the bin ratios by their median:

| Dataset | Median sigma ratio | Eligible bins / tasks |
|---|---:|---:|
| GR1 | 4.98x | 58 / 23 |
| Bridge | 1.26x | 76 / 20 |
| Fractal | 1.35x | 94 / 24 |

These are coarse amplitude-controlled associations, not exact matching or a causal intervention. Bins do not control visual occlusion, progress, or all aspects of motion.

GR1 is especially clear: event and non-event physical arm-amplitude medians are 0.09499 and 0.09514 rad RMS, while sigma medians are 0.11747 and 0.02235. Hand channels contribute 86.5% of event-state error energy. The event definition finds 300/2,880 states; 49.8% of top-sigma states but none of bottom-sigma states contain this transition.

Bridge transitions appear in 49.0% of high-sigma versus 12.2% of low-sigma states; Fractal in 66.6% versus 10.8%. In Bridge, arm error still accounts for 69.9% of high-sigma error energy, so a hand-only explanation is incorrect.

## Rollout and image checks

Re-read all saved episodes (including failures) of the five GR1 sigma-video tasks. Excluding each episode's first call, sigma versus commanded hand-range correlations are 0.752–0.923; sigma versus physical arm-offset magnitude ranges from -0.217 to 0.544. These are predictions versus predictions, not error measurements, but agree with the qualitative video observation.

The selected high-sigma images include reaching for a drawer handle, grasping/releasing near bowls, manipulating cabinet/microwave doors, and initial approach/posture states. Low-sigma controls can also be near objects or maintaining a grasp. Therefore proximity/contact alone is not a binary explanation. `visual_pairs.jpg` uses each of the first eight preselected tasks' peak-sigma state and the closest-amplitude state among its low-sigma half; these are illustrative selected examples, not a prevalence estimate. Some tasks lack a close-amplitude control; numeric amplitudes are printed rather than concealing that mismatch.

GR1 also has elevated initial sigma: initial-frame median 0.0851 versus 0.0230 later; 128/144 initial frames fall in their task's upper sigma quintile. Bridge and Fractal do not show that same initial-frame pattern. Do not call sigma a contact detector.

## How to interpret grasp/alignment peaks

The checkpoint loss uses one scalar sigma for the full valid chunk, including hands/gripper (GR1 8x29, Bridge/Fractal 8x7). Its sigma-dependent terms are

`L = (nu+d)/2 * log(1 + S/(nu*sigma^2)) + d*log(sigma)`, with `S = sum_j r_j^2`.

For a fixed nonzero residual vector and a free per-example sigma, the stationary point is `sigma^2=S/d`. A trained shared network does not fit that optimum independently per example, but this calculation identifies what the objective rewards: predicting residual scale, not physical tolerance. Thus hand errors can raise the scale assigned to arm coordinates as well.

Alignment can require tiny physical errors for success while the action target is difficult to predict: which correction direction, whether/when to close or release, and the next chunk's trajectory. The probes support the role of fitting error and hand transitions. They do **not** establish that the remaining error is irreducible, or that timing/occlusion rather than model bias is its cause. Heteroscedastic likelihoods can absorb mean-model error into variance; see Seitzer et al., ICLR 2022, https://arxiv.org/abs/2203.09168 .

Important correction to previous video reading: arm/hand/waist means of the sigma-decoder outputs are not separately supervised variances. Only their masked aggregate enters this loss. Their similar curves cannot identify which group's errors caused the scalar to rise; use actual residual-energy decomposition above instead.

Bottom line: the observations support state-dependent **difficulty of fitting the demonstrated action chunk**, with substantial hand-transition contributions, rather than “contact allows less precise movement” or “contact must involve larger arm motion.”
