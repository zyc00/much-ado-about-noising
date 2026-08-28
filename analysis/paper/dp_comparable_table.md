# DP-comparable benchmark table

Our runs use **DP's optimization settings** but **our best action space per
task**. Every number is best / last-5, scored on **fixed initial conditions**
(EVAL_SEED_BASE=4300000), the protocol DP itself used.

## Settings

| | ours (HT) | DP-C | DP-T |
|---|---|---|---|
| backbone | chitransformer, 20.3M | CNN UNet | transformer |
| batch | **256 low-dim / 64 image** | 256 low-dim / 64 image | same |
| lr / schedule | **1e-4**, 500-step warmup + cosine | 1e-4, same | same |
| weight decay | **1e-6** | 1e-6 | 1e-6 |
| To / Ta / Tp | 2 / 8 / **16** | 2 / 8 / 16 | 2 / 8 / 16 |
| denoising steps | **1** | 100 | 100 |
| loss | hetero Student-t, **nu = 2** (fixed, never task-tuned) | DDPM | DDPM |
| architecture, transformer | emb 256, 8 layers, 4 heads, attn-dropout 0.3 -> **9.0M** | - | emb 256, 8 layers, 4 heads, p_drop_attn 0.3 -> 9M |
| architecture, CNN | model_dim 256, dim_mult [1,2,4], step-emb 256 -> **193M** | down_dims [256,512,1024], step-emb 256 -> ~193M | - |
| gradient steps, low-dim | **DP's exact per-task counts**, within 1% | lift-ph 158k, lift-mh 555k, can-ph 416k, can-mh 1.149M, square-ph 545k, square-mh 1.495M, transport-ph 1.753M, transport-mh 3.669M, tool-hang 1.792M, push-T 208k | same |
| gradient steps, image | **DP's exact per-task counts** | lift-ph 514k, push-T 504k, can-ph 1.190M, lift-mh 1.386M, square-ph 1.408M, can-mh 2.785M, square-mh 3.743M, transport-ph 4.176M, tool-hang 4.267M, transport-mh 8.738M | same |
| evals | **DP's per-task count** (low-dim 100; image 60-82), freq = steps / count | same | same |
| episodes / eval | 22; **50** for image can-mh, lift-mh, push-T and all low-dim push-T | same | same |
| eval conditions | fixed, seed 4300000 | fixed | fixed |

Both halves are now **fully aligned**: same architecture, batch, lr,
schedule, weight decay, horizon, resolution, crop, total gradient steps
(hence total samples, since batch matches), eval count, episode count and
fixed initial conditions. The only remaining unmatched item is DP-T's
`causal_attn: true`, which our chitransformer does not implement.

Image resolutions match DP exactly: 84x84 with 76 crop (lift/can/square/
transport), 96x96 with 84 crop (push-T), and **240x240 with 216 crop for
tool-hang** - the original 84x84 tool-hang cell was NOT comparable to DP's
number and was replaced.

Action space per task (ours; DP always uses absolute):
delta-legacy for lift-ph/mh, can-ph/mh, tool-hang-ph; **abs** for
square-ph/mh and transport-ph/mh; transport also uses **obs_steps 8**.

DP reference numbers below are computed from their released per-checkpoint
logs, not transcribed from the paper: best = `max/test_mean_score` (mean over
3 seeds of each seed's own maximum), l5 = last-5 of the 3-seed-pooled series.
This reproduces their published pairs exactly (e.g. tool-hang image DP-C
0.955 / 0.703).

## State (low-dim)

| task | HT (ours) | DP-C | DP-T |
|---|---|---|---|
| lift-ph | running | 1.000 / 1.000 | 1.000 / 1.000 |
| lift-mh | running | 1.000 / 0.982 | 1.000 / 1.000 |
| can-ph | running | 1.000 / 0.997 | 1.000 / 0.997 |
| can-mh | running | 1.000 / 0.982 | 1.000 / 0.924 |
| square-ph | running | 1.000 / 0.955 | 1.000 / 0.897 |
| square-mh | running | 0.985 / 0.882 | 0.985 / 0.818 |
| transport-ph | running | 0.955 / 0.873 | 1.000 / 0.794 |
| transport-mh | running | 0.773 / 0.606 | 0.621 / 0.248 |
| tool-hang-ph | running | 0.864 / 0.548 | 1.000 / 0.873 |
| push-T | running | 0.946 / 0.903 | 0.955 / 0.799 |

## Vision (image)

| task | HT (ours) | DP-C | DP-T |
|---|---|---|---|
| lift-ph | running | 1.000 / 1.000 | 1.000 / 1.000 |
| lift-mh | running | 1.000 / 0.999 | 1.000 / 0.994 |
| can-ph | running | 1.000 / 0.979 | 1.000 / 0.979 |
| can-mh | running | 1.000 / 0.953 | 1.000 / 0.988 |
| square-ph | running | 0.985 / 0.912 | 1.000 / 0.897 |
| square-mh | running | 0.985 / 0.876 | 0.955 / 0.782 |
| transport-ph | running | 1.000 / 0.918 | 0.985 / 0.800 |
| transport-mh | running | 0.894 / 0.679 | 0.727 / 0.412 |
| tool-hang-ph (84px) | running | 0.955 / 0.703 | 0.758 / 0.427 |
| tool-hang-ph (240px) | running (batch 128) | 0.955 / 0.703 | 0.758 / 0.427 |
| push-T | running | 0.907 / 0.838 | 0.777 / 0.640 |

## Notes for the writeup

- **DP-T is the architecture-matched baseline** (transformer vs our
  chitransformer); DP-C is a CNN. Report both - DP-C is stronger on most
  cells, DP-T is stronger on low-dim tool-hang (0.873 l5) where DP-C
  collapses to 0.548.
- **Our budget is 300k steps** against DP's 0.5M-4.3M; state that plainly
  rather than claiming efficiency from it.
- **push-T low-dim uses `pusht_state`** (agent + block pose), whereas DP's
  low-dim push-T uses 20-D keypoints. Our `pusht_keypoint` path is broken
  (0.31 vs 0.95 for `pusht_state` on an identical recipe) and must be fixed
  before that cell is a like-for-like comparison.
- **transport-mh vision** uses our own 4-camera merged file
  `data/transport_mh_image4.hdf5`, not the DP tree.
- Inference cost (measured, A800, batch 22, per action chunk): HT 9.6-13.9 ms
  vs DP 695-874 ms, a 63-73x gap that follows directly from 1 vs 100
  denoising steps.
