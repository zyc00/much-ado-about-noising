# Efficiency comparison: HT vs Diffusion Policy vs MIP (3 vision tasks)

All numbers measured, not quoted. DP columns come from their released
per-checkpoint logs (`diffusion-policy.cs.columbia.edu/data/experiments/
image/<task>/diffusion_policy_cnn/train_0/logs.json.txt`); ours come from
`logs/t12_*/metrics.jsonl`. Ours: chitransformer + HT (heteroscedastic
Student-t, nu=2), DP-harness recipe, 300k steps, 3 seeds unless noted.

## 1. Inference cost per action chunk (measured, A800, batch 22)

The vision encoder runs once per chunk; the action head repeats once per
denoising/flow step. DP-C uses 100 DDPM steps, MIP 2, HT 1.

| task | encoder | head | HT (1 step) | MIP (2 steps) | DP-C (100 steps) | HT speedup |
|---|---|---|---|---|---|---|
| can-mh | 5.19 ms | 8.69 ms | **13.9 ms** | 22.6 ms | 874 ms | **63x** |
| square-mh | 3.86 | 8.59 | **12.5** | 21.1 | 863 | **69x** |
| push-T | 2.65 | 6.93 | **9.6** | 16.5 | 695 | **73x** |

This is the one axis where the advantage is unambiguous and needs no
caveats. At 10 Hz control, DP-C's 0.7-0.9 s per chunk exceeds the control
period; HT's 10-14 ms does not.

## 2. Final performance (task success)

| task | DP-C best / l5 | HT best / l5 |
|---|---|---|
| can-mh | 1.00 / 0.94 | 1.00 / **0.96** |
| square-mh | 1.00 / **0.85** | 0.91 / 0.77 |
| push-T | 0.90 / 0.82 | 0.93 / **0.91** |

## 3. Training cost to reach a fixed quality level

Two axes disagree, and the disagreement is a batch-size artifact: ours ran
at the repo default 1024, DP at 64.

| task | level | DP steps / samples | HT steps / samples |
|---|---|---|---|
| can-mh | >=0.9 | 47k / 3.0M | 4k / 5.1M |
| square-mh | >=0.8 | 62k / 3.9M | 59k / 61.4M |
| push-T | >=0.8 | 34k / 2.2M | 9k / 10.2M |
| push-T | >=0.9 | 51k / 3.2M | 49k / 51.2M |

Total training: DP 2.88M / 3.74M / 0.50M steps (184M / 240M / 32M samples);
ours 300k steps (307M samples at bs 1024).

**Honest reading:** HT reaches a given quality in fewer gradient steps
(1-12x), but not in fewer samples, because each of our steps sees 16x more
data. On a FLOPs proxy (samples) DP is ahead on 2 of 3 tasks. A
batch-matched HT run is required before any training-efficiency claim.
Evidence that batch is the cause: on low-dim tool-hang, HT crosses 0.54 at
20.5M samples with bs 1024 but at 2.6M samples with bs 128 - an 8x
sample-efficiency gain from batch size alone.

## 4. Resolution caveat on the DP columns

DP's eval grid differs per task and bounds how early their convergence can
be observed: can-mh every 46k steps, square-mh every 60k, push-T every 8k.
Only **push-T** is fine enough to resolve their learning curve; for the
other two, DP's threshold crossings are upper bounds (they may have
converged anywhere inside the preceding interval), so those two rows cannot
support a convergence claim in either direction.

## 4b. State at the time nodes .13/.17 were freed (2026-08-08)

Killed mid-run to release GPUs; metrics.jsonl and 2 checkpoints per run are
retained on PFS, so each can be resumed or re-run. Partial numbers:

| run | progress | best / l5 |
|---|---|---|
| MIP loss, MIP harness, s1/s2/s3 | 199k/300k | 0.90/0.82, 0.90/0.79, 0.88/0.81 |
| MIP loss, DP harness, s1/s2/s3 | 189-194k/300k | 0.91/0.75, 0.95/0.85, 0.95/0.82 |
| can-mh aligned (50-ep evals), s1/s2/s3 | 164k/300k | 0.94/0.89, 0.96/0.92, 0.96/0.91 |
| Barron hetero-diag, DP harness | 299k **complete** | 0.95/0.83 |
| MIP vision (square-mh, push-T) | ~18% | see logs |

**HT reference on the same task/harness (complete, 3 seeds):** 0.95/0.85,
0.95/0.85, 1.00/0.83. MIP's DP-harness arms at 63% were tracking
0.95/0.82-0.85 - i.e. converging toward HT's endpoint, not clearly below
it. The endpoint question is therefore still open; only the steps-to-quality
result (HT ahead at every matched step through 150k) is established.

**Do NOT quote MIP's published tool-hang number as the baseline.** Our July
replicas measured 88/72.4 and 88/76.8 in this harness against a published
85/64 - the table understates MIP by 8-13 points of last-5, and using it
would inflate any HT-vs-MIP gap by that much. Training is also
GPU-nondeterministic here: identical config and seed have spread 13 points
(82 vs 69 at 150k), so single runs are weak anchors and 3 seeds are the
minimum.

## 5. Pending (launched 2026-08-08)

- `htb64-{can,sq,pt}`: HT at DP's batch size 64, same recipe otherwise.
  Fills the batch-matched row - the only version of section 3 that is fair.
- `mipvis-{can,sq,pt}-{a,b}`: MIP loss on the same 3 tasks, 2 EMA recipes,
  matching the HT arms. MIP has never been trained on vision in this repo;
  every prior MIP number came from evaluating checkpoints post hoc.
- `mipcurve-*`, `flowcurve-*`, `longrun-{flow,mip}`: tool-hang state,
  identical grid/backbone/seeds, plus 1.2M-step arms to separate "slower to
  converge" from "converges lower".
