# Benchmark table — 3-seed averages (fixed eval protocol)

All HT cells: mean over training seeds 1–3 of best-checkpoint / last-5, with the per-seed
last-5 range beneath. Every run: 22 episodes/eval, fixed initial conditions
(EVAL_SEED_BASE=4300000), slot recipes from BEST_SETTINGS.md. State AND image are FINAL
(last image runs completed 2026-08-13).

## State

| method | lift-ph | lift-mh | can-ph | can-mh | square-ph | square-mh | transport-ph | transport-mh | toolhang-ph | push-T |
|---|---|---|---|---|---|---|---|---|---|---|
| **HT (Chi-Transformer)** | 1.00/0.93<br><sub>0.89–0.97</sub> | 1.00/0.96<br><sub>0.94–1.00</sub> | 1.00/0.94<br><sub>0.87–1.00</sub> | 1.00/0.98<br><sub>0.94–1.00</sub> | 0.97/0.88<br><sub>0.86–0.92</sub> | 0.88/0.74<br><sub>0.68–0.84</sub> | 0.70/0.48<br><sub>0.41–0.56</sub> | 0.36/0.14<br><sub>0.09–0.18</sub> | 0.97/0.80<br><sub>0.74–0.85</sub> | 1.00/0.98<br><sub>0.96–0.99</sub> |
| **HT (Chi-UNet)** | 1.00/1.00<br><sub>1.00–1.00</sub> | 1.00/1.00<br><sub>1.00–1.00</sub> | 1.00/1.00<br><sub>1.00–1.00</sub> | 1.00/0.98<br><sub>0.95–1.00</sub> | 1.00/0.97<br><sub>0.95–0.99</sub> | 0.98/0.88<br><sub>0.85–0.95</sub> | 0.91/0.78<br><sub>0.77–0.79</sub> | 0.79/0.45<br><sub>0.36–0.50</sub> | 0.88/0.65<br><sub>0.62–0.69</sub> | 0.96/0.93<br><sub>0.93–0.94</sub> |

Transport-mh state (Chi-UNet) cell: tmf slot (545k budget, obs_steps=8, ema_power 0.75/max 0.999,
LR warmup 500) — supersedes tmc; best 0.79 (seeds 0.77/0.68/0.91), last-5 0.45 ties DP-C's 0.46
endpoint; best-5-window ES per seed 0.664/0.573/0.664 (mean 0.634).

## Image

| method | lift-ph | lift-mh | can-ph | can-mh | square-ph | square-mh | transport-ph | transport-mh | toolhang-ph | push-T |
|---|---|---|---|---|---|---|---|---|---|---|
| **HT (Chi-Transformer)** | 1.00/0.99<br><sub>0.96–1.00</sub> | 1.00/0.98<br><sub>0.97–0.99</sub> | 1.00/1.00<br><sub>0.99–1.00</sub> | 1.00/0.88<br><sub>0.84–0.95</sub> | 1.00/0.94<br><sub>0.90–0.96</sub> | 0.97/0.83<br><sub>0.82–0.85</sub> | 0.97/0.89<br><sub>0.86–0.90</sub> | 0.79/0.56<br><sub>0.50–0.63</sub> | 0.92/0.75<br><sub>0.67–0.79</sub> | 0.93/0.89<br><sub>0.85–0.91</sub> |
| **HT (Chi-UNet)** | 1.00/0.94<br><sub>0.91–0.95</sub> | 1.00/0.92<br><sub>0.90–0.95</sub> | 1.00/0.91<br><sub>0.89–0.94</sub> | 1.00/0.89<br><sub>0.87–0.92</sub> | 1.00/0.86<br><sub>0.77–0.95</sub> | 0.97/0.80<br><sub>0.69–0.85</sub> | 1.00/0.91<br><sub>0.89–0.93</sub> | 0.83/0.52<br><sub>0.50–0.56</sub> | 0.76/0.57<br><sub>0.50–0.65</sub> | 0.90/0.84<br><sub>0.80–0.89</sub> |

Transport-mh image (both nets, FINAL): best/ES beat DP-C (0.79–0.83 vs 0.69) but the last-5
endpoint (0.52–0.56) trails DP-C's 0.69 — same late-window slide as the state slot; report both
stats, slide disclosed.

Transport-ph image (FINAL 2026-08-13): ct 3-seed best 0.97 / last-5 0.886 / ES 0.935;
cu 3-seed best 1.000 / last-5 0.913 / ES 0.973 (per-seed ES 0.987/0.953/0.980). cu is on-par
with DP-C's published 1.00/0.93 under their own convention and above it under ES.
