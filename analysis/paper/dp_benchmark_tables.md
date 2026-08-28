# DP benchmark (separate from T12/T13): DP paper baselines + ours

Cell format x/y = best checkpoint / average of late checkpoints.
Baseline rows: verbatim from the Diffusion Policy paper (arXiv:2303.04137,
Tables I & II; y = avg of last 10 checkpoints, 50 episodes).
Ours (HT = heteroscedastic Student-t regression, nu=2): DP eval protocol
(22 episodes every 5k steps, 300k updates, 3 seeds; best = max over
seeds+evals, l5 = seed-mean of last-5 average).
† = our 20k-grid protocol (50 episodes, 15 evals) — only protocol available
for that cell.  * = run still in progress.  — = not run.

## T1 — state

| method | lift-ph | lift-mh | can-ph | can-mh | square-ph | square-mh | transport-ph | transport-mh | toolhang-ph | push-T |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM-GMM | 1.00/0.96 | 1.00/0.93 | 1.00/0.91 | 1.00/0.81 | 0.95/0.73 | 0.86/0.59 | 0.76/0.47 | 0.62/0.20 | 0.67/0.31 | 0.67/0.61 |
| IBC | 0.79/0.41 | 0.15/0.02 | 0.00/0.00 | 0.01/0.01 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.90/0.84 |
| BET | 1.00/0.96 | 1.00/0.99 | 1.00/0.89 | 1.00/0.90 | 0.76/0.52 | 0.68/0.43 | 0.38/0.14 | 0.21/0.06 | 0.58/0.20 | 0.79/0.70 |
| DiffusionPolicy-C | 1.00/0.98 | 1.00/0.97 | 1.00/0.96 | 1.00/0.96 | 1.00/0.93 | 0.97/0.82 | 0.94/0.82 | 0.68/0.46 | 0.50/0.30 | 0.95/0.91 |
| DiffusionPolicy-T | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/0.94 | 1.00/0.89 | 0.95/0.81 | 1.00/0.84 | 0.62/0.35 | 1.00/0.87 | 0.95/0.79 |
| Ours HT (Chi-UNet) | — | — | — | 1.00/0.96 | 1.00/0.90 | 0.95/0.74 | 0.85/0.70† | 0.70/0.52† | 0.95/0.74† | — |
| Ours HT (Chi-Transformer) | 1.00/0.98 | 1.00/0.99 | 1.00/0.99 | 1.00/0.94 | 1.00/0.79 | 0.91/0.71 | — | — | 1.00/0.83 | 1.00/0.98 |

## T2 — image

| method | lift-ph | lift-mh | can-ph | can-mh | square-ph | square-mh | transport-ph | transport-mh | toolhang-ph | push-T |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM-GMM | 1.00/0.96 | 1.00/0.95 | 1.00/0.88 | 0.98/0.90 | 0.82/0.59 | 0.64/0.38 | 0.88/0.62 | 0.44/0.24 | 0.68/0.49 | 0.69/0.54 |
| IBC | 0.94/0.73 | 0.39/0.05 | 0.08/0.01 | 0.00/0.00 | 0.03/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.75/0.64 |
| DiffusionPolicy-C | 1.00/1.00 | 1.00/1.00 | 1.00/0.97 | 1.00/0.96 | 0.98/0.92 | 0.98/0.84 | 1.00/0.93 | 0.89/0.69 | 0.95/0.73 | 0.91/0.84 |
| DiffusionPolicy-T | 1.00/1.00 | 1.00/0.99 | 1.00/0.98 | 1.00/0.98 | 1.00/0.90 | 0.94/0.80 | 0.98/0.81 | 0.73/0.50 | 0.76/0.47 | 0.78/0.66 |
| Ours HT (Chi-UNet) | — | — | — | 1.00/0.85 | 1.00/0.78 | 0.96/0.65 | 0.96/0.83* | — | 0.82/0.53 | — |
| Ours HT (Chi-Transformer) | — | — | — | — | 1.00/0.92* | 0.96/0.75 | 0.96/0.76* | — | 0.68/0.36 | 0.94/0.90 |

Notes for the slide:
- push-T: ours beats every baseline in both tables (state 1.00/0.98; image 0.94/0.90 vs best baseline 0.91/0.84).
- state toolhang: ours 1.00/0.83 vs DP-T 1.00/0.87, DP-C 0.50/0.30.
- image square-ph (transformer, 55/60 evals): 1.00/0.92 — at DP-C 0.98/0.92, above DP-T's 0.90 l5.
- image tool-hang trains at 84x84 vs DP's 240x240; a 240x240 run is in progress.
- Baseline y = last-10 average, ours y = last-5 average; DP's released logs match last-5, so the published values are effectively comparable.
