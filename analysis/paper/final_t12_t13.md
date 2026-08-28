# Final Table 12 / Table 13 (paper-ready, all methods)

Protocol (all rows, theirs and ours): 3 seeds x 300k updates; in-train eval
every 20k steps (15 checkpoints), 50 episodes per eval; best = seed-mean of
per-seed max; last-5 = seed-mean of the final-5-checkpoint average.
HT = heteroscedastic Student-t regression (pooled NLL, nu=2, learned sigma
head, single forward at t=0). Unified recipe per backbone; transport uses
obs_steps=8. Column order follows the paper (mh before ph).

Sources. Flow / Regression / Straight Flow rows: paper values verbatim
(arXiv:2512.01809) — no artifacts were released for these baselines, so
they cannot be audited. MIP rows: recomputed from released checkpoints
(huggingface.co/ChaoyiPan/mip-checkpoints) wherever artifacts exist; (c)
marks cells where the artifact value replaces the paper claim (both
directions occur). HT rows: our runs.

## Table 12 — state

| arch | method | lift-mh | lift-ph | can-mh | can-ph | square-mh | square-ph | transport-mh | transport-ph | tool-hang | push-T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DiT | Flow | 1.00/0.99 | 1.00/1.00 | 1.00/0.94 | 1.00/1.00 | 0.88/0.75 | 1.00/0.94 | 0.40/0.27 | 0.80/0.70 | 0.86/0.75 | 0.98/0.95 |
| DiT | Regression | 1.00/0.99 | 1.00/1.00 | 0.92/0.90 | 1.00/0.98 | 0.72/0.53 | 0.94/0.86 | 0.12/0.06 | 0.50/0.44 | 0.52/0.39 | 0.92/0.83 |
| DiT | Straight Flow | 1.00/0.98 | 1.00/1.00 | 0.96/0.90 | 1.00/0.99 | 0.72/0.66 | 0.96/0.93 | 0.20/0.14 | 0.56/0.48 | 0.70/0.59 | 0.90/0.86 |
| DiT | MIP | 1.00/0.99 | 1.00/1.00 | 0.98/0.95 | 1.00/1.00 | 0.82(c)/0.81 | 0.98/0.94 | 0.44/0.38 | 0.68(c)/0.55(c) | 0.50(c)/0.37(c) | 0.95/0.92 |
| DiT | HT (ours) | 1.00/1.00 | 1.00/0.98 | 0.98/0.90 | 1.00/0.97 | 0.72/0.62 | 0.89/0.82 | 0.33/0.21 | 0.54/0.45 | 0.65/0.56 | 0.97/0.95 |
| Chi-Tf | Flow | 1.00/1.00 | 1.00/1.00 | 1.00/0.93 | 1.00/0.98 | 0.78/0.74 | 0.96/0.89 | 0.44/0.34 | 0.88/0.64 | 0.68/0.54 | 0.91/0.89 |
| Chi-Tf | Regression | 1.00/0.99 | 1.00/0.99 | 0.98/0.92 | 1.00/0.96 | 0.74/0.61 | 0.92/0.85 | 0.28/0.20 | 0.68/0.51 | 0.40/0.36 | 0.93/0.88 |
| Chi-Tf | Straight Flow | 1.00/0.99 | 1.00/1.00 | 0.98/0.92 | 1.00/0.99 | 0.68/0.58 | 0.96/0.89 | 0.24/0.16 | 0.62/0.54 | 0.60/0.55 | 0.94/0.90 |
| Chi-Tf | MIP | 1.00/1.00 | 1.00/1.00 | 0.96/0.95 | 1.00/1.00 | 0.80(c)/0.73 | 0.96/0.89 | 0.42/0.37 | 0.69(c)/0.58(c) | 0.74(c)/0.62(c) | 0.94/0.92 |
| Chi-Tf | HT (ours) | 1.00/0.99 | 1.00/0.98 | 1.00/0.94 | 1.00/0.96 | 0.72/0.60 | 0.95/0.85 | 0.33/0.22 | 0.62/0.47 | 0.90/0.80 | 0.96/0.93 |
| Chi-UNet | Flow | 1.00/1.00 | 1.00/1.00 | 1.00/0.98 | 1.00/1.00 | 0.90/0.78 | 0.98/0.94 | 0.52/0.40 | 0.80/0.73 | 0.84/0.70 | 0.98/0.94 |
| Chi-UNet | Regression | 1.00/1.00 | 1.00/1.00 | 1.00/0.96 | 1.00/0.99 | 0.94/0.82 | 1.00/0.91 | 0.22/0.16 | 0.64/0.55 | 0.68/0.64 | 0.95/0.89 |
| Chi-UNet | Straight Flow | 1.00/1.00 | 1.00/1.00 | 1.00/0.92 | 1.00/0.99 | 0.94/0.79 | 0.98/0.90 | 0.22/0.15 | 0.64/0.52 | 0.50/0.00 | 0.93/0.88 |
| Chi-UNet | MIP | 1.00/1.00 | 1.00/1.00 | 1.00/0.98 | 1.00/0.99 | 0.89(c)/0.81 | 1.00/0.94 | 0.62/0.46 | 0.81(c)/0.66(c) | 0.57(c)/0.43(c) | 0.97/0.95 |
| Chi-UNet | HT (ours) | 1.00/1.00 | 1.00/1.00 | 1.00/0.97 | 1.00/0.99 | 0.87/0.74 | 0.99/0.92 | 0.63/0.49 | 0.78/0.67 | 0.75/0.60 | 0.93/0.91 |

Kitchen is omitted: our reproduction of the eval harness stalls at 2/4
subtasks for every method, including MIP's released checkpoints; no
faithful number exists (paper kitchen column: all methods 0.86-1.00).

## Table 13 — vision

| arch | method | lift-mh | lift-ph | can-mh | can-ph | square-mh | square-ph | transport-mh | transport-ph | tool-hang | push-T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DiT | Flow | 1.00/1.00 | 1.00/1.00 | 0.96/0.94 | 1.00/0.99 | 0.82/0.76 | 0.96/0.94 | 0.32/0.20 | 0.84/0.83 | 0.78/0.57 | 0.92/0.89 |
| DiT | Regression | 1.00/0.99 | 1.00/1.00 | 0.92/0.81 | 1.00/1.00 | 0.74/0.67 | 0.94/0.84 | 0.14/0.08 | 0.74/0.56 | 0.28/0.18 | 0.83/0.77 |
| DiT | Straight Flow | 1.00/0.99 | 1.00/0.99 | 0.98/0.95 | 1.00/0.98 | 0.82/0.72 | 1.00/0.93 | 0.26/0.19 | 0.86/0.83 | 0.46/0.40 | 0.85/0.79 |
| DiT | MIP | 1.00/0.99 | 1.00/1.00 | 0.99(c)/0.96 | 1.00/0.98 | 0.83(c)/0.83 | 0.94(c)/0.92 | 0.50/0.31 | 0.90/0.84 | 0.49(c)/0.66* | 0.93(c)/0.87 |
| DiT | HT (ours) | 1.00/0.99 | 1.00/1.00 | 0.99/0.94 | 1.00/0.98 | 0.83/0.73 | 0.96/0.88 | 0.72/0.58 | 0.90/0.83 | 0.75/0.66 | 0.91/0.87 |
| Chi-Tf | Flow | 1.00/0.99 | 1.00/1.00 | 0.98/0.92 | 1.00/0.96 | 0.70/0.66 | 0.98/0.93 | 0.24/0.22 | 0.80/0.77 | 0.54/0.40 | 0.89/0.85 |
| Chi-Tf | Regression | 1.00/0.98 | 1.00/0.98 | 1.00/0.94 | 1.00/0.96 | 0.76/0.70 | 0.98/0.90 | 0.40/0.27 | 0.94/0.87 | 0.44/0.36 | 0.85/0.81 |
| Chi-Tf | Straight Flow | 1.00/1.00 | 1.00/1.00 | 1.00/0.95 | 1.00/0.98 | 0.90/0.78 | 0.98/0.94 | 0.32/0.25 | 0.86/0.70 | 0.36/0.28 | 0.86/0.80 |
| Chi-Tf | MIP | 1.00/0.98 | 1.00/1.00 | 0.99(c)/0.91 | 0.99(c)/0.98 | 0.82(c)/0.21 | 0.92(c)/0.04 | 0.18/0.06 | 0.86/0.69 | 0.53(c)/0.48 | 0.92(c)/0.83 |
| Chi-Tf | HT (ours) | 1.00/0.99 | 1.00/0.99 | 0.99/0.93 | 1.00/0.97 | 0.85/0.75 | 0.98/0.90 | 0.78/0.66 | 0.90/0.83 | 0.54/0.38 | 0.89/0.82 |
| Chi-UNet | Flow | 1.00/1.00 | 1.00/1.00 | 1.00/0.97 | 1.00/0.98 | 0.90/0.79 | 0.96/0.90 | 0.24/0.16 | 0.78/0.61 | 0.48/0.37 | 0.92/0.87 |
| Chi-UNet | Regression | 1.00/0.96 | 1.00/0.99 | 0.84/0.70 | 0.98/0.87 | 0.74/0.66 | 0.94/0.86 | 0.18/0.10 | 0.66/0.64 | 0.30/0.23 | 0.90/0.85 |
| Chi-UNet | Straight Flow | 1.00/0.94 | 1.00/0.99 | 0.98/0.93 | 1.00/0.96 | 0.72/0.68 | 0.92/0.62 | 0.00/0.00 | 0.50/0.22 | 0.06/0.02 | 0.87/0.82 |
| Chi-UNet | MIP | 1.00/1.00 | 1.00/1.00 | 0.98(c)/0.95 | 1.00/0.98 | 0.84(c)/0.84 | 0.90(c)/0.91 | 0.52/0.37 | 0.96/0.91 | 0.62(c)/0.50 | 0.94(c)/0.78 |
| Chi-UNet | HT (ours) | 1.00/0.94 | 1.00/0.97 | 0.93/0.85 | 0.97/0.88 | 0.76/0.64 | 0.90/0.81 | 0.62/0.48 | 0.87/0.76 | 0.62/0.47 | 0.85/0.80 |

## Column summary (best-of-backbones, ties at table precision count as parity)

vs artifact-audited MIP:
- T12: SOTA-or-tie on best 7/10, last-5 6/10. Wins: tool-hang
  (0.90/0.80 vs 0.74/0.62; HT > MIP on every backbone), transport-mh,
  push-T last-5 parity. Losses: square-mh, square-ph, transport-ph best.
- T13: SOTA-or-tie on best 7/10, last-5 5/10. Wins: transport-mh
  (0.78/0.66 vs 0.52/0.37), tool-hang (0.75/0.66 vs 0.62/0.50),
  square-mh/square-ph best. Losses: transport-ph, push-T best.

vs the full paper field (Flow / Regression / Straight Flow rows are
unaudited paper claims — every audited MIP cell moved on recomputation,
so treat these comparisons accordingly):
- T12: best 6/10, last-5 5/10. tool-hang still ours (0.90/0.80 vs paper
  DiT Flow 0.86/0.75); transport-ph flips against us on paper CT Flow
  0.88 best / CU Flow 0.73 last-5.
- T13: best 4/10, last-5 3/10. transport-mh remains a large win
  (0.78/0.66 vs 0.52/0.37); tool-hang last-5 ties the starred cell.

Footnotes:
(c) checkpoint-recomputed MIP cell (replaces the paper claim; revisions
go both directions — e.g. state tool-hang DiT 0.92 -> 0.50, but vision
square-mh Tf 0.72 -> 0.82).
* internally inconsistent reference: artifact best (0.49) is below the
paper last-5 (0.66), and no eval series was released for image runs to
check it; keep the flag or drop the last-5 at edit time.
Push-T appears in both tables: T12 = state observations (our runs:
pusht_state), T13 = image (pusht_image). An earlier draft leaked state
runs into the vision cell; this version separates them.
Cross-check supporting the audit caveat: our 3-seed rerun of Chi-UNet
Flow on transport (obs8) gives ph 0.72/0.59 and mh 0.28/0.21 vs the
paper's 0.80/0.73 and 0.52/0.40 — the unaudited baseline rows likely
carry the same upward bias the MIP artifacts revealed.

Stability note for the text: MIP's transformer collapses on vision
last-5 (square-mh 0.21, square-ph 0.04, transport-mh 0.06) while the
transformer is HT's strongest vision backbone on those columns
(0.75/0.90/0.66) — the late-training stability asymmetry favors HT.
