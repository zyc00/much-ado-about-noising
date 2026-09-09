# RoboMimic training convergence: HT vs flow (Transport-PH / Transport-MH)

Figure `fig_robomimic_convergence.png` (script `fig_robomimic_convergence.py`, data `rm_convergence_curves.json`,
extracted 2026-09-06 from the cluster run logs `logs/t12_transport_{ph,mh}_state_abs_chiunet_s{1,2,3}_{flow,ht,mip}_taskobssteps8`
and `logs/trmha_os8_s{1000,2000,42,5}`).

Protocol (identical for every head): our harness, absolute actions, Chi-UNet, observation history 8, 300k steps,
in-train evaluation of 50 episodes every 20k steps (15 points), 3 seeds per head (Transport-MH HT: 3 + 4 seeds from
the same recipe launched through `k_align_anytask.sh`; MSE: `trpha_l2os8_s{1000,2000,42}`, `trmha_l2os8_s{1000,2000}`,
launched 2026-09-06). Flow is read at its default 9-step Euler sampler
(`mean_success_9`); the dashed line is the same flow checkpoints at 1 step. HT is one pass. Eval noise per point is
about +-7 points (50 episodes); the shaded band is the seed range.

Seed-mean success (%) at 20k ... 300k:

| task | head | 20k | 40k | 60k | 80k | 100k | 120k | 140k | 160k | 180k | 200k | 220k | 240k | 260k | 280k | 300k | best | last-5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transport-PH | MSE (1 pass, 3 seeds) | 57 | 50 | 61 | 51 | 44 | 62 | 54 | 56 | 52 | 46 | 50 | 56 | 55 | 53 | 58 | 70.0 | 54.5 |
| Transport-PH | flow (9 steps) | 39 | 47 | 60 | 67 | 63 | 72 | 70 | 74 | 74 | 72 | 68 | 72 | 70 | 75 | 79 | 84.2 | 72.8 |
| Transport-PH | flow (1 step) | 0 | 0 | 3 | 16 | 44 | 44 | 59 | 55 | 64 | 67 | 52 | 62 | 64 | 56 | 62 | 72.5 | 59.3 |
| Transport-PH | HT | 68 | 67 | 62 | 62 | 73 | 69 | 62 | 64 | 67 | 68 | 73 | 68 | 66 | 58 | 68 | 78.3 | 66.5 |
| Transport-PH | MIP | 65 | 63 | 65 | 61 | 71 | 65 | 68 | 68 | 72 | 66 | 68 | 72 | 70 | 70 | 68 | 77.5 | 69.7 |
| Transport-MH | MSE (1 pass, 2 seeds) | 18 | 11 | 14 | 26 | 15 | 16 | 25 | 16 | 15 | 18 | 14 | 18 | 14 | 10 | 16 | 27.5 | 14.2 |
| Transport-MH | flow (9 steps) | 2 | 5 | 24 | 31 | 42 | 38 | 47 | 38 | 50 | 52 | 43 | 40 | 43 | 42 | 44 | 57.5 | 42.7 |
| Transport-MH | flow (1 step) | 0 | 0 | 0 | 3 | 2 | 8 | 12 | 14 | 16 | 12 | 18 | 19 | 22 | 23 | 24 | 28.3 | 21.3 |
| Transport-MH | HT (7 seeds) | 37 | 47 | 50 | 41 | 48 | 49 | 50 | 50 | 48 | 49 | 51 | 49 | 44 | 51 | 51 | 61.4 | 49.1 |

Reading. MSE also starts fast (57 at 20k on PH) but plateaus 12 points under HT on PH (54.5 vs 66.5 last-5) and
at 14 on MH, so HT's early plateau is not a property of single-pass regression as such. HT is at its plateau at the first evaluation (20k): 68 on PH, and 47-50 by 40-60k on MH. Flow at 9 steps
needs 60-80k steps on PH to reach HT's 20k level and 140-180k on MH; at 1 step it needs 140k+ on PH and never
reaches it on MH. Endpoints differ by task: on PH flow's late average is 6 points above HT (72.8 vs 66.5, and MIP
69.7), on MH HT stays 6 points above flow (49.1 vs 42.7). So the convergence claim holds on both tasks (3-4x fewer
steps on PH, 4-7x on MH to reach the same success), while the endpoint claim holds only on MH.

Caveats. These are our own runs with observation history 8 and absolute actions (HT's fair cell from the transport
tail study); the five-head table in `robomimic_5method_table.md` uses a different harness and is not comparable
point by point. Flow's 9-step readout is the harness default; the earlier one-step readout in this session was a
key-selection error (`mean_success_1` is the 1-step column of a flow run), corrected here.
