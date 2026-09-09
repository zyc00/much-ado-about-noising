# Component ablation: HT vs homoscedastic Student-t vs heteroscedastic Gaussian (RoboMimic)

Cells = best checkpoint / last-5 average of in-train 50-episode evals every 20k over 300k steps, Chi-UNet, state
input, our harness. Tool-Hang: delta-action recipe, plain family (report parts CCXX / CCCXVIII), 2 seeds.
Transport: absolute actions, observation history 4 (PH) / 8 (MH), the "fair" cells of the transport tail study
(part CCCXXIII); HT 3-4 seeds, hetero-Gaussian and MSE 1 seed, homoscedastic-t 2 seeds (`trpha_gtos4_s{1000,42}`,
`trmha_gtos8_s{1000,42}`, launched 2026-09-06 via `k_align_anytask.sh`, LOSS=regression_globalt).

| task | HT (input-dependent scale, t tail) | homoscedastic Student-t (global scale, t tail) | heteroscedastic Gaussian (input-dependent scale, Gaussian) | MSE |
|---|---|---|---|---|
| Tool-Hang | 86-88 / 75.4-81.6 | 64 / 57.6, 58 / 44.0 | 86 / 68.8 (pooled truth 73) | 33-43 |
| Transport-PH | 75.8 / 63.2 (75.0/62.5, 75.0/65.0, 77.5/62.0) | 77.5 / 67.8 (77.5/68.0, 77.5/67.5) | 70.0 / 61.5 | 62.5 / 45.0 |
| Transport-MH | 60.0 / 49.5 (52.5/45.5, 72.5/55.5, 55.0/49.0, 60.0/48.0) | 50.0 / 39.5 (52.5/38.0, 47.5/41.0) | 52.5 / 41.5 | 27.5 / 11.0 |

Homoscedastic-t curves (20k..300k): PH s1000 .55 .53 .50 .60 .62 .45 .70 .72 .75 .53 .55 .65 .68 .75 .78;
PH s42 .55 .42 .60 .68 .50 .57 .72 .65 .78 .72 .75 .68 .60 .68 .68; MH s1000 .20 .28 .40 .42 .25 .45 .38 .53 .38 .38
.33 .40 .38 .45 .35; MH s42 .28 .33 .28 .25 .38 .33 .40 .28 .47 .35 .42 .42 .42 .45 .33.

Reading. The contribution of the input-dependent scale is task-dependent: on Tool-Hang removing it costs 20-35
points (the global-scale arm lands at the fixed-scale Cauchy band), on Transport-MH about 10 points on both
statistics, on Transport-PH nothing (the global-scale arm ties or edges HT within eval noise, +-6). The Student-t
tail contributes on every task: the heteroscedastic Gaussian sits below HT by 4-8 points on transport and, on
Tool-Hang, keeps HT's peak but loses stability (last-5 69 vs 75-82, pooled truth 73 vs 84-87). Both components
matter, but which one dominates depends on the data: the state-conditional scale where difficulty is strongly
state-dependent (Tool-Hang alignment, multi-operator MH data), the tail everywhere.

Caveats: single hetero-Gaussian/MSE seed on transport; 50-episode evals; the Tool-Hang row is an older battery
whose absolute HT numbers differ from the five-head table's Tool-Hang cell.

## Paper table (last-5 average only; seed means; Tool-Hang HT = stronger seed 81.6, per user 2026-09-06)

| task | HT | homoscedastic Student-t | heteroscedastic Gaussian |
|---|---|---|---|
| Tool-Hang | 81.6 | 50.8 | 68.8 |
| Transport-PH | 63.2 | 67.8 | 61.5 |
| Transport-MH | 49.5 | 39.5 | 41.5 |
