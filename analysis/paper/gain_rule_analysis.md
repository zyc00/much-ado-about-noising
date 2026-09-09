# Can residual statistics of the data predict the gain of flow and HT over MSE?

Question (2026-09-06): given the residual scale (rms), its tail heaviness and its heteroscedasticity,
is there a rule that predicts how much flow and HT gain over MSE on a dataset?
Scripts and raw outputs: `scripts_gain_rule/` (`resid_stats.py` computes the statistics on the utility
pod, `gain_rule2.py` runs the tests; `report_*.txt` are the three variants). Figure: `fig_gain_rule.png`.

## Setup

Outcomes. RoboMimic five-head table (`robomimic_5method_table.md`, late-average success), 10 datasets
x 3 backbones x 2 modalities; per dataset the gain is the mean over the 6 cells of flow - MSE,
HT - MSE and HT - flow. VLA stacks with an MSE cell: GR1 (37.8 / 44.1 / 47.5), pi0.5 (96.8 / 96.9 /
97.6), OFT libero-long at 20k (84 / no flow / 89); WidowX, Google Robot and Cosmos3 MSE cells pending.

Predictors, computed from the demonstrations alone (no model), on q01/q99-normalized actions clipped to
[-1, 1] as the GR00T processor does (variant `out_c1`; unclipped `out_c0` also run), at each policy's
chunk horizon (10 for RoboMimic, 8 for GR00T/OFT, 16 Cosmos3, 50 pi0.5; H=8 also run). Three residual
definitions per chunk: `knn` = chunk minus the mean chunk of the 8 nearest-state chunks from other
episodes of the same task (state-conditional spread, closest to an MSE model's training residual),
`zoh` = chunk minus the last executed action (innovation), `sg` = chunk minus a Savitzky-Golay smooth
(jitter). Per residual: rms; tail = top-1% chunk share of the squared residual divided by the Gaussian
reference at the same H*d (`tail_x`), element kurtosis, Student-t df fit; heteroscedasticity = split-half
reliability of the per-chunk log mean-square across the two time halves (`rel_time`), and the excess
dispersion of log chunk scale over the chi-square expectation (`sd_excess`). 78 predictor columns.

Validation of the proxy against real model residuals (`modelside.json`, existing dumps): GR1 MSE
checkpoint residuals give tail_x 5.8, kurtosis 8.5, rel_time 0.965; the data-side knn statistic gives
5.3 / 5.8 / 0.99 (0.95 unclipped). Google Robot flow/HT residuals give rel_time 0.65-0.69, data-side
0.65-0.67. The data-side statistic reproduces the ordering the model residuals show.

## Results

1. Headroom explains the gains. Spearman of (1 - MSE success) with flow - MSE is +0.95 and with
   HT - MSE +0.84 (n = 10). A one-term rule gain = a + b*(1 - MSE) has leave-one-out R^2 0.82 for the
   HT gain (0.18 for the flow gain, whose LOO is dominated by transport-ph). HT - flow has no headroom
   dependence (+0.13).
2. No residual statistic adds predictive power beyond headroom. Adding the best of the 78 columns to
   the headroom rule changes leave-one-out R^2 by at most +0.05 (flow) and 0.00 (HT), in all three
   variants (clipped H=10, unclipped H=10, clipped H=8).
3. After residualizing both sides on headroom, no column reaches p < 0.05 for flow - MSE or HT - MSE at
   n = 10, and the top-ranked column changes from variant to variant (rms, kurtosis, rel_time, rel_dim,
   frac > 4 median), the signature of selection noise over 78 columns. Relative gains
   (gain / headroom, n = 8 non-ceiling datasets) show a few p = 0.01-0.05 entries that again differ
   between variants.
4. The one direction that is consistent in every variant: heavier tails go with a smaller HT gain, not
   a larger one. Partial rho of element kurtosis with HT - MSE is -0.54 (clipped), -0.7 to -0.9 for the
   relative HT gain (unclipped), and in the three same-task ph -> mh pairs (multi-human data: tails
   4-11x heavier unclipped) the relative HT gain falls in all three (can -0.43, square -0.10,
   transport -0.16) while the relative flow gain rises in all three (+0.04, +0.17, +0.15). HT - flow
   drops by 0.22 on transport. Heavy demonstration tails favour flow over HT.
5. VLA stacks do not order by either statistic. WidowX has the lowest chunk-scale reliability (0.60)
   and the lightest clipped tail (2.3) yet the largest HT - flow gain (+5.8); Google Robot has similar
   statistics (0.65, 3.3) and HT loses (-3.8); GR1 has the highest reliability (0.99) and HT wins
   (+3.4); Cosmos3 (0.77, 4.6) ties at the c=2 point. Among the three stacks with an MSE cell the HT
   gain again tracks headroom (GR1 62 points of headroom: +9.7; OFT 16: +5.0; pi0.5 3: +0.8).

## Conclusion

There is no rule from tail or heteroscedasticity statistics of the data that predicts the gain of flow
or HT over MSE. What predicts both gains is how far MSE is from the ceiling; the residual statistics
carry no additional leave-one-out signal on 10 RoboMimic datasets and do not order the VLA stacks.
The only consistent statistical direction is the opposite of the noise-matching intuition: datasets
with heavier demonstration tails give HT smaller gains and flow larger ones. This agrees with the
mechanism results already in the paper (HT's gain is a precision gain on the reliable, state-dependent
part of the residual, and heavy unpredictable tails are what it must ignore), and with the WidowX
gate-reliability study. Recommendation: do not claim a predictive rule; if a statistic is named at all,
name headroom, and present the tail direction as a caveat against the "heavy-tailed noise" framing.

Caveats: n = 10 datasets for the main test (statistically weak against 78 candidates); the knn
residual uses proprioceptive state only, so for scene-diverse data (Bridge, Google Robot) it includes
scene variation; pending MSE cells (WidowX, Google Robot, Cosmos3) can be added by editing
`scripts_gain_rule/outcomes.json` and rerunning `gain_rule2.py`.
