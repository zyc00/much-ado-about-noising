# Heavy tails survive scale modelling: motivation figure for the Student-t likelihood

Regenerate from the repository root: `python scripts/plot_student_t_tails.py`
(inputs: `analysis/paper/widowx_heterogeneous_scale/raw/{hg,mse,flow}_rank*.npz`).

Data. The same 1,728 WidowX / BridgeData V2 demonstration states as the heterogeneous-scale figure (48 episodes
each of the three most frequent instructions, 12 chunk starts per episode), six continuous action channels over
eight steps (82,944 residual coordinates per head; 16 samples per state for Flow, 1,327,104 deviations).

Standardization (the same logic as the Tool-Hang data-side figure `data_ht_motivation/fig_heavy_tailed_residuals`):
every quantity is divided by a state-local scale first and by the coordinate scale second, so that neither
state-to-state nor coordinate-to-coordinate scale variation masquerades as a tail.
- state-local scale: sigma(x) of the heteroscedastic-Gaussian head (`groot/ft_wxpurehg/checkpoint-9000`, pure HG; the
  loss's per-state scalar: masked chunk-mean softplus(s_raw + sbias) + 1e-3), used for all three panels. It is a learned,
  observation-conditional scale estimated independently of the MSE residual and of the Flow samples (log-correlation
  with the MSE residual RMS 0.57; q10/50/90 = 0.13 / 0.23 / 0.33).
- coordinate scale: each of the six channels divided by its own RMS after the first step.
- a: HG head residual r / sigma(x).  b: MSE head (`groot/ft_wxmse/checkpoint-20000`) residual r / sigma(x).
  c: Flow head (`groot/ft_wxflow/checkpoint-20000`), 16 samples per state, (a_k - mean_k a_k) / sigma(x).
Every z is then rescaled to unit RMS. Curves: empirical residual tail P(|Z| > |z|), the fraction of coordinates with magnitude above |z|, on a log axis, the standard normal,
and a Student-t whose (nu, scale) are fitted to the empirical tail on |z| in [1, 5] (least squares in log tail probability,
the same role as the fitted curve of the Tool-Hang figure). Annotation: share of |z| > 3 relative to the Gaussian 0.27%.

Numbers (summary.json): HG 4.7x (1.26% beyond 3 sigma; kurtosis 3.1; tail-fit nu 6.9, MLE nu 3.8), MSE 5.0x (1.34%; 3.3;
tail-fit nu 6.3, MLE 3.7), Flow 6.6x (1.78%; 7.9; tail-fit nu 3.9, MLE 2.4). Tool-Hang (data-side, kNN local radius, held-out demos) reads 3.6x with nu 6.3 on
cleaner simulated data; Bridge's rotation channels (y, yaw) are the heaviest (per-channel kurtosis 5 and 5.5 for MSE).

Reconciliation note (2026-09-06): a first version divided b and c by ONE global scale and read 6.8x / 8.2x; that
excess included the state-to-state scale variation and is not comparable to the Tool-Hang protocol. With a global
scale the HG panel read 5.1x. Both variants are produced by the script (the global variant is not used).

Note. The maximum-likelihood Student-t (also in summary.json) has a smaller nu and overshoots the far tail because it is
dominated by the bulk; the drawn curve is the tail fit. HG is
the 9k checkpoint of a stopped 20k run (its sigma head is trained; the run was stopped for budget, not divergence).
