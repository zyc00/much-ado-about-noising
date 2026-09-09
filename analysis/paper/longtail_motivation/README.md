# Long-tail motivation figure

RECOMMENDED: `fig_longtail_motivation_v4.{png,pdf}` (four panels, per-coordinate |z| axis), drawn by
`scripts/plot_longtail_motivation_v4.py` from `longtail_curves.json`, which `compute_longtail_curves.py` (kept in
`/mnt/pfs/yuchen/widowx_general_scale_20260906/`, copy in the scratchpad) computes on the cluster from the probe files;
the same statistic in one auditable file with a fast and a loop version is `scripts/tail_stats_single.py`. v3 (three
panels) is the previous layout. The chunk-norm variants (`fig_longtail_motivation.{png,pdf}` = earlier version, `_v2` = whitened
redraw, `summary*.json`) are kept for reference; on a 48-D norm the Gaussian reference collapses to a cliff and the
plot reads as "a line against a wall" rather than as a heavy tail, so it does not make the Student-t case on its own.

v4 protocol (2026-09-06, final; v3 identical except that Flow had 16 draws and one panel): 1,728 BridgeData V2 / WidowX states, 8-step chunks of the 6 continuous action
channels. Each model is standardized under ITS OWN scale, never with another model's sigma: a HG residual divided by the HG
head's predicted sigma(x) (`groot/ft_wxpurehg/checkpoint-9000`); b MSE residual (`ft_wxmse/checkpoint-20000`) with one
fixed scale (MSE assumes a homoscedastic Gaussian); c Flow samples (`ft_wxflow/checkpoint-20000`, 1024 per state, probe `scripts/probe_widowx_flow_k1024.py`, files
`widowx_heterogeneous_scale/raw_k1024/`, cluster `/mnt/pfs/yuchen/widowx_general_scale_20260906/raw_k1024/`) minus their
per-state mean with one fixed scale; d the same deviations divided by the Flow model's OWN conditional scale = the scalar
RMS spread of all 1024 draws at that state, computed in the RMS-normalized coordinate space (every 16th draw enters the
pooled curve; the scale uses all 1024). c vs d isolates the state-dependent part of Flow's spread (heteroscedasticity). In all panels the fixed/coordinate scale is the
per-coordinate RMS estimated on the other four episode folds (5 folds within task), and z is rescaled to unit RMS.
Curves: empirical residual tail (fraction of coordinates with magnitude > |z|); standard normal; Student-t with (nu, scale) matched to the tail on [1, 5] (fitted nu
printed in each panel: HG 7.1, MSE 5.7, Flow fixed 3.3, Flow own 4.9); Laplace with the same mean absolute value (a pure exponential tail). Box: P(|z| > 3) vs the Gaussian
0.27%, and the held-out NLL gain per coordinate of a maximum-likelihood Student-t over a maximum-likelihood Gaussian
(1-D fits on the other folds).

v4 numbers (longtail_curves.json): a HG 1.27% (4.7x; Student-t held-out gain 0.051 nat/dim), b MSE 1.64% (6.1x; 0.082),
c Flow fixed scale 2.12% (7.9x; 0.203), d Flow / own conditional scale 1.48% (5.5x; 0.081); flow sample minus the
demonstration label / own scale (spec item 6b, JSON only): 1.35% (5.0x; 0.063). Held-out NLL per coordinate
(Gaussian / Laplace / Student-t): a 1.418 / 1.362 / 1.368, b 1.419 / 1.331 / 1.337, c 1.418 / 1.225 / 1.215,
d 1.419 / 1.335 / 1.338.

What the tails are (checked, 2026-09-06): the log-tail slope is constant from 2 to 6 sigma for the regression residuals,
and a Laplace fits held-out data as well as the Student-t there (HG: Gaussian 1.418 / Laplace 1.362 / t 1.368 nat per
coordinate; MSE 1.419 / 1.331 / 1.337). So regression residuals are heavier-tailed than Gaussian with roughly
EXPONENTIAL decay, not a power law. After dividing by its own per-state spread the Flow model's samples are also near-exponential (kurtosis 4.3;
held-out Laplace 1.343 vs t 1.348); the earlier 'heavier than exponential' read (kurtosis 8.4) came from the fixed scale. Wording for the paper: "heavier-tailed than Gaussian" (which a Gaussian likelihood over-penalizes and a
moderate-nu Student-t accommodates), not "long-tailed". This matches the campaign result that light-but-finite nu is
SR-optimal. The queued heteroscedastic-L1 arms (GR1, OFT) are the direct test of Laplace + sigma(x) vs Student-t + sigma(x).

Chunk-norm variants for the record: whitened v2 (chi_48 null exact after cross-fit whitening) P(R > 1.4 sqrt d) = 9.5% /
9.5% / 11.6% vs 0.008%; the earlier coordinate-scaled version read 8.9 / 9.7 / 11.7% against a chi_48 null that ignored
within-chunk correlation (effective dimension ~25). Those versions also divided MSE/Flow by the HG sigma, which is no
longer done.
