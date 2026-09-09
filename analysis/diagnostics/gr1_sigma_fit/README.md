# GR1 sigma versus actual fitting error

Checkpoint: `/mnt/pfs/yuchen/groot/ft_gr1c2/checkpoint-60000`.

Original probe: 2,400 training states, eval-mode forward; all 24 GR1 datasets configured.
The dump lacks task/episode IDs and contact annotations. Results are pooled, descriptive, and not held-out.
Both sigma and residual use exactly the first 8 valid steps and all 29 valid joint channels.
Residual is prediction minus demonstration label in the original probe's normalized action space.

Spearman rho: 0.9090; log-log Pearson r: 0.9109.
Highest / lowest predicted-scale quintile residual RMS: 9.766x.
Median per-state RMS / sigma: 0.977.

| Predicted-scale group | N | Predicted scale RMS | Actual residual RMS |
|---|---:|---:|---:|
| 1 | 480 | 0.01391 | 0.01494 |
| 2 | 480 | 0.02281 | 0.02717 |
| 3 | 480 | 0.03307 | 0.03884 |
| 4 | 480 | 0.05175 | 0.06460 |
| 5 | 480 | 0.13445 | 0.14595 |

The groups are post-hoc visualization bins of ONE general HT policy, not separately trained specialists.
Strong association does not by itself establish exact calibration, irreducible noise, within-task variation, or a contact-specific mechanism.
The Student-t scale is not exactly its standard deviation. For the actual nu=928 checkpoint the conversion factor is sqrt(928/926), only ~1.0011.
The original probe overrides loss configuration, so we do not interpret its saved loss values as the checkpoint's original NLL.
The prediction and sigma statistics above are recomputed directly from stored decoder outputs and checked against saved m/sigma.
No bootstrap interval is reported without episode IDs, to avoid treating correlated states as independent.
