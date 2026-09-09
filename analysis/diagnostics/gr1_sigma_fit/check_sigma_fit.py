"""Descriptive training-fit diagnostic, not a held-out/contact validation.

Run on the copied original NPZ. No fitting, rescaling, or sample selection.
The source dump has no task/episode IDs, so no independent-sample CIs are claimed.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "vectors.npz"
z = np.load(SOURCE)
sigma = z["sigma"].astype(float)
energy = z["m_recomputed"].astype(float)
rms = np.sqrt(energy)
assert tuple(z["residual_shape"]) == (2400, 8, 29)
assert np.all(z["d"] == 232)
assert np.all(np.isfinite(energy)) and np.all(sigma > 0)
assert np.allclose(energy, z["m"], rtol=2e-6, atol=1e-9)
assert np.allclose(z["sigma_reconstructed"], sigma, rtol=2e-6)

def association(s, e):
    y = np.sqrt(e)
    order = np.argsort(s, kind="stable")
    bins = []
    for j, idx in enumerate(np.array_split(order, 5)):
        bins.append(dict(group=j + 1, n=len(idx), sigma_median=float(np.median(s[idx])),
                         sigma_rms=float(np.sqrt(np.mean(s[idx] ** 2))),
                         residual_rms=float(np.sqrt(np.mean(e[idx]))),
                         residual_rms_median=float(np.median(y[idx]))))
    return dict(spearman=float(spearmanr(s, y).statistic),
                pearson_log=float(pearsonr(np.log(s), np.log(y)).statistic),
                residual_to_sigma_quantiles=dict(zip(["p10", "p50", "p90"],
                                                     np.quantile(y / s, [.1, .5, .9]).tolist())),
                pooled_residual_rms=float(np.sqrt(np.mean(e))),
                pooled_sigma_rms=float(np.sqrt(np.mean(s ** 2))),
                normalized_residual_rms=float(np.sqrt(np.mean(e / s ** 2))),
                high_low_residual_ratio=bins[-1]["residual_rms"] / bins[0]["residual_rms"],
                bins=bins)

result = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              checkpoint=str(z["ckpt"]), n=len(sigma), steps=8, channels=29,
              scope="Training data; task and episode identities not saved. No contact labels.",
              eval_mode=association(sigma, energy),
              train_mode=association(z["sigma_train"].astype(float), z["m_train"].astype(float)),
              original_source="/mnt/pfs/yuchen/groot/resid_dump3_gr1_c2_60k_all.npz",
              original_sha256="d26f053332bbeff048a3f4a0224888bcd0be233d3ae0622a3d2b8f03f859eb5b",
              channel_checks={name: association(sigma, z["channel_energies"][:, j])
                              for j, name in enumerate(["arms", "hands", "waist"])})
(ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
ax = axes[0]
ax.scatter(sigma, rms, s=8, alpha=.22, color="#2878A8", rasterized=True, linewidths=0)
lo = min(sigma.min(), rms.min()) * .8
hi = max(sigma.max(), rms.max()) * 1.2
ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=1, label="RMS = sigma (reference)")
ax.set(xscale="log", yscale="log", xlabel="Predicted chunk scale sigma",
       ylabel="Measured chunk residual RMS", title=f"Same-state fit: Spearman rho = {result['eval_mode']['spearman']:.3f}",
       xlim=(lo, hi), ylim=(lo, hi))
ax.legend(fontsize=8)
ax = axes[1]
bins = result["eval_mode"]["bins"]
x = np.arange(1, 6)
ax.plot(x, [b["sigma_rms"] for b in bins], "o-", color="#2878A8", label="Predicted: sqrt(mean sigma²)")
ax.plot(x, [b["residual_rms"] for b in bins], "s-", color="#D55E00", label="Measured: sqrt(mean residual²)")
ax.set(xticks=x, xticklabels=["Lowest", "2", "3", "4", "Highest"], xlabel="Groups defined only by predicted sigma (480 states each)",
       ylabel="Scale (normalized action units)", title="Scale ordering and calibration")
ax.legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=.16)
fig.suptitle("GR1 HT c=2 / 60k — 2,400 training states, 8 × 29 action chunks", fontsize=11)
fig.savefig(ROOT / "sigma_vs_fit.png", dpi=180)
fig.savefig(ROOT / "sigma_vs_fit.pdf")

e = result["eval_mode"]
lines = ["# GR1 sigma versus actual fitting error", "", f"Checkpoint: `{result['checkpoint']}`.",
         "", "Original probe: 2,400 training states, eval-mode forward; all 24 GR1 datasets configured.",
         "The dump lacks task/episode IDs and contact annotations. Results are pooled, descriptive, and not held-out.",
         "Both sigma and residual use exactly the first 8 valid steps and all 29 valid joint channels.",
         "Residual is prediction minus demonstration label in the original probe's normalized action space.",
         "", f"Spearman rho: {e['spearman']:.4f}; log-log Pearson r: {e['pearson_log']:.4f}.",
         f"Highest / lowest predicted-scale quintile residual RMS: {e['high_low_residual_ratio']:.3f}x.",
         f"Median per-state RMS / sigma: {e['residual_to_sigma_quantiles']['p50']:.3f}.",
         "", "| Predicted-scale group | N | Predicted scale RMS | Actual residual RMS |", "|---|---:|---:|---:|"]
lines += [f"| {b['group']} | {b['n']} | {b['sigma_rms']:.5f} | {b['residual_rms']:.5f} |" for b in bins]
lines += ["", "The groups are post-hoc visualization bins of ONE general HT policy, not separately trained specialists.",
          "Strong association does not by itself establish exact calibration, irreducible noise, within-task variation, or a contact-specific mechanism.",
          "The Student-t scale is not exactly its standard deviation. For the actual nu=928 checkpoint the conversion factor is sqrt(928/926), only ~1.0011.",
          "The original probe overrides loss configuration, so we do not interpret its saved loss values as the checkpoint's original NLL.",
          "The prediction and sigma statistics above are recomputed directly from stored decoder outputs and checked against saved m/sigma.",
          "No bootstrap interval is reported without episode IDs, to avoid treating correlated states as independent.", ""]
(ROOT / "README.md").write_text("\n".join(lines))
print(json.dumps(result, indent=2))
