"""Q-Q and calibration figure for the converged HG (hetero-Gaussian, nu=inf) and
HT (hetero-t, nu=464) GR1 policies.

Input: resid_dump_gr1_{hg22k,ht60k}.npz produced by groot/resid_dump_gr1.py on
the cluster (eval mode, 4 GR1 training datasets, same samples for both arms):
  r     [N, T, A]  signed residual pred - target (normalized action units)
  sigma [N]        learned per-sample sigma = chunk-mean softplus(s_raw + sbias) + 1e-3
  m     [N]        mean squared residual over the d masked elements
  d     [N]        masked element count (T*A)

Usage: python plot_qq_hg_ht.py <dump_dir> <out.png> [file pattern with {suf}, default resid_dump_gr1_{suf}.npz] [data tag for the title]
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dump_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
PATTERN = sys.argv[3] if len(sys.argv) > 3 else "resid_dump_gr1_{suf}.npz"
DATA_TAG = sys.argv[4] if len(sys.argv) > 4 else "4 of 24 GR1 datasets"

# palette (dataviz reference instance, light mode): slot 1 blue, slot 2 orange
ARMS = [
    ("HG (Gaussian NLL, 22k)", "hg22k", "#2a78d6"),
    ("HT (Student-t nu=464, 60k)", "ht60k", "#eb6834"),
    ("HT c=2 (Student-t nu=928, 60k)", "c2_60k", "#1baf7a"),
]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titleweight": "regular", "axes.titlesize": 10.5,
    "axes.titlelocation": "left", "legend.frameon": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
})


def qq_subsample(n, n_mid=3000, n_tail=1500):
    """Order-statistic indices: dense in both tails, even in the middle."""
    idx = np.concatenate([np.arange(min(n_tail, n)), np.linspace(0, n - 1, n_mid).astype(int),
                          np.arange(max(0, n - n_tail), n)])
    return np.unique(idx)


data = {}
for label, suf, color in ARMS:
    if not (dump_dir / PATTERN.format(suf=suf)).exists():
        print(f"skip {label}: no {PATTERN.format(suf=suf)}"); continue
    z = np.load(dump_dir / PATTERN.format(suf=suf))
    r, sigma, m, d = z["r"], z["sigma"], z["m"], z["d"]
    zz = (r / sigma[:, None, None]).reshape(-1)          # per-element standardized residual
    ratio = np.sqrt(m) / sigma                            # per-sample residual RMS / sigma
    zdim = np.sqrt(((r / sigma[:, None, None]) ** 2).mean(axis=(0, 1)))  # per-dim RMS of z
    data[suf] = dict(label=label, color=color, z=zz, ratio=ratio, zdim=zdim, sigma=sigma, m=m,
                     d=float(d.mean()), n=len(sigma), ckpt=str(z["ckpt"]), sbias=float(z["sbias"]),
                     loss_eval=float(z["loss_eval"].mean()) if "loss_eval" in z.files else None,
                     loss_train=float(z["loss_train"].mean()) if "loss_train" in z.files else None)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), dpi=180)
(axA, axB), (axC, axD) = axes

# ---- A: per-element r/sigma vs N(0,1) (scale + shape) -------------------------
lines = []
for suf, D in data.items():
    z = np.sort(D["z"]); n = len(z)
    p = (np.arange(1, n + 1) - 0.5) / n
    idx = qq_subsample(n)
    th = stats.norm.ppf(p[idx])
    axA.scatter(th, z[idx], s=5, color=D["color"], alpha=0.8, linewidths=0, label=D["label"])
    D["z_std"] = float(z.std()); D["z_mad"] = float(1.4826 * np.median(np.abs(z - np.median(z))))
    D["frac_gt3"] = float((np.abs(z) > 3).mean()); D["n_el"] = n
lim = max(abs(axA.get_ylim()[0]), abs(axA.get_ylim()[1]))
xx = np.linspace(-5, 5, 2)
axA.plot(xx, xx, ls="--", lw=1, color=INK2, label="y = x (model-consistent)")
axA.set_xlabel("N(0,1) quantile")
axA.set_ylabel("empirical quantile of r / sigma (per element)")
axA.set_title("A  Normal Q-Q of standardized residual r / sigma")
axA.legend(loc="upper left", fontsize=8.5)
txt = "\n".join(f"{D['label'].split(' (')[0]}: std {D['z_std']:.2f}, robust std {D['z_mad']:.2f}, |z|>3: {100*D['frac_gt3']:.1f}%"
                for D in data.values())
axA.text(0.98, 0.03, txt, transform=axA.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)

# ---- B: shape only: z / std(z) vs N(0,1), with fitted Student-t df ------------
for suf, D in data.items():
    z = np.sort(D["z"] / D["z_std"]); n = len(z)
    p = (np.arange(1, n + 1) - 0.5) / n
    idx = qq_subsample(n)
    th = stats.norm.ppf(p[idx])
    axB.scatter(th, z[idx], s=5, color=D["color"], alpha=0.8, linewidths=0, label=D["label"])
    rng = np.random.default_rng(0)
    sub = rng.choice(D["z"], size=min(200_000, len(D["z"])), replace=False)
    df_fit, loc_fit, sc_fit = stats.t.fit(sub, floc=0.0)
    D["t_df"] = float(df_fit); D["t_scale"] = float(sc_fit)
    D["kurt"] = float(stats.kurtosis(sub))
axB.plot(xx, xx, ls="--", lw=1, color=INK2)
axB.set_xlabel("N(0,1) quantile")
axB.set_ylabel("quantile of (r / sigma) / std")
axB.set_title("B  Same, rescaled by its own std (tail shape only)")
txt = "\n".join(f"{D['label'].split(' (')[0]}: Student-t fit df {D['t_df']:.1f}, excess kurtosis {D['kurt']:.1f}"
                for D in data.values())
axB.text(0.98, 0.03, txt, transform=axB.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
axB.legend(loc="upper left", fontsize=8.5)

# ---- C: per-sample residual RMS / sigma --------------------------------------
allr = np.concatenate([D["ratio"] for D in data.values()])
bins = np.logspace(np.log10(allr.min() * 0.9), np.log10(allr.max() * 1.1), 60)
for suf, D in data.items():
    axC.hist(D["ratio"], bins=bins, histtype="stepfilled", alpha=0.25, color=D["color"], linewidth=0)
    axC.hist(D["ratio"], bins=bins, histtype="step", color=D["color"], linewidth=1.6, label=D["label"])
    q = np.quantile(D["ratio"], [0.1, 0.5, 0.9])
    D["ratio_q"] = q
    axC.axvline(q[1], color=D["color"], lw=1, ls=":")
axC.axvline(1.0, color=INK2, lw=1, ls="--")
d0 = list(data.values())[0]["d"]
band = np.sqrt(stats.chi2.ppf([0.01, 0.99], d0) / d0)
axC.axvspan(band[0], band[1], color=INK2, alpha=0.12, lw=0)
axC.set_xscale("log")
axC.set_xlabel(f"residual RMS / sigma per sample  (dashed: 1; band: Gaussian 1-99% range, d = {int(d0)})")
axC.set_ylabel("samples")
axC.set_title("C  Distribution of residual RMS / sigma per training sample")
txt = "\n".join(f"{D['label'].split(' (')[0]}: q10 {D['ratio_q'][0]:.2f}, median {D['ratio_q'][1]:.2f}, q90 {D['ratio_q'][2]:.2f}"
                for D in data.values())
axC.text(0.98, 0.97, txt, transform=axC.transAxes, ha="right", va="top", fontsize=8, color=INK2)
axC.legend(loc="center right", fontsize=8.5)

# ---- D: per-dim RMS of r/sigma ------------------------------------------------
A = len(list(data.values())[0]["zdim"])
x = np.arange(A)
for k, (suf, D) in enumerate(data.items()):
    axD.scatter(x + (k - (len(data) - 1) / 2) * 0.25, D["zdim"], s=22, color=D["color"], label=D["label"], linewidths=0, zorder=3)
    axD.vlines(x + (k - (len(data) - 1) / 2) * 0.25, 0, D["zdim"], color=D["color"], lw=1.0, alpha=0.5)
axD.axhline(1.0, color=INK2, lw=1, ls="--")
axD.set_yscale("log")
axD.set_xlabel("action dimension (normalized action space, real extent)")
axD.set_ylabel("RMS of r / sigma per dimension")
axD.set_title("D  Per-dimension RMS of r / sigma")
axD.set_xticks(x[::2])
axD.legend(loc="upper left", fontsize=8.5)
hg = data["hg22k"]["zdim"]; top = np.argsort(hg)[-3:]
for t in top:
    axD.annotate(f"dim {t}", (t - (len(data) - 1) / 2 * 0.25, hg[t]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7.5, color=INK2)

n = list(data.values())[0]["n"]
fig.suptitle(f"GR1 / GR00T N1.7 finetunes, residual r / sigma: {n} training samples ({DATA_TAG}), {int(d0)} elements each, eval mode",
             fontsize=11, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.965))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=SURF)
print(f"saved {out}")

for suf, D in data.items():
    print(f"\n== {D['label']}  ckpt={D['ckpt']}  sbias={D['sbias']:.4f}  n_samples={D['n']}  d={D['d']:.0f}")
    if D["loss_eval"] is not None:
        print(f"  model's own per-element NLL on these samples: eval-mode {D['loss_eval']:.3f}  train-mode {D['loss_train']:.3f}")
    print(f"  sigma: median {np.median(D['sigma']):.4f}  q10 {np.quantile(D['sigma'],0.1):.4f}  q90 {np.quantile(D['sigma'],0.9):.4f}")
    print(f"  residual RMS (sqrt m): median {np.median(np.sqrt(D['m'])):.4f}  rms-of-mean {np.sqrt(D['m'].mean()):.4f}")
    print(f"  per-sample RMS/sigma: q10 {D['ratio_q'][0]:.3f} median {D['ratio_q'][1]:.3f} q90 {D['ratio_q'][2]:.3f}  corr(log sigma, log rms) {np.corrcoef(np.log(D['sigma']), np.log(np.sqrt(D['m'])))[0,1]:.3f}")
    print(f"  per-element z: std {D['z_std']:.3f} robust-std {D['z_mad']:.3f} |z|>3 {100*D['frac_gt3']:.2f}%  t-fit df {D['t_df']:.2f} scale {D['t_scale']:.3f} excess kurtosis {D['kurt']:.2f}")
    zd = D["zdim"]; o = np.argsort(zd)[::-1]
    print(f"  per-dim RMS(z): top5 dims {[(int(i), round(float(zd[i]),2)) for i in o[:5]]}  bottom3 {[(int(i), round(float(zd[i]),2)) for i in o[-3:]]}")
