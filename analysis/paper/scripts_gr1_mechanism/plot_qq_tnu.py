"""Q-Q of the GR1 HG / HT residuals against Student-t references at two levels.

Level 1 (per element): z = r / sigma_i vs t_nu quantiles.  Slope 1 would mean
  r ~ sigma_i * t_nu element-wise.  Core slope fitted on 25-75%, tail slope on
  0.1-2% and 98-99.9%; they agree only if the Q-Q is straight.
Level 2 (per chunk, the model's own likelihood level): under a multivariate t
  with nu dof and scale sigma_i on the d=232 masked elements, u = m / sigma^2
  (m = mean squared residual over the chunk) is distributed as F(d, nu).  We plot
  log u against log F(d, nu) quantiles: slope 1 and intercept 0 means the chunk
  residual IS a scale-sigma multivariate t with that nu.  The slope-1 nu is the
  data's own chunk-level nu.

Usage: python plot_qq_tnu.py <dump_dir> <out.png> [file pattern with {suf}] [data tag]
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dump_dir = Path(sys.argv[1]); out = Path(sys.argv[2])
PATTERN = sys.argv[3] if len(sys.argv) > 3 else "resid_dump3_gr1_{suf}_all.npz"
DATA_TAG = sys.argv[4] if len(sys.argv) > 4 else "all 24 GR1 training datasets"

ARMS = [("HG (Gaussian NLL, 22k)", "hg22k", "#2a78d6", np.inf),
        ("HT (Student-t nu=464, 60k)", "ht60k", "#eb6834", 464.0),
        ("HT c=2 (Student-t nu=928, 60k)", "c2_60k", "#1baf7a", 928.0)]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titlesize": 9.6, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True})

D_EFF = 232
# element-level probability grid (dense tails) and chunk-level grid (2400 chunks -> 0.5%..99.5%)
P = np.unique(np.concatenate([np.logspace(-4, -1, 400), np.linspace(0.1, 0.9, 400), 1 - np.logspace(-4, -1, 400)]))
CORE = (P >= 0.25) & (P <= 0.75)
TAIL = ((P >= 0.001) & (P <= 0.02)) | ((P >= 0.98) & (P <= 0.999))
PC = np.linspace(0.005, 0.995, 300)
NUS = np.logspace(0, 3, 61)          # 1 .. 1000


def tq(p, nu):
    return stats.norm.ppf(p) if not np.isfinite(nu) else stats.t.ppf(p, nu)


def fq_log(p, nu):
    """log-quantiles of F(d_eff, nu); nu=inf -> chi2_d/d."""
    if not np.isfinite(nu):
        return np.log(stats.chi2.ppf(p, D_EFF) / D_EFF)
    return np.log(stats.f.ppf(p, D_EFF, nu))


def fit(x, y, sel):
    b, a = np.polyfit(x[sel], y[sel], 1)
    res = y[sel] - (a + b * x[sel])
    return b, a, 1 - res.var() / y[sel].var()


def crossing(nus, vals, target=1.0):
    """first nu (ascending) where vals crosses target, linear interp in log nu."""
    s = np.sign(vals - target)
    idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    t = (target - vals[i]) / (vals[i + 1] - vals[i])
    return float(np.exp(np.log(nus[i]) + t * (np.log(nus[i + 1]) - np.log(nus[i]))))


data = {}
for label, suf, color, nu_model in ARMS:
    if not (dump_dir / PATTERN.format(suf=suf)).exists():
        print(f"skip {label}: no {PATTERN.format(suf=suf)}"); continue
    z = np.load(dump_dir / PATTERN.format(suf=suf))
    sig = z["sigma"]; m = z["m"]
    zz = (z["r"] / sig[:, None, None]).reshape(-1); zz = zz[np.isfinite(zz)]
    rng = np.random.default_rng(0)
    nu_ml, _, sc_ml = stats.t.fit(rng.choice(zz, min(200_000, len(zz)), replace=False), floc=0.0)
    D = dict(label=label, color=color, nu_model=nu_model, n=len(zz), nu_ml=float(nu_ml), sc_ml=float(sc_ml),
             eq=np.quantile(zz, P), lu=np.log(np.quantile(m / sig ** 2, PC)), u=m / sig ** 2)
    # element-level sweep: core slope, tail slope
    D["el"] = np.array([(fit(tq(P, nu), D["eq"], CORE)[0], fit(tq(P, nu), D["eq"], TAIL)[0]) for nu in NUS])
    D["nu_el_eq"] = crossing(NUS, D["el"][:, 1] / D["el"][:, 0])       # tail slope == core slope
    # chunk-level sweep (log-log): slope, intercept, R2
    D["ch"] = np.array([fit(fq_log(PC, nu), D["lu"], np.ones_like(PC, bool)) for nu in NUS])
    D["nu_ch_s1"] = crossing(NUS, D["ch"][:, 0])                        # slope == 1
    D["nu_ch_lin"] = float(NUS[np.argmax(D["ch"][:, 2])])               # straightest
    data[suf] = D

fig, axes = plt.subplots(len(data), 4, figsize=(15, 3.9 * len(data)), dpi=170, squeeze=False)
for row, (suf, D) in enumerate(data.items()):
    arm = D["label"].split(" (")[0]
    own = "N(0,1)" if not np.isfinite(D["nu_model"]) else f"t, nu={D['nu_model']:.0f}"
    # --- element level: own model, and the tail=core nu
    for col, (name, nu) in enumerate([(f"own model, {own}", D["nu_model"]),
                                      (f"t_{D['nu_el_eq']:.2f} (tail slope = core slope)", D["nu_el_eq"])]):
        ax = axes[row, col]
        x, y = tq(P, nu), D["eq"]
        bc, ac, _ = fit(x, y, CORE); bt, at, _ = fit(x, y, TAIL)
        ax.scatter(x, y, s=5, color=D["color"], linewidths=0, alpha=0.85)
        lim = np.abs(x[(P >= 0.001) & (P <= 0.999)]).max()
        xx = np.array([-lim, lim])
        ax.plot(xx, xx, ls="--", lw=1, color=INK2, label="y = x")
        ax.plot(xx, ac + bc * xx, ls=":", lw=1.3, color=INK, label=f"core fit, slope {bc:.2f}")
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        yl = np.abs(y[(P >= 0.001) & (P <= 0.999)]).max() * 1.05
        ax.set_ylim(-yl, yl)
        ax.set_title(f"{arm} per element vs {name}", fontsize=9)
        ax.set_xlabel(("N(0,1)" if not np.isfinite(nu) else f"t_{nu:.3g}") + " quantile")
        ax.set_ylabel("empirical quantile of r / sigma")
        ax.text(0.03, 0.97, f"core slope {bc:.2f}\ntail slope {bt:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK2)
        ax.legend(loc="lower right", fontsize=7.5)
    # --- chunk level: own model, and slope-1 nu
    own_ch = "chi2/d (own model)" if not np.isfinite(D["nu_model"]) else f"F(232, {D['nu_model']:.0f}) (own model)"
    for col, (name, nu) in enumerate([(own_ch, D["nu_model"]),
                                      (f"F(232, {D['nu_ch_s1']:.1f}) (slope = 1)", D["nu_ch_s1"])], start=2):
        ax = axes[row, col]
        x, y = fq_log(PC, nu), D["lu"]
        b, a, r2 = fit(x, y, np.ones_like(PC, bool))
        ax.scatter(x, y, s=7, color=D["color"], linewidths=0, alpha=0.9)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        xx = np.array([lo, hi])
        ax.plot(xx, xx, ls="--", lw=1, color=INK2, label="y = x")
        ax.plot(xx, a + b * xx, ls=":", lw=1.3, color=INK, label=f"fit, slope {b:.2f}, intercept {a:+.2f}")
        ax.set_title(f"{arm} per chunk: u vs {name}", fontsize=9)
        ax.set_xlabel("log quantile of reference F(232, nu)")
        ax.set_ylabel("log quantile of u (2400 chunks)")
        ax.text(0.03, 0.97, f"slope {b:.2f}\nintercept {a:+.2f}\nR² {r2:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK2)
        ax.legend(loc="lower right", fontsize=7.5)
fig.suptitle(f"GR1 residual vs Student-t references.  Left pair: per element r / sigma ({data['hg22k']['n']:,} values per arm; axes 0.1-99.9%).  "
             f"Right pair: per chunk u = m / sigma² vs F(232, nu), log axes, 0.5-99.5%.  {DATA_TAG}", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 1 - 0.08 / len(data)))
fig.savefig(out, facecolor=SURF); print(f"saved {out}")

# sweep figure
fig2, axs = plt.subplots(1, 4, figsize=(16, 3.8), dpi=170)
for suf, D in data.items():
    axs[0].plot(NUS, D["el"][:, 0], color=D["color"], lw=2, label=D["label"] + " core"); axs[0].plot(NUS, D["el"][:, 1], color=D["color"], lw=1.2, ls="--", label=D["label"] + " tail")
    axs[1].plot(NUS, D["el"][:, 1] / D["el"][:, 0], color=D["color"], lw=2, label=D["label"]); axs[1].axvline(D["nu_el_eq"], color=D["color"], lw=1, ls=":")
    axs[2].plot(NUS, D["ch"][:, 0], color=D["color"], lw=2, label=D["label"]); axs[2].axvline(D["nu_ch_s1"], color=D["color"], lw=1, ls=":")
    axs[3].plot(NUS, D["ch"][:, 2], color=D["color"], lw=2, label=D["label"]); axs[3].axvline(D["nu_ch_lin"], color=D["color"], lw=1, ls=":")
for ax_ in axs:
    ax_.set_xscale("log"); ax_.set_xlabel("nu of the reference"); ax_.legend(fontsize=7.5, loc="best")
axs[0].axhline(1, color=INK2, lw=1, ls="--"); axs[0].set_yscale("log"); axs[0].set_ylabel("Q-Q slope"); axs[0].set_title("A  Per element: core and tail slope vs t_nu")
axs[1].axhline(1, color=INK2, lw=1, ls="--"); axs[1].set_yscale("log"); axs[1].set_ylabel("tail slope / core slope"); axs[1].set_title("B  Per element: straightness (1 = straight)")
axs[2].axhline(1, color=INK2, lw=1, ls="--"); axs[2].set_ylabel("log-log Q-Q slope"); axs[2].set_title("C  Per chunk: slope of log u vs log F(232, nu)")
axs[3].set_ylabel("R² of log-log Q-Q"); axs[3].set_title("D  Per chunk: linearity (dotted: best nu)")
fig2.tight_layout()
out2 = out.with_name(out.stem + "_sweep" + out.suffix)
fig2.savefig(out2, facecolor=SURF); print(f"saved {out2}")

for suf, D in data.items():
    print(f"\n== {D['label']}: n_el={D['n']:,}  element ML t-fit nu={D['nu_ml']:.2f} scale={D['sc_ml']:.3f}")
    print(f"   element level: tail slope == core slope at nu={D['nu_el_eq']:.2f}")
    for name, nu in (("own", D["nu_model"]), ("1.5", 1.5), ("2", 2.0), ("3", 3.0), ("5", 5.0), ("10", 10.0)):
        x = tq(P, nu); print(f"     t_{name:>4}: core slope {fit(x, D['eq'], CORE)[0]:.3f}  tail slope {fit(x, D['eq'], TAIL)[0]:.3f}")
    print(f"   chunk level (u = m/sigma^2 vs F(232, nu), log-log): slope=1 at nu={D['nu_ch_s1']:.2f}; straightest nu={D['nu_ch_lin']:.2f} (R² {D['ch'][:,2].max():.4f}); median u {np.median(D['u']):.3f}")
    for name, nu in (("own", D["nu_model"]), ("2", 2.0), ("3", 3.0), ("4", 4.0), ("5", 5.0), ("6", 6.0), ("8", 8.0), ("12", 12.0), ("30", 30.0), ("464", 464.0)):
        b, a, r2 = fit(fq_log(PC, nu), D["lu"], np.ones_like(PC, bool)); print(f"     F(232,{name:>4}): slope {b:.3f} intercept {a:+.3f} R² {r2:.4f}")
    # KS distance of u against F(232, nu) at the slope-1 nu and at the model nu
    for name, nu in (("slope-1 nu", D["nu_ch_s1"]), ("model nu", D["nu_model"])):
        if np.isfinite(nu):
            ks = stats.kstest(D["u"], lambda q: stats.f.cdf(q, D_EFF, nu)).statistic
            print(f"     KS distance of u vs F(232, {nu:.2f}): {ks:.3f}")
