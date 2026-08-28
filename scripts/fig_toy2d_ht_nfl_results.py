"""Visualize HT/HG on the trained 2D MIP no-free-lunch benchmark."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "analysis/toy2d_mip_nfl_hght_exact.json"
HIGH_PATH = ROOT / "analysis/toy2d_mip_nfl_hght_exact_highcap.json"
OUT_PNG = ROOT / "analysis/paper/toy2d_ht_nfl_results.png"
OUT_PDF = ROOT / "analysis/paper/toy2d_ht_nfl_results.pdf"

R = json.loads(INP.read_text())
HIGH = json.loads(HIGH_PATH.read_text()) if HIGH_PATH.exists() else None
MODES = ("skew", "symmetric", "gaussian")
METHODS = ("regression", "hg", "ht", "mip_full")
ERROR_METHODS = ("regression", "hg", "mip_step1", "ht", "mip_full")
NAME = {
    "regression": "L2",
    "hg": "HG",
    "ht": "HT",
    "mip_step1": "MIP step 1",
    "mip_full": "MIP full",
    "skew": "skewed\n80/20",
    "symmetric": "symmetric\n50/50",
    "gaussian": "Gaussian-like\nantithetic",
}
COLOR = {
    "regression": "#2878b5",
    "hg": "#159b88",
    "ht": "#ef8a25",
    "mip_step1": "#8064a2",
    "mip_full": "#d84a3a",
}
DOCK_TOL = R["constants"]["dock_tol_mm"]


def cells(mode):
    return [c for c in R["cells"] if c["mode"] == mode]


def vals(mode, method, key):
    return np.array([c["methods"][method][key] for c in cells(mode)], float)


def mean_std(mode, method, key):
    v = vals(mode, method, key)
    return v.mean(), v.std()


fig = plt.figure(figsize=(15.2, 9.7), facecolor="white")
gs = fig.add_gridspec(
    2, 2, left=0.07, right=0.98, bottom=0.085, top=0.875,
    hspace=0.38, wspace=0.27,
)
fig.suptitle(
    "HT does not rescue MIP under skewed zero-mean labels: both become mode-seeking",
    fontsize=15.5, weight="bold", y=0.965,
)
fig.text(
    0.5, 0.917,
    "TRAINED MODELS — 8 seeds/cell; HT = repository multivariate Student-t NLL "
    "(ν=2, learned state scale); HG is its Gaussian control",
    ha="center", fontsize=10.4, color="#333333",
)

rng = np.random.default_rng(7)
x = np.arange(len(MODES))
w = 0.19

# A. Thresholded task success.
ax = fig.add_subplot(gs[0, 0])
for j, method in enumerate(METHODS):
    xpos = x + (j - 1.5) * w
    means = [vals(mode, method, "success_rate").mean() for mode in MODES]
    ax.bar(xpos, means, width=w * 0.90, color=COLOR[method], alpha=0.86,
           label=NAME[method], zorder=2)
    for i, mode in enumerate(MODES):
        v = vals(mode, method, "success_rate")
        jitter = rng.uniform(-0.035, 0.035, len(v))
        ax.scatter(np.full(len(v), xpos[i]) + jitter, v, s=18, color="black",
                   alpha=0.50, zorder=4)
ax.axhline(1.0, color="#888888", ls=":", lw=1)
ax.set_xticks(x, [NAME[m] for m in MODES])
ax.set_ylim(-0.04, 1.10)
ax.set_ylabel(f"dock success  ($|y|<{DOCK_TOL:.1f}$ mm)")
ax.set_title("A   Skew makes failure reproducible for both HT and MIP",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.7, ncol=4, loc="lower right")
ax.grid(axis="y", alpha=0.16)

# B. Continuous estimator bias.
ax = fig.add_subplot(gs[0, 1])
for j, method in enumerate(METHODS):
    xpos = x + (j - 1.5) * w
    for i, mode in enumerate(MODES):
        v = vals(mode, method, "first_lateral_residual_mean")
        ax.bar(xpos[i], v.mean(), width=w * 0.90, color=COLOR[method],
               alpha=0.86, zorder=2)
        jitter = rng.uniform(-0.035, 0.035, len(v))
        ax.scatter(np.full(len(v), xpos[i]) + jitter, v, s=18, color="black",
                   alpha=0.50, zorder=4)
for i, mode in enumerate(MODES):
    c = cells(mode)[0]
    for method, key in (("ht", "analytic_ht_first_lateral_bias"),
                        ("mip_full", "analytic_mip_first_lateral_bias")):
        j = METHODS.index(method)
        ax.scatter(x[i] + (j - 1.5) * w, c[key], marker="*", s=125,
                   facecolor="#f4c542", edgecolor="black", linewidth=0.7,
                   zorder=6,
                   label="population prediction" if i == 0 and method == "ht" else None)
ax.axhline(0, color="#777777", lw=1)
ax.set_xticks(x, [NAME[m] for m in MODES])
ax.set_ylabel("first-action lateral residual\n(normalized action units)")
ax.set_title("B   HT and MIP hit the predicted −0.1 mode; HG keeps the mean",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.7, loc="lower right")
ax.grid(axis="y", alpha=0.16)

# C. Continuous task error in the primary cell.
ax = fig.add_subplot(gs[1, 0])
for i, method in enumerate(ERROR_METHODS):
    v = vals("skew", method, "abs_y_goal_p50")
    ax.bar(i, v.mean(), width=0.62, color=COLOR[method], alpha=0.86, zorder=2)
    jitter = rng.uniform(-0.12, 0.12, len(v))
    ax.scatter(np.full(len(v), i) + jitter, v, s=25, color="black",
               alpha=0.58, zorder=4)
    ax.text(i, v.mean() + 0.075, f"{v.mean():.2f}", ha="center",
            fontsize=8.8, weight="bold")
ax.axhline(DOCK_TOL, color="#27863e", lw=1.8, ls="--",
           label=f"dock tolerance = {DOCK_TOL:.1f} mm")
ax.set_xticks(range(len(ERROR_METHODS)), [NAME[m] for m in ERROR_METHODS],
              rotation=8)
ax.set_ylabel("median absolute dock error (mm)")
ax.set_ylim(0, 2.55)
ax.set_title("C   The shared mode bias becomes a two-millimeter control miss",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.7, loc="upper left")
ax.grid(axis="y", alpha=0.16)
if HIGH is not None and len(HIGH.get("cells", [])) == 4:
    hc = HIGH["cells"]
    hs = {m: np.mean([c["methods"][m]["success_rate"] for c in hc])
          for m in ("regression", "hg", "ht", "mip_full")}
    hb = np.mean([c["methods"]["ht"]["first_lateral_residual_mean"] for c in hc])
    ax.text(
        2.0, 1.20,
        "High-capacity check\nW=256, 12k, 4 seeds\n"
        f"SR L2/HG/HT/MIP = {hs['regression']:.0%}/{hs['hg']:.0%}/"
        f"{hs['ht']:.0%}/{hs['mip_full']:.0%}\nHT bias = {hb:+.3f}",
        ha="center", va="center", fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999"),
    )

# D. Representative trajectories.
ax = fig.add_subplot(gs[1, 1])
xx = np.linspace(0, 160, 300)
half = 5.0 + 9.0 * (1.0 - np.clip(xx / 130.0, 0.0, 1.0))
ax.fill_between(xx, -half, half, color="#e9eef2", zorder=0)
ax.plot(xx, half, color="#7c858c", lw=1.4)
ax.plot(xx, -half, color="#7c858c", lw=1.4)
ax.axhline(0, color="#999999", lw=0.9, ls=":")
seed0 = [c for c in cells("skew") if c["seed"] == 0][0]
for method in METHODS:
    for k, tr in enumerate(seed0["methods"][method]["traces"]):
        ax.plot([0.0] + tr["x"], [tr["y0"]] + tr["y"],
                color=COLOR[method], alpha=0.27 if k else 0.96,
                lw=1.15 if k else 2.5,
                label=NAME[method] if k == 0 else None)
ax.add_patch(Rectangle((158.8, -DOCK_TOL), 2.4, 2 * DOCK_TOL,
                       facecolor="#83c995", edgecolor="#26723c", lw=1.2,
                       zorder=5))
ax.scatter([160], [0], marker="*", s=115, color="#26723c", zorder=6)
ax.set_xlim(-2, 164)
ax.set_ylim(-15, 15)
ax.set_xlabel("forward position $x$ (mm)")
ax.set_ylabel("lateral position $y$ (mm)")
ax.set_title("D   Same recovery-covered data, different deployed fields",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.7, loc="upper right")
ax.grid(alpha=0.12)

ht_b, ht_bs = mean_std("skew", "ht", "first_lateral_residual_mean")
mip_b, mip_bs = mean_std("skew", "mip_full", "first_lateral_residual_mean")
fig.text(
    0.5, 0.020,
    f"Skew cell (8 seeds): L2 8/8, HG 7/8, HT 0/8, MIP 0/8; "
    f"bias HT {ht_b:+.3f}±{ht_bs:.3f}, MIP {mip_b:+.3f}±{mip_bs:.3f} "
    "(population prediction −0.100). Symmetry controls return both biases to ≈0.",
    ha="center", fontsize=9.3, color="#333333",
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
