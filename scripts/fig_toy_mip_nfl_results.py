"""Plot the trained-model results of scripts/toy2d_mip_nfl.py."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "analysis/toy2d_mip_nfl_results.json"
OUT_PNG = ROOT / "analysis/paper/toy_mip_nfl_results.png"
OUT_PDF = ROOT / "analysis/paper/toy_mip_nfl_results.pdf"

R = json.loads(INP.read_text())
HIGH_PATH = ROOT / "analysis/toy2d_mip_nfl_highcap.json"
HIGH = json.loads(HIGH_PATH.read_text()) if HIGH_PATH.exists() else None
MODES = ("skew", "symmetric", "gaussian")
METHODS = ("regression", "mip_step1", "mip_full")
NAME = {
    "regression": "Regression",
    "mip_step1": "MIP step 1",
    "mip_full": "MIP full",
    "skew": "skewed\n80/20",
    "symmetric": "symmetric\n50/50",
    "gaussian": "Gaussian-like\nantithetic",
}
COLOR = {
    "regression": "#2878b5",
    "mip_step1": "#7b5aa6",
    "mip_full": "#d84a3a",
}
DOCK_TOL = R["constants"]["dock_tol_mm"]


def cells(mode):
    return [c for c in R["cells"] if c["mode"] == mode]


def vals(mode, method, key):
    return np.array([c["methods"][method][key] for c in cells(mode)], float)


fig = plt.figure(figsize=(14.8, 9.4), facecolor="white")
gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, bottom=0.08, top=0.88,
                      hspace=0.36, wspace=0.27)
fig.suptitle(
    "2D no-free-lunch result: MIP's second-step denoiser turns skewed "
    "zero-mean nuisance into control bias",
    fontsize=15.5,
    weight="bold",
    y=0.965,
)
fig.text(
    0.5,
    0.92,
    "TRAINED MODELS — 8 seeds per cell, identical regression/MIP network class, "
    "30k recovery-covered state-action chunks, H=8",
    ha="center",
    fontsize=10.5,
    color="#333333",
)

# A. Success rates across the primary and falsifier cells.
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(MODES))
w = 0.23
rng = np.random.default_rng(1)
for j, method in enumerate(METHODS):
    means = [vals(mode, method, "success_rate").mean() for mode in MODES]
    xpos = x + (j - 1) * w
    ax.bar(xpos, means, width=w * 0.92, color=COLOR[method], alpha=0.83,
           label=NAME[method], zorder=2)
    for i, mode in enumerate(MODES):
        v = vals(mode, method, "success_rate")
        jitter = rng.uniform(-0.045, 0.045, len(v))
        ax.scatter(np.full(len(v), xpos[i]) + jitter, v, s=20, color="black",
                   alpha=0.55, zorder=4)
ax.axhline(1.0, color="#888888", ls=":", lw=1)
ax.set_xticks(x, [NAME[m] for m in MODES])
ax.set_ylim(-0.03, 1.10)
ax.set_ylabel(f"dock success  ($|y|<{DOCK_TOL:.1f}$ mm)")
ax.set_title("A   The reversal exists only under label skew",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower right")
ax.grid(axis="y", alpha=0.16)

# B. Learned field bias vs population prediction.
ax = fig.add_subplot(gs[0, 1])
for j, method in enumerate(METHODS):
    xpos = x + (j - 1) * w
    for i, mode in enumerate(MODES):
        v = vals(mode, method, "first_lateral_residual_mean")
        ax.bar(xpos[i], v.mean(), width=w * 0.92, color=COLOR[method], alpha=0.83,
               zorder=2)
        jitter = rng.uniform(-0.045, 0.045, len(v))
        ax.scatter(np.full(len(v), xpos[i]) + jitter, v, s=20, color="black",
                   alpha=0.55, zorder=4)
for i, mode in enumerate(MODES):
    analytic = cells(mode)[0]["analytic_mip_first_lateral_bias"]
    ax.scatter(x[i] + w, analytic, marker="*", s=175, facecolor="#f4c542",
               edgecolor="black", linewidth=0.8, zorder=6,
               label="population MIP prediction" if i == 0 else None)
ax.axhline(0, color="#777777", lw=1)
ax.set_xticks(x, [NAME[m] for m in MODES])
ax.set_ylabel("first-action lateral residual\n(normalized action units)")
ax.set_title("B   Full MIP learns the predicted mean→mode shift",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.grid(axis="y", alpha=0.16)

# C. Continuous causal readout on the primary cell.
ax = fig.add_subplot(gs[1, 0])
xm = np.arange(len(METHODS))
for i, method in enumerate(METHODS):
    v = vals("skew", method, "abs_y_goal_p50")
    ax.bar(i, v.mean(), width=0.58, color=COLOR[method], alpha=0.83, zorder=2)
    jitter = rng.uniform(-0.11, 0.11, len(v))
    ax.scatter(np.full(len(v), i) + jitter, v, s=28, color="black", alpha=0.62,
               zorder=4)
    ax.text(i, v.mean() + 0.08, f"{v.mean():.2f} mm", ha="center", fontsize=9,
            weight="bold")
ax.axhline(DOCK_TOL, color="#27863e", lw=1.8, ls="--",
           label=f"dock tolerance = {DOCK_TOL:.1f} mm")
ax.set_xticks(xm, [NAME[m] for m in METHODS])
ax.set_ylabel("median absolute lateral error at dock (mm)")
ax.set_ylim(0, 2.55)
ax.set_title("C   Same MIP weights: step 1 nearly centers; step 2 creates the miss",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.grid(axis="y", alpha=0.16)
if HIGH is not None:
    hc = HIGH["cells"]
    hsr = [np.mean([c["methods"][m]["success_rate"] for c in hc])
           for m in METHODS]
    hb = np.mean([c["methods"]["mip_full"]["first_lateral_residual_mean"]
                  for c in hc])
    ax.text(
        1.95,
        1.20,
        "High-capacity check\n"
        "W=256, 12k, 4 seeds\n"
        f"SR reg/step1/full = {hsr[0]:.0%}/{hsr[1]:.0%}/{hsr[2]:.0%}\n"
        f"full bias = {hb:+.3f}",
        ha="right",
        va="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999"),
    )

# D. Representative closed-loop rollouts in the skewed cell.
ax = fig.add_subplot(gs[1, 1])
xx = np.linspace(0, 160, 300)
half = 5.0 + 9.0 * (1.0 - np.clip(xx / 130.0, 0.0, 1.0))
ax.fill_between(xx, -half, half, color="#e9eef2", zorder=0)
ax.plot(xx, half, color="#7c858c", lw=1.4)
ax.plot(xx, -half, color="#7c858c", lw=1.4)
ax.axhline(0, color="#999999", lw=0.9, ls=":")
seed0 = [c for c in cells("skew") if c["seed"] == 0][0]
for method in METHODS:
    traces = seed0["methods"][method]["traces"]
    for k, tr in enumerate(traces):
        tx = [0.0] + tr["x"]
        ty = [tr["y0"]] + tr["y"]
        ax.plot(tx, ty, color=COLOR[method], alpha=0.32 if k else 0.95,
                lw=1.2 if k else 2.4, label=NAME[method] if k == 0 else None)
ax.add_patch(Rectangle((158.8, -DOCK_TOL), 2.4, 2 * DOCK_TOL,
                       facecolor="#83c995", edgecolor="#26723c", lw=1.2,
                       zorder=5))
ax.scatter([160], [0], marker="*", s=115, color="#26723c", zorder=6)
ax.set_xlim(-2, 164)
ax.set_ylim(-15, 15)
ax.set_xlabel("forward position $x$ (mm)")
ax.set_ylabel("lateral position $y$ (mm)")
ax.set_title("D   Representative skew-cell rollouts (seed 0)",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.grid(alpha=0.12)

# Compact numerical statement kept on the figure to prevent over-reading SR.
skew_full = vals("skew", "mip_full", "first_lateral_residual_mean")
skew_reg_err = vals("skew", "regression", "abs_y_goal_p50")
skew_step_err = vals("skew", "mip_step1", "abs_y_goal_p50")
skew_full_err = vals("skew", "mip_full", "abs_y_goal_p50")
fig.text(
    0.5,
    0.018,
    f"Skew cell: full-MIP bias {skew_full.mean():+.3f}±{skew_full.std():.3f} "
    f"(analytic −0.100); dock error Regression {skew_reg_err.mean():.2f}, "
    f"MIP-step1 {skew_step_err.mean():.2f}, full MIP {skew_full_err.mean():.2f} mm.  "
    "Symmetric and Gaussian-like controls: 100% for all methods in all seeds.",
    ha="center",
    fontsize=9.3,
    color="#333333",
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
