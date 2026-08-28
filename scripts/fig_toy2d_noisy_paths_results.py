"""Results figure for the executed-noise, successful-path benchmark."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "analysis/toy2d_noisy_paths_results.json"
OUT_PNG = ROOT / "analysis/paper/toy2d_noisy_paths_results.png"
OUT_PDF = ROOT / "analysis/paper/toy2d_noisy_paths_results.pdf"
R = json.loads(INP.read_text())
CELLS = R["cells"]
DOCK_TOL = R["constants"]["dock_tol_mm"]

METHODS = ("regression", "hg", "mip_step1", "mip_full", "ht")
NAME = {
    "regression": "L2",
    "hg": "HG",
    "mip_step1": "MIP step 1",
    "mip_full": "MIP full",
    "ht": "HT",
}
COLOR = {
    "regression": "#2878b5",
    "hg": "#159b88",
    "mip_step1": "#8064a2",
    "mip_full": "#d84a3a",
    "ht": "#ef8a25",
}


def vals(method, key):
    return np.array([c["methods"][method][key] for c in CELLS], float)


def half_width(x):
    x = np.asarray(x)
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


fig = plt.figure(figsize=(15.2, 9.6), facecolor="white")
gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, bottom=0.09, top=0.875,
                      hspace=0.38, wspace=0.27)
fig.suptitle(
    "Executed noisy paths remove the binary reversal: MIP and HT both recover precise control",
    fontsize=15.5, weight="bold", y=0.965,
)
fig.text(
    0.5, 0.917,
    "TRAINED MODELS — 8 seeds; 3,000 dynamically valid paths / seed; "
    "120k H=8 chunks; every noisy demonstration succeeds",
    ha="center", fontsize=10.4, color="#333333",
)

# A. Actual data paths.
ax = fig.add_subplot(gs[0, 0])
xx = np.linspace(0, 160, 300)
hw = half_width(xx)
ax.fill_between(xx, -hw, hw, color="#e9eef2", zorder=0)
ax.plot(xx, hw, color="#7c858c", lw=1.5)
ax.plot(xx, -hw, color="#7c858c", lw=1.5)
ax.axhline(0, color="#999999", lw=0.9, ls=":")
for i, path in enumerate(CELLS[0]["data"]["paths"]):
    ax.plot(path["x"], path["y"], color="#d9794a", alpha=0.38, lw=1.25,
            label="executed kick–recovery demo" if i == 0 else None)
ax.add_patch(Rectangle((158.5, -DOCK_TOL), 3.0, 2 * DOCK_TOL,
                       facecolor="#83c995", edgecolor="#27863e", lw=1.3,
                       zorder=6))
ax.scatter([160], [0], marker="*", s=110, color="#27863e", zorder=7)
ax.text(106, -10.8,
        "data-policy validity\n3,000/3,000 = 100% SR\n0 collisions",
        ha="center", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#90979c"))
ax.set_xlim(-2, 164)
ax.set_ylim(-15, 15)
ax.set_xlabel("forward position $x$ (mm)")
ax.set_ylabel("lateral position $y$ (mm)")
ax.set_title("A   Training data are real successful noisy trajectories",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.7, loc="upper right")
ax.grid(alpha=0.11)

# B. Binary success.
ax = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(21)
for i, method in enumerate(METHODS):
    v = vals(method, "success_rate")
    ax.bar(i, v.mean(), width=0.66, color=COLOR[method], alpha=0.86, zorder=2)
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v,
               color="black", alpha=0.58, s=24, zorder=4)
    ax.text(i, 1.025, "8/8", ha="center", fontsize=9.3, weight="bold")
ax.axhline(1, color="#888888", lw=1, ls=":")
ax.set_xticks(range(len(METHODS)), [NAME[m] for m in METHODS], rotation=8)
ax.set_ylim(-0.03, 1.10)
ax.set_ylabel(f"dock success  ($|y|<{DOCK_TOL:.1f}$ mm)")
ax.set_title("B   All objectives learn a successful recovery policy",
             loc="left", fontsize=11.5, weight="bold")
ax.grid(axis="y", alpha=0.16)

# C. Action at the exactly shared pre-kick states.
ax = fig.add_subplot(gs[1, 0])
for i, method in enumerate(METHODS):
    v = vals(method, "branch_state_lateral_action_mean")
    ax.bar(i, v.mean(), width=0.66, color=COLOR[method], alpha=0.86, zorder=2)
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v,
               color="black", alpha=0.58, s=24, zorder=4)
    ax.text(i, v.mean() + 0.007, f"{v.mean():+.3f}", ha="center",
            fontsize=8.8, weight="bold")
ax.axhline(0.0, color="#27863e", lw=1.6, ls="--", label="clean branch = 0")
ax.axhline(0.2, color="#2878b5", lw=1.4, ls=":", label="raw kick-label mean = +0.2")
ax.set_xticks(range(len(METHODS)), [NAME[m] for m in METHODS], rotation=8)
ax.set_ylim(-0.015, 0.225)
ax.set_ylabel("lateral action at shared pre-kick states\n(normalized units)")
ax.set_title("C   Full MIP and HT suppress the optional kick",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.7, loc="upper right")
ax.grid(axis="y", alpha=0.16)

# D. Continuous docking precision.
ax = fig.add_subplot(gs[1, 1])
for i, method in enumerate(METHODS):
    v = vals(method, "abs_y_goal_p50")
    ax.bar(i, v.mean(), width=0.66, color=COLOR[method], alpha=0.86, zorder=2)
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v,
               color="black", alpha=0.58, s=24, zorder=4)
    ax.text(i, v.mean() + 0.014, f"{v.mean():.3f}", ha="center",
            fontsize=8.8, weight="bold")
ax.axhline(DOCK_TOL, color="#27863e", lw=1.8, ls="--",
           label=f"dock tolerance = {DOCK_TOL:.1f} mm")
ax.set_xticks(range(len(METHODS)), [NAME[m] for m in METHODS], rotation=8)
ax.set_ylim(0, 0.56)
ax.set_ylabel("median absolute dock error (mm)")
ax.set_title("D   MIP and HT are the precision co-winners",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.7, loc="upper right")
ax.grid(axis="y", alpha=0.16)

l2 = vals("regression", "abs_y_goal_p50").mean()
mip = vals("mip_full", "abs_y_goal_p50").mean()
ht = vals("ht", "abs_y_goal_p50").mean()
fig.text(
    0.5, 0.022,
    f"Eight-seed mean dock error: L2 {l2:.3f} mm, full MIP {mip:.3f} mm, "
    f"HT {ht:.3f} mm. Full MIP is {l2 / mip:.1f}× more precise than L2, "
    "but SR is tied because every method remains inside the 0.5-mm dock.",
    ha="center", fontsize=9.3, color="#333333",
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
