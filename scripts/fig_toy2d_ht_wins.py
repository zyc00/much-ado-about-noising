"""Figure for the 2D contamination cell where HT beats L2 and MIP."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "analysis/toy2d_ht_wins_results.json"
OUT_PNG = ROOT / "analysis/paper/toy2d_ht_wins.png"
OUT_PDF = ROOT / "analysis/paper/toy2d_ht_wins.pdf"
R = json.loads(INP.read_text())
CELLS = R["cells"]
DOCK_TOL = R["constants"]["dock_tol_mm"]

METHODS = ("regression", "hg", "mip_step1", "mip_full", "ht")
ROLLOUT_METHODS = ("regression", "hg", "mip_full", "ht")
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


fig = plt.figure(figsize=(15.2, 9.7), facecolor="white")
gs = fig.add_gridspec(
    2, 2, left=0.07, right=0.98, bottom=0.09, top=0.875,
    hspace=0.38, wspace=0.27,
)
fig.suptitle(
    "Winner reversal: HT selects the clean branch while L2 and MIP follow contamination",
    fontsize=15.5, weight="bold", y=0.965,
)
fig.text(
    0.5, 0.917,
    "TRAINED 2D CONTROL MODELS — 8 seeds; 30k recovery-covered chunks; "
    "70% clean labels + 20% mild and 10% gross same-direction corruption",
    ha="center", fontsize=10.4, color="#333333",
)

# A. Data geometry and population estimators.
ax = fig.add_subplot(gs[0, 0])
atoms = np.array([0.0, 0.2, 1.6])
prob = np.array([0.7, 0.2, 0.1])
atom_colors = ["#4daf4a", "#e98973", "#a94442"]
ax.bar(atoms, prob, width=[0.09, 0.09, 0.09], color=atom_colors,
       edgecolor="white", linewidth=1.2, zorder=3)
for x, p, label in zip(atoms, prob, ("clean", "mild corrupt", "gross corrupt")):
    ax.text(x, p + 0.025, f"{p:.0%}\n{label}", ha="center", va="bottom",
            fontsize=9, weight="bold" if x == 0 else "normal")
mean = float((atoms * prob).sum())
mip_population = CELLS[0]["analytic_mip_first_lateral_bias"]
ax.axvline(0, color=COLOR["ht"], lw=2.5, zorder=5)
ax.axvline(mip_population, color=COLOR["mip_full"], lw=2.1, ls="--", zorder=5)
ax.axvline(mean, color=COLOR["regression"], lw=2.1, ls=":", zorder=5)
ax.annotate("HT / clean target = 0", xy=(0, 0.55), xytext=(0.32, 0.73),
            arrowprops=dict(arrowstyle="->", color=COLOR["ht"], lw=1.5),
            color=COLOR["ht"], fontsize=9.5, weight="bold")
ax.annotate(f"MIP population ≈ {mip_population:.3f}",
            xy=(mip_population, 0.28), xytext=(0.48, 0.50),
            arrowprops=dict(arrowstyle="->", color=COLOR["mip_full"], lw=1.4),
            color=COLOR["mip_full"], fontsize=9.3, weight="bold")
ax.annotate(f"L2 / HG mean = {mean:.1f}", xy=(mean, 0.17), xytext=(0.58, 0.33),
            arrowprops=dict(arrowstyle="->", color=COLOR["regression"], lw=1.4),
            color=COLOR["regression"], fontsize=9.3, weight="bold")
ax.set_xlim(-0.10, 1.75)
ax.set_ylim(0, 0.88)
ax.set_xlabel("demonstrated first-action residual")
ax.set_ylabel("probability")
ax.set_title("A   Put the label mean on a corrupt branch",
             loc="left", fontsize=11.5, weight="bold")
ax.grid(axis="y", alpha=0.16)

# B. Closed-loop success.
ax = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(11)
for i, method in enumerate(METHODS):
    v = vals(method, "success_rate")
    ax.bar(i, v.mean(), width=0.66, color=COLOR[method], alpha=0.86, zorder=2)
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v,
               color="black", alpha=0.58, s=25, zorder=4)
    n_pass = int(np.sum(v == 1.0))
    ax.text(i, v.mean() + 0.045, f"{n_pass}/8", ha="center", fontsize=9.2,
            weight="bold")
ax.axhline(1.0, color="#888888", ls=":", lw=1)
ax.set_xticks(range(len(METHODS)), [NAME[m] for m in METHODS], rotation=8)
ax.set_ylim(-0.04, 1.12)
ax.set_ylabel(f"dock success  ($|y|<{DOCK_TOL:.1f}$ mm)")
ax.set_title("B   HT is the only successful estimator",
             loc="left", fontsize=11.5, weight="bold")
ax.grid(axis="y", alpha=0.16)

# C. Learned deployed field vs population targets.
ax = fig.add_subplot(gs[1, 0])
analytic = {
    "regression": mean,
    "hg": mean,
    "mip_step1": mean,
    "mip_full": mip_population,
    "ht": 0.0,
}
for i, method in enumerate(METHODS):
    v = vals(method, "first_lateral_residual_mean")
    ax.bar(i, v.mean(), width=0.66, color=COLOR[method], alpha=0.86, zorder=2)
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v,
               color="black", alpha=0.58, s=25, zorder=4)
    ax.scatter(i, analytic[method], marker="*", s=150, facecolor="#f4c542",
               edgecolor="black", linewidth=0.7, zorder=6,
               label="population target" if i == 0 else None)
    ax.text(i, v.mean() + 0.013, f"{v.mean():+.3f}", ha="center", fontsize=8.8,
            weight="bold")
ax.axhline(0, color="#777777", lw=1)
ax.set_xticks(range(len(METHODS)), [NAME[m] for m in METHODS], rotation=8)
ax.set_ylim(-0.025, 0.235)
ax.set_ylabel("learned first-action lateral residual")
ax.set_title("C   HT rejects both contaminant scales and recovers zero",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.7, loc="upper right")
ax.grid(axis="y", alpha=0.16)

# D. Representative trajectories.
ax = fig.add_subplot(gs[1, 1])
xx = np.linspace(0, 160, 300)
half = 5.0 + 9.0 * (1.0 - np.clip(xx / 130.0, 0.0, 1.0))
ax.fill_between(xx, -half, half, color="#e9eef2", zorder=0)
ax.plot(xx, half, color="#7c858c", lw=1.4)
ax.plot(xx, -half, color="#7c858c", lw=1.4)
ax.axhline(0, color="#999999", lw=0.9, ls=":")
seed0 = [c for c in CELLS if c["seed"] == 0][0]
for method in ROLLOUT_METHODS:
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
ax.set_title("D   Robust branch commitment is correct in this data regime",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.7, loc="upper right")
ax.grid(alpha=0.12)

ht_b = vals("ht", "first_lateral_residual_mean")
l2_e = vals("regression", "abs_y_goal_p50")
mip_e = vals("mip_full", "abs_y_goal_p50")
ht_e = vals("ht", "abs_y_goal_p50")
fig.text(
    0.5, 0.022,
    f"8 seeds: HT bias {ht_b.mean():+.4f}±{ht_b.std():.4f}, dock error "
    f"{np.nanmean(ht_e):.2f} mm; L2 {np.nanmean(l2_e):.2f} mm; "
    f"full MIP {np.nanmean(mip_e):.2f} mm. "
    "Same HT mode-seeking inductive bias as before—opposite outcome because the majority mode is task-correct.",
    ha="center", fontsize=9.2, color="#333333",
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
