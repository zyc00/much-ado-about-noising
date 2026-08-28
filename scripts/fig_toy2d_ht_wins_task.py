"""Task schematic for the 2D contamination cell where HT wins."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "analysis/paper/toy2d_ht_wins_task.png"
OUT_PDF = ROOT / "analysis/paper/toy2d_ht_wins_task.pdf"

BLUE = "#2878b5"
GREEN = "#27863e"
ORANGE = "#ef8a25"
RED = "#d84a3a"
PURPLE = "#8064a2"
GREY = "#65717a"


def funnel(ax, goal=True):
    xx = np.linspace(0, 160, 300)
    half = 5.0 + 9.0 * (1.0 - np.clip(xx / 130.0, 0.0, 1.0))
    ax.fill_between(xx, -half, half, color="#e9eef2", zorder=0)
    ax.plot(xx, half, color="#7c858c", lw=1.6)
    ax.plot(xx, -half, color="#7c858c", lw=1.6)
    ax.axhline(0, color="#a0a7ac", lw=1, ls=":")
    if goal:
        ax.add_patch(Rectangle((158.5, -0.5), 3.0, 1.0, facecolor="#83c995",
                               edgecolor=GREEN, lw=1.4, zorder=6))
        ax.scatter([160], [0], marker="*", s=120, color=GREEN, zorder=7)
        ax.annotate("tight dock\n$|y|<0.5$ mm", (160, 0), (133, 8.6),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
                    color=GREEN, ha="center", fontsize=9.5, weight="bold")
    ax.set_xlim(-4, 165)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("forward position $x$ (mm)")
    ax.set_ylabel("lateral position $y$ (mm)")
    ax.grid(alpha=0.10)


fig = plt.figure(figsize=(15.8, 5.8), facecolor="white")
gs = fig.add_gridspec(1, 3, left=0.055, right=0.98, bottom=0.15, top=0.80,
                      wspace=0.28, width_ratios=(1.2, 0.9, 1.2))
fig.suptitle(
    "2D robot-like winner-reversal task: the clean action is the majority branch, not the label mean",
    fontsize=15.5, weight="bold", y=0.96,
)
fig.text(
    0.5, 0.865,
    "A point robot replans an 8-action correction chunk while traversing a narrowing funnel; "
    "training covers recovery states, but action labels are contaminated",
    ha="center", fontsize=10.3, color="#333333",
)

# A. Physical task and recovery-covered state distribution.
ax = fig.add_subplot(gs[0, 0])
funnel(ax)
rng = np.random.default_rng(4)
xs = rng.uniform(4, 152, 170)
half = 5.0 + 9.0 * (1.0 - np.clip(xs / 130.0, 0.0, 1.0))
ys = rng.uniform(-0.78 * half, 0.78 * half)
ax.scatter(xs, ys, s=8, color=BLUE, alpha=0.18, zorder=1)
ax.text(17, 10.3, "recovery-covered\ntraining states", color=BLUE,
        fontsize=9.3, weight="bold", ha="center")
robot = (47, -7.5)
ax.add_patch(Circle(robot, 1.25, facecolor="#333333", edgecolor="white",
                    lw=1.1, zorder=8))
ax.add_patch(FancyArrowPatch(robot, (55, -3.5), arrowstyle="-|>",
                             mutation_scale=15, lw=2.4, color=GREEN, zorder=9))
ax.text(58, -5.7, "clean servo correction\npoints toward centerline",
        color=GREEN, fontsize=9.2, weight="bold")
ax.set_title("A   Closed-loop funnel-to-dock control", loc="left",
             fontsize=11.5, weight="bold")

# B. Conditional demonstration distribution at one state.
ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(-0.26, 1.82)
ax.set_ylim(-0.10, 1.08)
ax.axis("off")
origin = (-0.08, 0.13)
ax.add_patch(Circle(origin, 0.045, facecolor="#333333", zorder=5))
branches = [
    (0.0, 0.74, GREEN, "70%  clean", "task-correct"),
    (0.2, 0.53, "#e98973", "20%  mild", "corrupt"),
    (1.6, 0.30, "#a94442", "10%  gross", "corrupt"),
]
for residual, height, color, label, sub in branches:
    end = (residual, height)
    ax.add_patch(FancyArrowPatch(origin, end, arrowstyle="-|>",
                                 mutation_scale=16, lw=3.0, color=color,
                                 connectionstyle="arc3,rad=0.02"))
    ax.text(end[0], end[1] + 0.06, f"{label}\n{sub}", ha="center",
            fontsize=9.3, color=color, weight="bold" if residual == 0 else "normal")
ax.axvline(0.2, ymin=0.02, ymax=0.94, color=BLUE, ls=":", lw=2)
ax.text(0.24, 0.96, "label mean = +0.2", color=BLUE, fontsize=9.5,
        weight="bold", ha="left")
ax.text(0.76, 0.08, "same observed state\nthree demonstrated action branches",
        ha="center", fontsize=9.3, color="#444444")
ax.set_title("B   Contaminated action labels", loc="left",
             fontsize=11.5, weight="bold")

# C. Deployed policy paths.
ax = fig.add_subplot(gs[0, 2])
funnel(ax)
x = np.linspace(0, 160, 80)
y0s = (-9.5, -5.0, 0.0, 5.0, 9.5)
targets = {
    "L2 / HG: mean branch": (3.95, BLUE, "--"),
    "MIP: partial retreat": (2.50, RED, "-."),
    "HT: clean branch": (0.02, ORANGE, "-"),
}
for label, (target, color, style) in targets.items():
    for j, y0 in enumerate(y0s):
        y = target + (y0 - target) * np.exp(-x / 8.0)
        ax.plot(x, y, color=color, ls=style, lw=2.5 if j == 0 else 1.2,
                alpha=0.95 if j == 0 else 0.35,
                label=label if j == 0 else None)
ax.scatter([160, 160], [3.95, 2.50], marker="x", s=75, color=[BLUE, RED],
           linewidth=2.3, zorder=8)
ax.scatter([160], [0.02], marker="o", s=52, facecolor=ORANGE,
           edgecolor="white", linewidth=1.0, zorder=8)
ax.annotate("miss", (160, 3.95), (136, 6.8), color=BLUE, fontsize=9.2,
            weight="bold", arrowprops=dict(arrowstyle="->", color=BLUE))
ax.annotate("miss", (160, 2.50), (135, -3.5), color=RED, fontsize=9.2,
            weight="bold", arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate("dock", (160, 0.02), (136, -8.0), color=ORANGE, fontsize=9.2,
            weight="bold", arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.legend(frameon=False, fontsize=8.6, loc="upper left")
ax.set_title("C   Replanning turns estimator bias into a miss", loc="left",
             fontsize=11.5, weight="bold")

fig.text(
    0.5, 0.035,
    "Reversible axis: HT commits to the densest branch. It failed when that branch was nuisance; "
    "here it wins because the dense branch is the latent clean controller.",
    ha="center", fontsize=10, color="#333333",
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
