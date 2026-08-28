"""Single-panel, paper-style visualization of the HT-wins 2D task."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "analysis/paper/toy2d_ht_wins_task_simple.png"
OUT_PDF = ROOT / "analysis/paper/toy2d_ht_wins_task_simple.pdf"

GREEN = "#27863e"
BLUE = "#2878b5"
ORANGE = "#e98943"
RED = "#c74440"
GREY = "#7c858c"


def half_width(x):
    x = np.asarray(x)
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def chunk_path(x0, y0, residual, horizon=8):
    """The same dynamically consistent 8-action chunks used by the toy."""
    xs, ys = [x0], [y0]
    x, y = x0, y0
    collided = False
    for _ in range(horizon):
        dy = -0.5 * y + 10.0 * residual
        x += 4.0
        y += dy
        xs.append(x)
        ys.append(y)
        if abs(y) > half_width(x):
            collided = True
            break
    return np.asarray(xs), np.asarray(ys), collided


fig, ax = plt.subplots(figsize=(15.8, 7.4), facecolor="white")
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.79)
fig.suptitle(
    "2D funnel-to-dock task: recover the clean controller from contaminated action labels",
    fontsize=17, weight="bold", y=0.965,
)
fig.text(
    0.5, 0.875,
    "TRAINING AND EVALUATION STATES OVERLAP  ·  the same state has 70% clean, "
    "20% mildly corrupt, and 10% grossly corrupt action chunks",
    ha="center", fontsize=11.2, color="#333333",
)

# Funnel and dock.
xx = np.linspace(0, 160, 400)
hw = half_width(xx)
ax.fill_between(xx, -hw, hw, color="#e9eef2", zorder=0)
ax.plot(xx, hw, color=GREY, lw=2.0)
ax.plot(xx, -hw, color=GREY, lw=2.0)
ax.axhline(0, color="#a0a7ac", lw=1.2, ls="--")
ax.add_patch(Rectangle((158.5, -0.5), 3.0, 1.0, facecolor="#83c995",
                       edgecolor=GREEN, lw=1.6, zorder=8))
ax.scatter([160], [0], marker="*", s=175, color=GREEN, zorder=9)
ax.annotate("tight dock\n$|y|<0.5$ mm", (160, 0), (135, 13.5),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6),
            color=GREEN, ha="center", fontsize=11, weight="bold")

# Recovery-covered train/evaluation states.
rng = np.random.default_rng(17)
sx = rng.uniform(2, 154, 360)
shw = half_width(sx)
sy = rng.uniform(-0.82 * shw, 0.82 * shw)
ax.scatter(sx, sy, s=10, color=BLUE, alpha=0.11, edgecolors="none", zorder=1)
ax.text(17, 10.5, "recovery-covered states\n(train = evaluation support)",
        color=BLUE, fontsize=10.2, weight="bold", ha="center")

# Start distribution and a few latent clean rollouts.
start_y = np.linspace(-9.5, 9.5, 9)
ax.plot(np.zeros_like(start_y), start_y, color="#333333", lw=3.0, zorder=5)
ax.scatter(np.zeros_like(start_y), start_y, s=22, color="#333333", zorder=6)
ax.annotate("start distribution", (0, 8.5), (20, 15.3),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.4),
            color="#444444", fontsize=10.5)
for j, y0 in enumerate(start_y):
    px = np.linspace(0, 160, 100)
    py = y0 * np.exp(-px / 8.0)
    ax.plot(px, py, color=GREEN, lw=1.2 if j != 0 else 1.8,
            alpha=0.25, zorder=2)
ax.text(26, -3.6, "latent clean servo trajectories", color=GREEN,
        fontsize=10.3, weight="bold")

# Same-state action-label branches, using actual chunk dynamics.
anchor = (72.0, -5.2)
ax.add_patch(Circle(anchor, 0.75, facecolor="#222222", edgecolor="white",
                    lw=1.0, zorder=10))
ax.annotate("same observed state", anchor, (54, -13.2),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4),
            color="#333333", fontsize=10.3, weight="bold", ha="center")

branches = [
    (0.0, GREEN, "70%  clean chunk\n(task-correct)"),
    (0.2, ORANGE, "20%  mild corruption\n(residual +0.2)"),
    (1.6, RED, "10%  gross corruption\n(residual +1.6)"),
]
for residual, color, label in branches:
    bx, by, collided = chunk_path(*anchor, residual)
    ax.plot(bx, by, color=color, lw=3.0, marker="o", markersize=3.5,
            zorder=7)
    end = (bx[-1], by[-1])
    if collided:
        ax.scatter(*end, marker="x", s=95, color=color, linewidth=2.5, zorder=9)
    if residual == 0.0:
        text_xy = (100, -7.1)
    elif residual == 0.2:
        text_xy = (105, 6.0)
    else:
        text_xy = (78, 17.0)
    ax.annotate(label, end, text_xy,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                color=color, fontsize=10.2, weight="bold",
                ha="center")

# One-line task readout.
ax.text(
    116, -13.8,
    "deployment target: follow the clean branch and dock\n"
    "HT selects it; L2/HG fit the +0.2 mean; MIP remains on the corrupt side",
    ha="center", va="center", fontsize=10.1,
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#9aa1a6"),
)

ax.set_xlim(-3, 165)
ax.set_ylim(-19, 20)
ax.set_xlabel("forward position $x$ (mm)", fontsize=11.5)
ax.set_ylabel("lateral position $y$ (mm)", fontsize=11.5)
ax.set_title("Point robot, 8-action replanning chunks, narrowing recovery corridor",
             loc="left", fontsize=12.5, weight="bold")
ax.grid(alpha=0.12)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
