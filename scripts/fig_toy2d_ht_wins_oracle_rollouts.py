"""Clean oracle rollouts for the HT-wins 2D funnel task."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "analysis/paper/toy2d_ht_wins_oracle_rollouts.png"
OUT_PDF = ROOT / "analysis/paper/toy2d_ht_wins_oracle_rollouts.pdf"

GREEN = "#27863e"
GREY = "#7c858c"
DOCK_TOL = 0.5


def half_width(x):
    x = np.asarray(x)
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def oracle_rollout(y0):
    """Exact deployed oracle: dx=4 mm and dy=-0.5*y each replanning cycle."""
    x, y = 0.0, float(y0)
    xs, ys = [x], [y]
    collision = False
    while x < 160.0:
        x += 4.0
        y += -0.5 * y
        xs.append(x)
        ys.append(y)
        collision = collision or abs(y) > half_width(x)
    return np.asarray(xs), np.asarray(ys), collision


starts = np.linspace(-10.0, 10.0, 21)
rollouts = [oracle_rollout(y0) for y0 in starts]
success = [not col and abs(ys[-1]) < DOCK_TOL for _, ys, col in rollouts]
dock_error = [abs(ys[-1]) for _, ys, _ in rollouts]

fig, ax = plt.subplots(figsize=(11.8, 7.0), facecolor="white")
fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.82)
fig.suptitle("Clean oracle policy trajectories — HT-wins task",
             fontsize=16.5, weight="bold", y=0.96)
fig.text(
    0.5, 0.875,
    "These are environment rollouts, not training-state samples. "
    "The action-label corruption is never executed by the oracle.",
    ha="center", fontsize=10.5, color="#333333",
)

xx = np.linspace(0, 160, 400)
hw = half_width(xx)
ax.fill_between(xx, -hw, hw, color="#e9eef2", zorder=0)
ax.plot(xx, hw, color=GREY, lw=1.8)
ax.plot(xx, -hw, color=GREY, lw=1.8)
ax.axhline(0, color="#9ca4aa", lw=1.0, ls=":")

for i, ((xs, ys, _), y0) in enumerate(zip(rollouts, starts)):
    ax.plot(xs, ys, color=GREEN, alpha=0.34, lw=1.5, zorder=2)
    ax.scatter([xs[0]], [ys[0]], s=15, color="#222222", alpha=0.65, zorder=4)
ax.plot(rollouts[-1][0], rollouts[-1][1], color=GREEN, lw=2.6,
        label=r"oracle $pi^*$:  $Delta y=-0.5y$", zorder=3)

ax.add_patch(Rectangle((158.5, -DOCK_TOL), 3.0, 2 * DOCK_TOL,
                       facecolor="#83c995", edgecolor=GREEN, lw=1.5,
                       zorder=6))
ax.scatter([160], [0], marker="*", s=150, color=GREEN, zorder=7)
ax.annotate("tight dock\n$|y|<0.5$ mm", (160, 0), (135, 8.8),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
            color=GREEN, fontsize=10.3, weight="bold", ha="center")
ax.annotate("start distribution", (0, 9.5), (21, 12.3),
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3),
            color="#444444", fontsize=10.2)

ax.text(
    103, -9.8,
    f"Oracle rollout result\n"
    f"success = {sum(success)}/{len(success)} = {np.mean(success):.0%}\n"
    f"collisions = 0/{len(success)}\n"
    f"max dock error = {max(dock_error):.2e} mm",
    ha="center", va="center", fontsize=10.3,
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#8b9398"),
)

ax.set_xlim(-3, 164)
ax.set_ylim(-15, 15)
ax.set_xlabel("forward position $x$ (mm)", fontsize=11.3)
ax.set_ylabel("lateral position $y$ (mm)", fontsize=11.3)
ax.set_title("21 initial lateral positions; one action executed per replanning cycle",
             loc="left", fontsize=11.8, weight="bold")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.grid(alpha=0.12)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"oracle SR={np.mean(success):.6f} collision={sum(r[2] for r in rollouts)}")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
