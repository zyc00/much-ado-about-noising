"""Construct actual 100%-successful noisy demonstration paths.

Unlike the label-corruption benchmark, noise is executed in the environment.
Each kick is followed by a compensating residual that rejoins the clean oracle
trajectory exactly after two cycles.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "analysis/paper/toy2d_compensated_noisy_paths.png"
OUT_PDF = ROOT / "analysis/paper/toy2d_compensated_noisy_paths.pdf"

GREEN = "#27863e"
ORANGE = "#e49a3a"
RED = "#c74440"
GREY = "#7c858c"
DOCK_TOL = 0.5
LAT_ACTION_MM = 2.0


def half_width(x):
    x = np.asarray(x)
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def rollout(y0, rng, noisy=True):
    x, y = 0.0, float(y0)
    xs, ys = [x], [y]
    residuals = []
    collision = False
    pending = 0.0
    for step in range(40):
        if not noisy:
            residual = 0.0
        elif step % 2 == 0:
            # Normalized residual modes: 70% clean, 20% mild, 10% gross.
            normalized = rng.choice([0.0, 0.2, 1.6], p=[0.7, 0.2, 0.1])
            residual = normalized * LAT_ACTION_MM
            pending = residual
        else:
            # Exact two-step compensation:
            # y1=.5*y0+d; y2=.5*y1-.5*d=.25*y0 (clean two-step state).
            residual = -0.5 * pending
            pending = 0.0
        dy = -0.5 * y + residual
        x += 4.0
        y += dy
        xs.append(x)
        ys.append(y)
        residuals.append(residual)
        collision = collision or abs(y) > half_width(x)
    success = not collision and abs(y) < DOCK_TOL
    return {
        "x": np.asarray(xs),
        "y": np.asarray(ys),
        "residual": np.asarray(residuals),
        "collision": collision,
        "success": success,
    }


rng = np.random.default_rng(20260805)
starts = np.linspace(-10.0, 10.0, 31)
noisy_paths = [rollout(y0, rng, noisy=True) for y0 in starts]
oracle_paths = [rollout(y0, rng, noisy=False) for y0 in starts]

fig, ax = plt.subplots(figsize=(15.6, 7.4), facecolor="white")
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.79)
fig.suptitle(
    "Path-level alternative: executed noise + corrective recovery, with every demonstration successful",
    fontsize=16.4, weight="bold", y=0.965,
)
fig.text(
    0.5, 0.875,
    "At each two-cycle block: sample a clean/mild/gross kick, then apply its exact compensating residual; "
    "the trajectory rejoins the clean oracle path",
    ha="center", fontsize=10.8, color="#333333",
)

xx = np.linspace(0, 160, 400)
hw = half_width(xx)
ax.fill_between(xx, -hw, hw, color="#e9eef2", zorder=0)
ax.plot(xx, hw, color=GREY, lw=2.0)
ax.plot(xx, -hw, color=GREY, lw=2.0)
ax.axhline(0, color="#a0a7ac", lw=1.1, ls="--")

for path in noisy_paths:
    has_gross = np.any(path["residual"] > 2.0)
    ax.plot(path["x"], path["y"], color=RED if has_gross else ORANGE,
            lw=1.15, alpha=0.30, zorder=2)
for i, path in enumerate(oracle_paths[::5]):
    ax.plot(path["x"], path["y"], color=GREEN, lw=1.7,
            alpha=0.50, zorder=3,
            label="clean oracle path" if i == 0 else None)
ax.plot([], [], color=ORANGE, lw=2.0, alpha=0.75,
        label="executed noisy demo (mild/clean blocks)")
ax.plot([], [], color=RED, lw=2.0, alpha=0.75,
        label="executed noisy demo (contains gross block)")

ax.scatter(np.zeros_like(starts), starts, s=15, color="#222222",
           alpha=0.62, zorder=5)
ax.annotate("start distribution", (0, 9.5), (18, 14.7),
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3),
            color="#444444", fontsize=10.3)

ax.add_patch(Rectangle((158.5, -DOCK_TOL), 3.0, 2 * DOCK_TOL,
                       facecolor="#83c995", edgecolor=GREEN, lw=1.5,
                       zorder=7))
ax.scatter([160], [0], marker="*", s=155, color=GREEN, zorder=8)
ax.annotate("all paths rejoin and dock\n$|y|<0.5$ mm", (160, 0), (136, 10.0),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
            color=GREEN, fontsize=10.3, weight="bold", ha="center")

# Explain one measured kick/recovery pair on a path containing a gross kick.
example = next(p for p in noisy_paths if np.any(p["residual"] > 2.0))
gross_step = int(np.flatnonzero(example["residual"] > 2.0)[0])
p0 = (example["x"][gross_step], example["y"][gross_step])
p1 = (example["x"][gross_step + 1], example["y"][gross_step + 1])
p2 = (example["x"][gross_step + 2], example["y"][gross_step + 2])
ax.scatter(*p1, marker="x", s=75, color=RED, linewidth=2.2, zorder=8)
ax.annotate("executed gross kick", p1, (p1[0] + 14, p1[1] + 6.0),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
            color=RED, fontsize=10.0, weight="bold")
ax.annotate("next action compensates;\npath rejoins oracle here", p2,
            (p2[0] + 21, p2[1] - 6.5),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
            color=GREEN, fontsize=10.0, weight="bold", ha="center")

demo_sr = np.mean([p["success"] for p in noisy_paths])
demo_coll = np.sum([p["collision"] for p in noisy_paths])
oracle_sr = np.mean([p["success"] for p in oracle_paths])
ax.text(
    102, -12.2,
    f"Path validity check\nnoisy demonstrations: {sum(p['success'] for p in noisy_paths)}/"
    f"{len(noisy_paths)} = {demo_sr:.0%} SR\n"
    f"collisions: {demo_coll}/{len(noisy_paths)}\n"
    f"clean oracle: {oracle_sr:.0%} SR",
    ha="center", va="center", fontsize=10.2,
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#8b9398"),
)

ax.set_xlim(-3, 165)
ax.set_ylim(-18, 18)
ax.set_xlabel("forward position $x$ (mm)", fontsize=11.4)
ax.set_ylabel("lateral position $y$ (mm)", fontsize=11.4)
ax.set_title("31 dynamically valid noisy demonstration trajectories",
             loc="left", fontsize=12.2, weight="bold")
ax.legend(frameon=False, fontsize=9.2, loc="upper right")
ax.grid(alpha=0.12)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"noisy_demo_SR={demo_sr:.6f} collisions={demo_coll}")
print(f"oracle_SR={oracle_sr:.6f}")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
