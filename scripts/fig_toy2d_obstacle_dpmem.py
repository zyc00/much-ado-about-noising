"""Obstacle-detour toy: memoryless DP vs history-conditioned DP.

Two panels from the same trained-arm JSON (flow = resample-per-replan,
flow_hist = previous action chunk as conditioning input).

Usage: python scripts/fig_toy2d_obstacle_dpmem.py [dpmem.json]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLK_X0, BLK_X1, BLK_HALF = 60.0, 100.0, 3.0
XG = 160.0

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "analysis/toy2d_obstacle_dpmem.json"))
arms = d["arms"]


def half_width(x):
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def draw_scene(ax):
    xs = np.linspace(0, XG, 200)
    ax.fill_between(xs, -half_width(xs), half_width(xs), color="#dfe7f0", zorder=0)
    ax.plot(xs, half_width(xs), color="0.45", lw=1.2)
    ax.plot(xs, -half_width(xs), color="0.45", lw=1.2)
    ax.add_patch(plt.Rectangle((BLK_X0, -BLK_HALF), BLK_X1 - BLK_X0, 2 * BLK_HALF,
                               facecolor="0.35", edgecolor="0.2", zorder=2))
    ax.text((BLK_X0 + BLK_X1) / 2, 0, "BLOCK", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=3)
    ax.plot(XG, 0, "*", color="tab:green", ms=18, zorder=4)
    ax.set_xlim(0, XG + 4)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("forward position x (mm)", fontsize=11.5)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 5.4), sharey=True)
fig.suptitle("Flow matching on the obstacle toy: replanning without memory flips the branch decision; "
             "conditioning on the previous chunk fixes it",
             fontsize=14.5, fontweight="bold")

panels = [(axL, "flow", "memoryless DP (resample each replan)", "#6b46c1"),
          (axR, "flow_hist", "DP + memory (previous chunk as input)", "#2f855a")]
for ax, key, label, color in panels:
    draw_scene(ax)
    a = arms[key]
    for t in a["seed0"]["traces"]:
        ax.plot(t["x"], t["y"], color=color, alpha=0.65, lw=1.3, zorder=3)
    ax.set_title(f"{label}\nSR {a['sr_mean']:.3f} ± {a['sr_std']:.2f}   "
                 f"collision {a['collision_mean']:.2f}",
                 fontsize=12, fontweight="bold", loc="left")
axL.set_ylabel("lateral position y (mm)", fontsize=11.5)

fig.tight_layout(rect=[0, 0.01, 1, 0.90])
fig.savefig("analysis/paper/toy2d_obstacle_dpmem.png", dpi=160)
fig.savefig("analysis/paper/toy2d_obstacle_dpmem.pdf")
print("saved analysis/paper/toy2d_obstacle_dpmem.png")
