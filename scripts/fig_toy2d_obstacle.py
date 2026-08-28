"""Obstacle-detour toy visualization: demos (top) vs trained policies (bottom).

Renders whichever arms are present in the given JSONs (main + flow); the block,
the two demo routes, and the hidden goal are drawn in both rows.

Usage: python scripts/fig_toy2d_obstacle.py [main.json] [flow.json]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLK_X0, BLK_X1, BLK_HALF = 60.0, 100.0, 3.0
DETOUR_A = 6.0
AIM_X0, AIM_X1 = 20.0, 100.0
P_MAJ, SIG = 0.6, 0.05
K, FWD, XG = 0.5, 4.0, 160.0

arms_data = {}
for path in sys.argv[1:] or ["analysis/toy2d_obstacle_l2.json", "analysis/toy2d_obstacle_ht.json",
                              "analysis/toy2d_obstacle_flow.json"]:
    try:
        d = json.load(open(path))
        for k, v in d["arms"].items():
            arms_data[k] = v
    except FileNotFoundError:
        pass


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
    ax.text(XG - 1, -1.8, "goal\n(hidden behind)", fontsize=9.5, color="tab:green",
            ha="right", va="top")
    ax.set_xlim(0, XG + 4)
    ax.set_ylim(-15, 15)
    ax.set_ylabel("lateral position y (mm)", fontsize=11.5)


fig, (axT, axB) = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True, sharey=True)
fig.suptitle("Obstacle-detour toy (push-T mechanism): demos fork around the block; "
             "trained policies differ in whether they can",
             fontsize=15.5, fontweight="bold")

# top: demonstrations
draw_scene(axT)
rng = np.random.default_rng(11)
for i in range(34):
    branch = 1.0 if rng.random() < P_MAJ else -1.0
    y = float(rng.uniform(-10, 10)); x = 0.0
    tx, ty = [x], [y]
    smear = rng.normal(0, SIG) * 10.0
    while x < XG:
        yt = (branch * DETOUR_A + smear) if (AIM_X0 <= x <= BLK_X1) else 0.0
        y = y + (-K * (y - yt)); x += FWD
        tx.append(x); ty.append(y)
    axT.plot(tx, ty, color="tab:red" if branch > 0 else "tab:orange",
             alpha=0.5, lw=1.1, zorder=3)
axT.set_title("A   Demonstrations: detour left (60%, red) or right (40%, orange), re-center, dock — oracle 100%",
              fontsize=12.5, fontweight="bold", loc="left")

# bottom: trained policies
draw_scene(axB)
STYLE = {"l2": ("MSE (mean)", "#2b6cb0"),
         "ht2": ("HT nu=2", "#dd6b20"),
         "ht05": ("HT nu=0.5", "#c23b22")}
for k, (lab, c) in STYLE.items():
    if k not in arms_data or not arms_data[k].get("seed0"):
        continue
    for t in arms_data[k]["seed0"]["traces"]:
        axB.plot(t["x"], t["y"], color=c, alpha=0.6, lw=1.3, zorder=3)
    sr = arms_data[k]["sr_mean"]
    axB.plot([], [], color=c, lw=2.4, label=f"{lab}  (SR {sr:.2f})")
axB.set_title("B   Trained policies (seed 0 rollouts)", fontsize=12.5, fontweight="bold", loc="left")
axB.set_xlabel("forward position x (mm)", fontsize=11.5)
axB.legend(fontsize=10.5, loc="upper right")

fig.tight_layout(rect=[0, 0.01, 1, 0.965])
fig.savefig("analysis/paper/toy2d_obstacle_trajs.png", dpi=160)
fig.savefig("analysis/paper/toy2d_obstacle_trajs.pdf")
print("saved analysis/paper/toy2d_obstacle_trajs.png; arms:", sorted(arms_data))
