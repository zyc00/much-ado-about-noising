"""Two-plate slot slalom visualization: demos (A), mean family (B),
diffusion (C), HT (D). Failures are terminal (plate collisions).

Usage: python scripts/fig_toy2d_slalom.py [slalom.json]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XG = 160.0
SLOTS = np.array([0.0, -6.0, -11.0])
PROBS = [0.60, 0.28, 0.12]
GAP = 1.2
PLATES = [(50.0, 58.0), (100.0, 108.0)]
SWITCH = [62.0, 112.0]
K, FWD, HORIZON = 0.5, 4.0, 8

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "analysis/toy2d_slalom.json"))
arms = d["arms"]


def half_width(x):
    return 14.0 - 1.5 * np.clip(x / XG, 0.0, 1.0)


def draw_scene(ax):
    xs = np.linspace(0, XG, 200)
    ax.fill_between(xs, -half_width(xs), half_width(xs), color="#dfe7f0", zorder=0)
    ax.plot(xs, half_width(xs), color="0.45", lw=1.2)
    ax.plot(xs, -half_width(xs), color="0.45", lw=1.2)
    for x0, x1 in PLATES:
        hw = float(half_width((x0 + x1) / 2))
        segs = [(SLOTS[0] + GAP, hw), (SLOTS[1] + GAP, SLOTS[0] - GAP),
                (SLOTS[2] + GAP, SLOTS[1] - GAP), (-hw, SLOTS[2] - GAP)]
        for lo, hi in segs:
            if hi > lo:
                ax.add_patch(plt.Rectangle((x0, lo), x1 - x0, hi - lo,
                                           facecolor="0.35", edgecolor="0.2", zorder=2))
    ax.plot(XG, 0, "*", color="tab:green", ms=15, zorder=6)
    ax.set_xlim(0, XG + 4)
    ax.set_ylim(-15, 15)


def demo_traj(rng):
    ks = rng.choice(3, p=PROBS, size=2)
    j = [0.08, 0.30, 0.50]
    lane1 = SLOTS[ks[0]] + float(np.clip(rng.normal(0, j[ks[0]]), -0.9, 0.9))
    lane2 = SLOTS[ks[1]] + float(np.clip(rng.normal(0, j[ks[1]]), -0.9, 0.9))
    y = float(rng.uniform(-10, 10))
    x = float(rng.uniform(0, FWD * HORIZON))
    tx, ty = [x], [y]
    while x < XG:
        lane = lane1 if x < SWITCH[0] else (lane2 if x < SWITCH[1] else 0.0)
        y = y - K * (y - lane); x += FWD
        tx.append(x); ty.append(y)
    return tx, ty


fig, axes = plt.subplots(2, 2, figsize=(16, 8.6), sharex=True, sharey=True)
(axA, axB), (axC, axD) = axes
fig.suptitle("Two-plate slot slalom: per-plate lane choice (60/28/12), all executed, oracle 100% — "
             "failures are terminal collisions",
             fontsize=14.5, fontweight="bold")

draw_scene(axA)
rng = np.random.default_rng(5)
for _ in range(22):
    tx, ty = demo_traj(rng)
    axA.plot(tx, ty, color="0.5", alpha=0.5, lw=1.0, zorder=3)
axA.set_title("A   Demonstrations: pick a gap per plate, thread it, dock at center",
              fontsize=12, fontweight="bold", loc="left")
axA.set_ylabel("lateral position y (mm)", fontsize=11)

draw_scene(axB)
for key, lab, c in [("l2", "MSE", "#2b6cb0"), ("mip", "MIP", "#3aa3a0")]:
    a = arms[key]
    for t in a["seed0"]["traces"]:
        axB.plot(t["x"], t["y"], color=c, alpha=0.6, lw=1.2, zorder=3)
    axB.plot([], [], color=c, lw=2.2,
             label=f"{lab}  SR {a['sr_mean']:.2f}, collision {a['collision_mean']:.2f}")
axB.legend(fontsize=9.5, loc="lower left")
axB.set_title("B   MSE: averages the lanes into the web — every rollout collides",
              fontsize=12, fontweight="bold", loc="left")

draw_scene(axC)
for key, lab, c in [("flow", "Flow matching", "#6b46c1"), ("flow_hist", "Flow matching +history", "#b794f4")]:
    a = arms[key]
    for t in a["seed0"]["traces"]:
        axC.plot(t["x"], t["y"], color=c, alpha=0.6, lw=1.2, zorder=3)
    axC.plot([], [], color=c, lw=2.2,
             label=f"{lab}  SR {a['sr_mean']:.2f}, collision {a['collision_mean']:.2f}")
axC.legend(fontsize=9.5, loc="lower left")
axC.set_title("C   Diffusion: churns between lane commitments — compounding terminal failure",
              fontsize=12, fontweight="bold", loc="left")
axC.set_ylabel("lateral position y (mm)", fontsize=11)
axC.set_xlabel("forward position x (mm)", fontsize=11)

draw_scene(axD)
for key, lab, c in [("ht05", "HT nu=0.5", "#c23b22"), ("ht2", "HT nu=2", "#dd6b20")]:
    a = arms[key]
    for t in a["seed0"]["traces"]:
        axD.plot(t["x"], t["y"], color=c, alpha=0.6, lw=1.2, zorder=3)
    axD.plot([], [], color=c, lw=2.2,
             label=f"{lab}  SR {a['sr_mean']:.2f}, collision {a['collision_mean']:.2f}")
axD.legend(fontsize=9.5, loc="lower left")
axD.set_title("D   HT: re-commits to the plurality gap at each plate",
              fontsize=12, fontweight="bold", loc="left")
axD.set_xlabel("forward position x (mm)", fontsize=11)

fig.tight_layout(rect=[0, 0.01, 1, 0.94])
fig.savefig("analysis/paper/toy2d_slalom_trajs.png", dpi=160)
fig.savefig("analysis/paper/toy2d_slalom_trajs.pdf")
print("saved analysis/paper/toy2d_slalom_trajs.png")
