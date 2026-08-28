"""Spurious-mode toy visualization: recorded-data design (A) and trained
rollouts for the three families (B: mean, C: diffusion, D: HT).

Usage: python scripts/fig_toy2d_spurious.py [spurious.json]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XG = 160.0
K, FWD, KICK, P_SPUR = 0.5, 4.0, 2.0, 0.25

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "analysis/toy2d_spurious.json"))
arms = d["arms"]


def half_width(x):
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def draw_scene(ax):
    xs = np.linspace(0, XG, 200)
    ax.fill_between(xs, -half_width(xs), half_width(xs), color="#dfe7f0", zorder=0)
    ax.plot(xs, half_width(xs), color="0.45", lw=1.2)
    ax.plot(xs, -half_width(xs), color="0.45", lw=1.2)
    ax.plot(XG, 0, "*", color="tab:green", ms=16, zorder=4)
    ax.set_xlim(0, XG + 4)
    ax.set_ylim(-15, 15)


fig, axes = plt.subplots(2, 2, figsize=(16, 8.6), sharex=True, sharey=True)
(axA, axB), (axC, axD) = axes
fig.suptitle("Spurious-mode toy: 25% of RECORDED chunks are glitches (stuck +y axis); "
             "executed demos are clean servo (oracle 100%)",
             fontsize=14.5, fontweight="bold")

# A: data design — executed demos (gray) + recorded glitch chunks (red segments)
draw_scene(axA)
rng = np.random.default_rng(7)
for i in range(15):
    y = float(rng.uniform(-10, 10)); x = 0.0
    tx, ty = [x], [y]
    while x < XG:
        y = y - K * y; x += FWD
        tx.append(x); ty.append(y)
    axA.plot(tx, ty, color="0.55", alpha=0.6, lw=1.0, zorder=3)
for _ in range(26):  # recorded glitch chunks: 8 steps of (+FWD, +KICK)
    x0 = rng.uniform(0, XG - 8 * FWD)
    y0 = rng.uniform(-0.7, 0.7) * half_width(x0)
    axA.plot([x0, x0 + 8 * FWD], [y0, y0 + 8 * KICK], color="tab:red",
             alpha=0.75, lw=1.4, zorder=4)
axA.plot([], [], color="0.55", lw=1.6, label="executed demos (clean servo)")
axA.plot([], [], color="tab:red", lw=1.6, label=f"recorded glitch chunks ({P_SPUR:.0%})")
axA.legend(fontsize=9.5, loc="lower left")
axA.set_title("A   Data: glitch mode lives only in the recorded labels",
              fontsize=12, fontweight="bold", loc="left")
axA.set_ylabel("lateral position y (mm)", fontsize=11)

panels = [(axB, "l2", "MSE (mean)", "#2b6cb0",
           "B   Mean family: persistent +y bias, never docks"),
          (axC, "flow", "Flow matching", "#6b46c1",
           "C   Flow matching: samples the glitch at data frequency"),
          (axD, "ht2", "HT nu=2", "#dd6b20",
           "D   Heteroscedastic Student-t: rejects the minority mode")]
for ax, key, lab, color, title in panels:
    draw_scene(ax)
    a = arms[key]
    for t in a["seed0"]["traces"]:
        ax.plot(t["x"], t["y"], color=color, alpha=0.65, lw=1.3, zorder=3)
    ax.plot([], [], color=color, lw=2.2,
            label=f"{lab}  SR {a['sr_mean']:.2f}, collision {a['collision_mean']:.2f}")
    ax.legend(fontsize=9.5, loc="lower left")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
axC.set_ylabel("lateral position y (mm)", fontsize=11)
axC.set_xlabel("forward position x (mm)", fontsize=11)
axD.set_xlabel("forward position x (mm)", fontsize=11)

fig.tight_layout(rect=[0, 0.01, 1, 0.94])
fig.savefig("analysis/paper/toy2d_spurious_trajs.png", dpi=160)
fig.savefig("analysis/paper/toy2d_spurious_trajs.pdf")
print("saved analysis/paper/toy2d_spurious_trajs.png")
