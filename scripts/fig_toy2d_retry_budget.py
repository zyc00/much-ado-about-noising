"""Retry cell, F=0.8: spatial retreat loops (left) and the budget mechanism in
the time domain (right) — every DP rollout that samples retries crosses the
budget lines late; HT finishes far left of all of them.

Usage: python scripts/fig_toy2d_retry_budget.py [retry_f0.8.json]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XG = 160.0
STAGE_Y = 8.0
HORIZON = 8
BUDGETS = {"1.0x (slowest demo)": 8, "1.25x": 10, "2x": 16}

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "analysis/toy2d_retry_f0.8.json"))
arms = d["arms"]
STYLE = {"flow": ("Flow matching", "#6b46c1"), "ht2": ("HT nu=2", "#dd6b20")}


def half_width(x):
    return 14.0 - 1.5 * np.clip(x / XG, 0.0, 1.0)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 5.6))
fig.suptitle("Retry cell (flaw rate 0.8): flow matching's sampled retries cost time — visible only against the budget",
             fontsize=14.5, fontweight="bold")

# left: spatial
xs = np.linspace(0, XG, 200)
axL.fill_between(xs, -half_width(xs), half_width(xs), color="#dfe7f0", zorder=0)
axL.plot(xs, half_width(xs), color="0.45", lw=1.2)
axL.plot(xs, -half_width(xs), color="0.45", lw=1.2)
axL.axhline(STAGE_Y, ls="--", color="0.6", lw=0.9)
axL.plot(XG, 0, "*", color="tab:green", ms=16, zorder=5)
for key, (lab, c) in STYLE.items():
    for t in arms[key]["seed0"]["traces"]:
        axL.plot(t["x"], t["y"], color=c, alpha=0.55, lw=1.2, zorder=3)
    axL.plot([], [], color=c, lw=2.2, label=lab)
axL.legend(fontsize=10, loc="lower left")
axL.set_title("A   Same rollouts in space: retreat loops look harmless",
              fontsize=12, fontweight="bold", loc="left")
axL.set_xlabel("forward position x (mm)", fontsize=11)
axL.set_ylabel("lateral position y (mm)", fontsize=11)
axL.set_xlim(0, XG + 4); axL.set_ylim(-15, 15)

# right: time domain (traces rolled out at the 2x budget; tighter budgets
# TERMINATE a rollout at their line — crossing right of a line = failure)
for key, (lab, c) in STYLE.items():
    for t in arms[key]["seed0"]["traces"]:
        steps = np.arange(len(t["x"]))
        axR.plot(steps / HORIZON, t["x"], color=c, alpha=0.5, lw=1.2, zorder=3)
        tc = (len(t["x"]) - 1) / HORIZON          # goal-crossing time
        if t["x"][-1] >= XG - 1e-6:
            ok_all = tc <= BUDGETS["1.0x (slowest demo)"]
            mid = tc <= BUDGETS["1.25x"]
            col = "tab:green" if ok_all else ("orange" if mid else "tab:red")
            axR.plot(tc, XG, "o" if ok_all else "X", color=col, ms=9 if ok_all else 11,
                     zorder=6, markeredgecolor="0.2")
    axR.plot([], [], color=c, lw=2.2, label=lab)
axR.plot([], [], "o", color="tab:green", ms=8, markeredgecolor="0.2",
         label="docks within every budget")
axR.plot([], [], "X", color="orange", ms=10, markeredgecolor="0.2",
         label="FAILS 1.0x (cut at chunk 8)")
axR.plot([], [], "X", color="tab:red", ms=10, markeredgecolor="0.2",
         label="FAILS 1.0x and 1.25x")
for lab, b in BUDGETS.items():
    axR.axvline(b, ls=":", color="0.3", lw=1.4)
    axR.text(b + 0.1, 8, lab, rotation=90, fontsize=9, color="0.25", va="bottom")
axR.axhline(XG, color="tab:green", lw=1.2, ls="--")
axR.text(0.3, XG + 3, "goal x=160", fontsize=9.5, color="tab:green")
f = arms["flow"]
sr_note = (f"measured SR (401 starts, 3 seeds) — Diffusion: 2x {f['generous']['sr_mean']:.2f} / "
           f"1.25x {f['tight']['sr_mean']:.2f} / 1.0x 0.63 — HT: 1.00 at all")
axR.text(0.35, 30, sr_note, fontsize=9.5, color="0.15",
         bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6"))
axR.set_title("B   Position vs time: X = cut at a budget line before finishing",
              fontsize=12, fontweight="bold", loc="left")
axR.set_xlabel("time (chunks executed)", fontsize=11)
axR.set_ylabel("forward position x (mm)", fontsize=11)
axR.set_xlim(0, 16.5); axR.set_ylim(0, XG + 12)
axR.legend(fontsize=9, loc="center right")

fig.tight_layout(rect=[0, 0.01, 1, 0.92])
fig.savefig("analysis/paper/toy2d_retry_budget.png", dpi=160)
fig.savefig("analysis/paper/toy2d_retry_budget.pdf")
print("saved analysis/paper/toy2d_retry_budget.png")
