"""Trained rollouts for the three-slot witness (seed 0)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ["SLOT_DOCKS"] = "0,-6.0,-11.0"
os.environ["SLOT_TOL"] = "1.0"
sys.path.insert(0, "scripts")
from toy2d_mip_nfl import (half_width_mm, make_dataset, rollout_metrics,
                           train, train_nll, X_GOAL_MM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state, action = make_dataset("slots", 3000, 10, 8, 20260805)
reg, _ = train("regression", state, action, 0, 128, 6000, 512, 5e-4, device)
mip, _ = train("mip", state, action, 0, 128, 6000, 512, 5e-4, device)
ht2, _ = train_nll("ht", state, action, 0, 128, 6000, 512, 5e-4, device)
ht05, _ = train_nll("ht05", state, action, 0, 128, 6000, 512, 5e-4, device)
SPECS = [("Regression", reg, "regression", "tab:blue"),
         ("HT $\\nu$=2", ht2, "ht", "#7fbf7f"),
         ("HT $\\nu$=0.5", ht05, "ht", "tab:green"),
         ("MIP full", mip, "mip_full", "tab:red")]
traces = {}
for name, net, samp, col in SPECS:
    m = rollout_metrics(net, samp, device, 401)
    traces[name] = (m["traces"], m["success_rate"], col)
    print(name, m["success_rate"], round(m["abs_y_goal_p50"], 2))

SLOTS_MM = [0.0, -6.0, -11.0]
TOL = 1.0
fig, (ax, axz) = plt.subplots(
    1, 2, figsize=(17.5, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
xs = np.linspace(0, 172, 300)
for a in (ax, axz):
    a.fill_between(xs, -half_width_mm(xs) - 9, half_width_mm(xs),
                   color="#dbe4ee", alpha=0.85)
for name, (trs, sr, col) in traces.items():
    for t in trs:
        ax.plot(t["x"], t["y"], color=col, lw=1.0, alpha=0.55)
        axz.plot(t["x"], t["y"], color=col, lw=1.7, alpha=0.8)
    ax.plot([], [], color=col, lw=2.5, label=f"{name} (SR {sr:.2f})")
for a in (ax, axz):
    a.axvline(X_GOAL_MM, color="k", lw=1)
    for sm in SLOTS_MM:
        a.axhspan(sm - TOL, sm + TOL, xmin=0.965 if a is ax else 0.9,
                  color="tab:green", alpha=0.5)
ax.set_xlim(-3, 174)
ax.set_ylim(-16, 13)
ax.set_xlabel("forward position x (mm)")
ax.set_ylabel("lateral position y (mm)")
ax.legend(fontsize=9.5, loc="upper right")
ax.set_title("Three-slot witness rollouts (seed 0, zero injected noise): "
             "only HT $\\nu$=0.5 reaches a slot", fontsize=11.5,
             loc="left")
axz.set_xlim(120, 172)
axz.set_ylim(-5.2, 1.2)
for sm in SLOTS_MM[:2]:
    axz.axhspan(sm - TOL, sm + TOL, color="tab:green", alpha=0.18)
axz.axhspan(SLOTS_MM[1] + TOL, SLOTS_MM[0] - TOL, color="k", alpha=0.13)
axz.text(122, -3.0, "wall (failure zone)", fontsize=9)
axz.set_xlabel("x (mm)")
axz.set_title("Gate zoom: L2 $-$3.0 mm, MIP $-$2.0 mm, HT $\\nu$=2 "
              "$-$1.3 mm\nall in the wall; HT $\\nu$=0.5 centers in the "
              "slot", fontsize=10.5, loc="left")
fig.tight_layout()
fig.savefig("analysis/paper/toy_slots_rollouts.png", dpi=145,
            bbox_inches="tight")
print("saved")
