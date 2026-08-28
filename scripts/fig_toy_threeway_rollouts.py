"""Representative three-branch rollouts (seed 0), codex panel-D style,
with a dock zoom for the sub-mm equilibria."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "scripts")
from toy2d_mip_nfl import (half_width_mm, make_dataset, rollout_metrics,
                           train, train_nll, DOCK_TOL_MM, X_GOAL_MM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state, action = make_dataset("threeway", 3000, 10, 8, 20260805)
arms = {}
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

fig, (ax, axz) = plt.subplots(
    1, 2, figsize=(17.5, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
xs = np.linspace(0, 172, 300)
for a in (ax, axz):
    a.fill_between(xs, -half_width_mm(xs), half_width_mm(xs),
                   color="#dbe4ee", alpha=0.85)
for name, (trs, sr, col) in traces.items():
    for t in trs:
        xx = np.array(t["x"])
        yy = np.array(t["y"])
        ax.plot(xx, yy, color=col, lw=1.0, alpha=0.55)
        axz.plot(xx, yy, color=col, lw=1.6, alpha=0.75)
    ax.plot([], [], color=col, lw=2.5,
            label=f"{name} (SR {sr:.2f})")
for a in (ax, axz):
    a.axvline(X_GOAL_MM, color="k", lw=1)
ax.add_patch(plt.Rectangle((X_GOAL_MM - 0.8, -DOCK_TOL_MM), 1.6,
                           2 * DOCK_TOL_MM, color="tab:green", alpha=0.8))
axz.axhspan(-DOCK_TOL_MM, DOCK_TOL_MM, color="tab:green", alpha=0.2)
axz.text(126, DOCK_TOL_MM + 0.06, "dock tolerance $\\pm$0.5 mm",
         fontsize=9, color="tab:green")
ax.set_xlim(-3, 174)
ax.set_ylim(-12.5, 12.5)
ax.set_xlabel("forward position x (mm)")
ax.set_ylabel("lateral position y (mm)")
ax.legend(fontsize=9.5, loc="upper right")
ax.set_title("Representative three-branch rollouts (seed 0, 11 starts per "
             "method): every arm funnels in — the story is at the dock",
             fontsize=11.5, loc="left")
axz.set_xlim(125, 168)
axz.set_ylim(-1.6, 1.1)
axz.set_xlabel("x (mm)")
axz.set_title("Dock zoom: MIP full settles on the bottom-branch "
              "equilibrium\n($\\approx-$0.7 mm, outside tolerance); HT "
              "$\\nu$=0.5 centers; HT $\\nu$=2 rides the edge",
              fontsize=10.5, loc="left")
fig.tight_layout()
fig.savefig("analysis/paper/toy_threeway_rollouts.png", dpi=145,
            bbox_inches="tight")
print("saved")
