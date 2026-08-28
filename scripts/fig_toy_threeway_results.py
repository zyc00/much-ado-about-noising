"""Trained-model results for the three-branch witness + the no-free-lunch
composition heatmap across all measured witnesses."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("analysis/toy2d_mip_nfl_threeway.json"))
ARMS = ["regression", "hg", "ht", "ht05", "htl", "mip_step1", "mip_full"]
LBL = {"regression": "Regression", "hg": "HG", "ht": "HT $\\nu$=2",
       "ht05": "HT $\\nu$=0.5", "htl": "HT learned-$\\nu$",
       "mip_step1": "MIP step 1", "mip_full": "MIP full"}
C = {"regression": "tab:blue", "hg": "tab:orange", "ht": "#7fbf7f",
     "ht05": "tab:green", "htl": "#2d6a4f", "mip_step1": "#8c6bb1",
     "mip_full": "tab:red"}

agg = {}
for c in d["cells"]:
    if c["mode"] != "threeway":
        continue
    for m, v in c["methods"].items():
        agg.setdefault(m, []).append((v["success_rate"],
                                      v["abs_y_goal_p50"],
                                      v["first_lateral_residual_mean"]))
agg = {m: np.array(v) for m, v in agg.items()}

fig = plt.figure(figsize=(20, 9.6))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
xi = np.arange(len(ARMS))
ax.bar(xi, [agg[m][:, 0].mean() for m in ARMS],
       color=[C[m] for m in ARMS], alpha=0.9)
for i, m in enumerate(ARMS):
    for v in agg[m][:, 0]:
        ax.plot(xi[i] + np.random.uniform(-0.08, 0.08), v, "o", color="k",
                ms=3.5, alpha=0.6)
ax.set_xticks(xi)
ax.set_xticklabels([LBL[m] for m in ARMS], fontsize=8.5, rotation=20)
ax.set_ylabel("dock success")
ax.set_ylim(0, 1.1)
ax.set_title("A   Three-branch witness, trained (8 seeds):\nHT $\\nu$=0.5 "
             "docks 1.000; MIP full 0.125", fontsize=11, loc="left")

ax = fig.add_subplot(gs[0, 1])
ax.bar(xi, [agg[m][:, 2].mean() * 20 for m in ARMS],
       color=[C[m] for m in ARMS], alpha=0.9)
for i, m in enumerate(ARMS):
    for v in agg[m][:, 2] * 20:
        ax.plot(xi[i] + np.random.uniform(-0.08, 0.08), v, "o", color="k",
                ms=3.5, alpha=0.6)
for xv, yv in ((6, -0.57), (2, -0.56), (3, -0.02)):
    ax.plot(xv, yv, marker="*", color="gold", ms=16, mec="k", mew=0.8,
            ls="none", zorder=5)
ax.axhspan(-0.5, 0.5, color="tab:green", alpha=0.12)
ax.set_xticks(xi)
ax.set_xticklabels([LBL[m] for m in ARMS], fontsize=8.5, rotation=20)
ax.set_ylabel("dock offset from first-action bias (mm)")
ax.set_title("B   Trained bias vs population prediction ($\\star$):\n"
             "MIP and HT $\\nu$=2 pulled to the bottom branch", fontsize=11,
             loc="left")

ax = fig.add_subplot(gs[0, 2])
pair = ["mip_step1", "mip_full"]
ax.bar([0, 1], [agg[m][:, 0].mean() for m in pair],
       color=[C[m] for m in pair], alpha=0.9, width=0.55)
for i, m in enumerate(pair):
    for v in agg[m][:, 0]:
        ax.plot(i + np.random.uniform(-0.06, 0.06), v, "o", color="k",
                ms=4, alpha=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["step 1 only", "full (step 2 composed)"], fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("dock success")
ax.set_title("C   Causal control, same trained MIP weights:\n1.000 "
             "$\\to$ 0.125 when the denoising step is composed",
             fontsize=11, loc="left")
ax.text(0.5, 0.55, "$\\hat\\nu$(chunk-norm learned-$\\nu$) = 1.75\n"
        "(pools 16 chunk dims $\\to$ misses the small $\\nu$;\n"
        "on skew data the same learner finds 0.59)", fontsize=8.5,
        ha="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray"))

ax = fig.add_subplot(gs[1, :])
rows = ["L2", "HG", "HT (best hypothesis)", "MIP full"]
cols = ["skew nuisance\n(toy, 8 seeds)", "three-branch\n(toy, 8 seeds)",
        "heavy tail + small data\n(toy, 8 seeds)",
        "script route data\n(REAL tool-hang, N=200)"]
M = np.array([
    [0.63, 1.00, 0.00, 0.51],
    [1.00, 1.00, 0.25, 0.805],
    [0.00, 1.00, 0.875, 0.74],
    [0.00, 0.125, 0.125, 0.84],
])
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                fontsize=12,
                color="k" if 0.25 < M[i, j] < 0.9 else "white")
ax.set_xticks(range(4))
ax.set_xticklabels(cols, fontsize=10)
ax.set_yticks(range(4))
ax.set_yticklabels(rows, fontsize=11)
ax.set_title("D   No-free-lunch composition: every method has a winning "
             "and a losing data regime (columns: measured witnesses; HT row "
             "uses $\\nu$=2 except three-branch $\\nu$=0.5; HG on real data "
             "= HG+condreg; L2 skew cell is the knife-edge value)",
             fontsize=11.5, loc="left")

fig.suptitle("Three-branch witness results + the no-free-lunch matrix",
             fontsize=15, y=0.99)
fig.savefig("analysis/paper/toy_threeway_nfl_results.png", dpi=140,
            bbox_inches="tight")
print("saved")
