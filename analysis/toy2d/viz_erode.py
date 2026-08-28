"""Erosion toy figure: predicted grating heatmaps (arms x checkpoints) +
fine-MSE curves."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ.setdefault("NPTS", "4000")
FREQ, AMP = 10.0, 0.15
gg = np.linspace(0, 1, 120)
GX, GY = np.meshgrid(gg, gg)
TRUTH = AMP * np.sin(FREQ * np.pi * GX) * np.sin(FREQ * np.pi * GY) * (GX > 0.7)

ARMS = ["l2", "flowF", "flowZ", "mip2"]
LBL = {"l2": "L2", "flowF": "flow (fresh noise)",
       "flowZ": "flow (FROZEN noise)", "mip2": "MIP 2-step"}
CKS = ["4000", "20000", "60000", "100000"]

fig = plt.figure(figsize=(16.5, 10.5))
gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 0.15, 1.7], wspace=0.15,
                      hspace=0.25)

for r, arm in enumerate(ARMS):
    Z = np.load(f"erode_snaps_{arm}.npz")
    for c, ck in enumerate(CKS):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(Z[ck], origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
                  vmin=-AMP, vmax=AMP)
        ax.axvline(0.7, color="k", lw=0.6, ls=":")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"{int(ck)//1000}k steps", fontsize=10)
        if c == 0:
            ax.set_ylabel(LBL[arm], fontsize=10)

# curves panel
axc = fig.add_subplot(gs[:, 5])
C = {"l2": "#d62728", "flowF": "#2ca02c", "flowZ": "#9467bd",
     "mip2": "#7f7f7f"}
import re
for arm in ARMS:
    steps, fm = [], []
    for line in open("erode_log_b.txt"):
        m = re.match(rf"ERODE {arm} (\d+) fmse ([\d.e+-]+)", line)
        if m:
            steps.append(int(m.group(1)) / 1000)
            fm.append(float(m.group(2)))
    axc.semilogy(steps, fm, "-", color=C[arm], lw=2, label=LBL[arm])
axc.axhline(5.6e-3, color="0.5", lw=1, ls="--")
axc.text(50, 6.1e-3, "no-fit level (grating power)", fontsize=8, color="0.4",
         ha="center")
axc.axhline(1.7e-5, color="0.5", lw=1, ls=":")
axc.text(50, 2.0e-5, "full-fit reference (clean control)", fontsize=8,
         color="0.4", ha="center")
axc.set_xlabel("training step ($\\times 10^3$)")
axc.set_ylabel("fine-region clean MSE (grating dims)")
axc.legend(frameon=False, fontsize=9, loc="center right")
axc.spines[["top", "right"]].set_visible(False)
axc.set_title("fine-structure fit over training", fontsize=11)

fig.suptitle("Erosion toy: fixed inputs + irreducible label noise elsewhere — "
             "who keeps the fine grating?", fontsize=13)
fig.savefig("erode_fig.png", dpi=150, bbox_inches="tight")
print("wrote erode_fig.png")
