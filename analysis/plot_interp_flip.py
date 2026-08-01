"""Figure: settle->stroke interpolation curves with flip points (f2i family).
Reads CURVE lines (name alpha cos_settle cos_stroke norm nn1) harvested from
probe_interp_curve.py."""
import sys

import matplotlib.pyplot as plt
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "curves.txt"
C = {"L2": "#d62728", "L2-20k": "#e8927c", "HG": "#1f77b4", "HT": "#2ca02c", "MIP": "#7f7f7f"}
STY = {"L2-20k": "--"}
LBL = {"L2-20k": "L2 20k"}

data = {}
for line in open(SRC):
    p = line.split()
    if len(p) >= 7 and p[0] == "CURVE":
        data.setdefault(p[1], []).append([float(x) for x in p[2:7]])
for k in data:
    data[k] = np.array(data[k])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6))

flips = {}
for name in ["L2", "L2-20k", "HG", "HT", "MIP"]:
    d = data[name]
    al, cs, ct = d[:, 0], d[:, 1], d[:, 2]
    diff = cs - ct
    axA.plot(al, diff, STY.get(name, "-"), marker="o", color=C[name], lw=2, ms=3.5)
    # flip point: first zero crossing of diff
    ix = np.where(np.diff(np.sign(diff)))[0]
    if len(ix):
        i = ix[0]
        a_star = al[i] + (al[i + 1] - al[i]) * diff[i] / (diff[i] - diff[i + 1])
        axA.plot([a_star], [0], "o", color=C[name], ms=9, mec="black", mew=1.2,
                 zorder=6)
        flips[name] = a_star

for i, name in enumerate(["L2-20k", "L2", "MIP", "HT", "HG"]):
    if name in flips:
        axA.text(0.02, -0.13 - 0.055 * i,
                 rf"{LBL.get(name, name)}:  $\alpha^*={flips[name]:.2f}$",
                 fontsize=10, color=C[name], fontweight="bold")
axA.axhline(0, color="0.4", lw=1, ls="--")
axA.set_xlabel(r"interpolation coordinate $\alpha$ (settle $\to$ stroke)",
               fontsize=10)
axA.set_ylabel("cos(out, settle chunk) $-$ cos(out, stroke chunk)", fontsize=10)
axA.text(0.03, 0.06, "settle-basin side", fontsize=9, color="0.35")
axA.text(0.72, -0.10, "stroke-basin side", fontsize=9, color="0.35")
axA.set_xlim(-0.02, 1.10)
axA.spines[["top", "right"]].set_visible(False)
axA.set_title("A. output identity along the path; flip point $\\alpha^*$ = "
              "basin boundary", fontsize=11)

for name in ["L2", "L2-20k", "HG", "HT", "MIP"]:
    d = data[name]
    axB.plot(d[:, 0], d[:, 4], STY.get(name, "-"), marker="o", color=C[name], lw=2, ms=3.5)

axB.legend(["L2", "L2 20k", "HG", "HT", "MIP"], loc="upper left", frameon=False,
           labelcolor=[C[k] for k in ["L2", "L2-20k", "HG", "HT", "MIP"]])
axB.set_xlabel(r"interpolation coordinate $\alpha$", fontsize=10)
axB.set_ylabel("1-NN distance of output to the\ndataset action-chunk pool",
               fontsize=10)
axB.set_xlim(-0.02, 1.10)
axB.spines[["top", "right"]].set_visible(False)
axB.set_title("B. validity: excursion off the action manifold mid-path",
              fontsize=11)

fig.suptitle("Settle$\\to$stroke interpolation (f2i checkpoints, 50 pairs): "
             "where each objective places the boundary, and how far outputs "
             "stray between basins", fontsize=12)
fig.tight_layout()
out = "analysis/paper/interp_flip_fig"
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out)
