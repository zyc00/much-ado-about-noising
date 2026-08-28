"""Noise-magnitude sweep summary: landings scale as -20b until the wall."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FILES = {0.10: "analysis/toy2d_skew_endpoint_orig.json",
         0.15: "analysis/toy2d_skew_endpoint_b015.json",
         0.20: "analysis/toy2d_skew_endpoint_b020.json",
         0.25: "analysis/toy2d_skew_endpoint_b025.json"}
ARMS = [("l2", "Regression", "#2b6cb0", "o"), ("hg", "HG", "#63b3ed", "s"),
        ("mip_step1", "MIP step 1", "#b794f4", "^"),
        ("mip_full", "MIP full", "#7c3aed", "o"),
        ("ht2", "HT nu=2", "#c23b22", "s"), ("ht05", "HT nu=0.5", "#f6ad55", "^")]
data = {b: json.load(open(f)) for b, f in FILES.items()}
bs = sorted(data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.8))
fig.suptitle("Increasing the skew magnitude: the mode family relocates linearly, then hits the wall",
             fontsize=16, fontweight="bold")

bb = np.linspace(0.09, 0.26, 50)
ax1.plot(bb, -20 * bb, ":", color="0.5", lw=1.5, label="analytic mode bias  -20b")
ax1.axhline(0, color="0.6", lw=0.8, ls="--")
ax1.axhspan(-5.6, -5.0, color="0.3", alpha=0.15)
ax1.text(0.092, -5.45, "funnel wall at goal (-5 mm)", fontsize=9.5, color="0.3")
for k, n, c, m in ARMS:
    ys = [data[b]["arms"][k]["landing_median"] if not np.isnan(data[b]["arms"][k]["landing_median"]) else np.nan for b in bs]
    ys = [data[b]["arms"][k]["peak_dock_mm"] if np.isnan(y) else y for b, y in zip(bs, ys)]
    ax1.plot(bs, ys, marker=m, color=c, lw=1.8, ms=7, label=n)
ax1.set_xlabel("skew magnitude b", fontsize=11.5)
ax1.set_ylabel("landing y (mm; peak dock if landing undefined)", fontsize=11.5)
ax1.set_title("A   Landing position vs b", fontsize=13, fontweight="bold", loc="left")
ax1.legend(fontsize=9.5, ncols=2)

for k, n, c, m in ARMS:
    ys = [data[b]["arms"][k]["peak_sr"] for b in bs]
    ax2.plot(bs, ys, marker=m, color=c, lw=1.8, ms=7, label=n)
ax2.axvline(0.25, color="0.5", lw=0.8, ls=":")
ax2.text(0.238, 0.35, "mode equilibrium\nreaches the wall", fontsize=9.5, color="0.35", ha="right")
ax2.set_xlabel("skew magnitude b", fontsize=11.5)
ax2.set_ylabel("SR at own best endpoint", fontsize=11.5)
ax2.set_ylim(0, 1.08)
ax2.set_title("B   Peak success vs b: mode-seekers fail structurally at the wall", fontsize=13, fontweight="bold", loc="left")

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("analysis/paper/toy_skew_bsweep.png", dpi=160)
fig.savefig("analysis/paper/toy_skew_bsweep.pdf")
print("saved analysis/paper/toy_skew_bsweep.png")
