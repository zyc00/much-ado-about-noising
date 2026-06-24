"""Start-point sweep (all reset+rerun, controller-consistent, MSE 2000 demos).
SR vs eval start point B (= align_done - B steps; B0=at hole, B80~=pick).
Three training-coverage curves: backoff range 0-40 / 0-60 / 0-80.
Shows: cliff tracks the training-coverage edge, but a hard horizon wall remains
at ~B65 (even in-coverage far starts fail).
"""
import numpy as np
import matplotlib.pyplot as plt

B = [0, 40, 50, 60, 70, 80]
curves = {
    "MSE train 0-40": [98, 96, 0,  0,  None, None],
    "MSE train 0-60": [97, 99, 98, 64, None, None],
    "MSE train 0-80": [87, 99, 99, 63, 6,    6],
    "MIP train 0-80": [98, 100, 100, 79, 7,  8],
}
COV = {"MSE train 0-40": 40, "MSE train 0-60": 60, "MSE train 0-80": 80, "MIP train 0-80": 80}
COL = {"MSE train 0-40": "#1f77b4", "MSE train 0-60": "#ff7f0e",
       "MSE train 0-80": "#d62728", "MIP train 0-80": "#2ca02c"}

plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(8.5, 6))
for k, ys in curves.items():
    xs = [b for b, y in zip(B, ys) if y is not None]
    yy = [y for y in ys if y is not None]
    style = "o--" if k.startswith("MIP") else "o-"
    ax.plot(xs, yy, style, color=COL[k], ms=9, lw=2.3, label=k)
    if not k.startswith("MIP"):
        ax.axvline(COV[k], color=COL[k], ls=":", alpha=0.35)

ax.set_xlabel("eval start point  B  (steps before align-done;  B0=at hole,  B80≈pick)")
ax.set_ylabel("insertion success rate (%)")
ax.set_title("Start-point sweep (reset+rerun, MSE, 2000 demos)\ndotted = training coverage edge")
ax.set_ylim(-3, 104)
ax.set_xticks(B)
ax.legend(fontsize=12, title="training start coverage")
ax.grid(True, alpha=0.3)
ax.annotate("horizon wall:\nfar starts fail even\nwhen in training coverage",
            xy=(75, 6), xytext=(55, 35), fontsize=10, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray"))
plt.tight_layout()
out = "analysis/startpoint_sweep.png"
plt.savefig(out, dpi=140)
print("SAVED", out)
