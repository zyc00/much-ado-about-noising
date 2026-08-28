"""Design visualization for the nuisance cell (before any training).

Left:  what the demonstrator EXECUTED (clean servo, every episode docks) with
       the RECORDED chunks drawn on top — each carries a skewed disturbance.
Right: the disturbance distribution — zero mean, but its MODE is at -b, which
       is what separates the three families:
         MSE  -> the mean (0)      = the correct action
         HT   -> the mode (-b)     = a constant lateral bias
         Diffusion -> samples it   = re-injects the noise at execution

Usage: python scripts/fig_nuisance_design.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = float(os.environ.get("NU_B", "0.05"))
ACTION_MM, K, FWD, XG, HORIZON = 10.0, 0.5, 4.0, 160.0, 8
DOCK = 0.5


def half_width(x):
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


rng = np.random.default_rng(3)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 4.8),
                               gridspec_kw={"width_ratios": [2.2, 1]})
fig.suptitle("Nuisance cell design: a zero-mean but SKEWED operator disturbance "
             "(executed motion is clean, so the oracle is 100%)",
             fontsize=13.5, fontweight="bold")

# ---------------- left: corridor, executed demos + recorded chunks
xs = np.linspace(0, XG, 200)
axL.fill_between(xs, -half_width(xs), half_width(xs), color="#e8eef6", zorder=0)
axL.plot(xs, half_width(xs), color="0.5", lw=1.0)
axL.plot(xs, -half_width(xs), color="0.5", lw=1.0)
axL.axhspan(-DOCK, DOCK, xmin=0.93, color="tab:green", alpha=0.25, zorder=1)
axL.plot(XG, 0, "*", color="tab:green", ms=16, zorder=8)

for i in range(14):
    y = float(rng.uniform(-10, 10)); x = 0.0
    tx, ty = [x], [y]
    while x < XG:
        # executed: clean servo
        y_next = y - K * y
        # recorded chunk for this step: clean + skewed disturbance
        d = (-B if rng.random() < 0.8 else 4.0 * B) * ACTION_MM
        if i < 6 and int(x) % (FWD * HORIZON) == 0:
            axL.plot([x, x + FWD], [y, y_next + d], color="tab:red",
                     alpha=0.55, lw=1.6, zorder=4)
        y = y_next; x += FWD
        tx.append(x); ty.append(y)
    axL.plot(tx, ty, color="0.45", alpha=0.75, lw=1.1, zorder=3)

axL.plot([], [], color="0.45", lw=1.6, label="EXECUTED motion (clean servo — every demo docks)")
axL.plot([], [], color="tab:red", lw=1.6, label="RECORDED action (clean + disturbance)")
axL.legend(fontsize=9, loc="lower left")
axL.set_xlim(0, XG + 4); axL.set_ylim(-15, 15)
axL.set_xlabel("forward x (mm)", fontsize=10.5)
axL.set_ylabel("lateral y (mm)", fontsize=10.5)
axL.set_title("A   The demonstrations", fontsize=11.5, fontweight="bold", loc="left")

# ---------------- right: the disturbance distribution
d = np.where(rng.random(200000) < 0.8, -B, 4.0 * B) * ACTION_MM
axR.hist(d, bins=60, color="0.7", edgecolor="0.4")
axR.axvline(d.mean(), color=C_MEAN if (C_MEAN := "#2b6cb0") else None, lw=2.6,
            label=f"MEAN = {d.mean():+.2f} mm  → MSE target (correct)")
axR.axvline(-B * ACTION_MM, color="#c23b22", lw=2.6, ls="--",
            label=f"MODE = {-B * ACTION_MM:+.2f} mm  → HT target (biased)")
axR.set_yscale("log")
axR.set_xlabel("per-chunk lateral disturbance (mm)", fontsize=10.5)
axR.set_ylabel("count (log)", fontsize=10.5)
axR.legend(fontsize=9, loc="upper center")
axR.set_title("B   80% at −b, 20% at +4b  ⇒  zero mean, mode ≠ mean",
              fontsize=11.5, fontweight="bold", loc="left")

fig.tight_layout(rect=[0, 0.01, 1, 0.93])
fig.savefig("analysis/paper/nuisance_design.png", dpi=170)
fig.savefig("analysis/paper/nuisance_design.pdf")
print("saved analysis/paper/nuisance_design.png")
