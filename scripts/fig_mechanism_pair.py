"""Trajectory-level mechanism illustration, seed 21025: natural L2 failure
vs same-placement MIP success. Aligned signals: d(t), rotation-action
error vs ground truth, frame misorientation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.load("analysis/pairpack.npz", allow_pickle=True)
SD = 21025
T0, T1 = 110, 255


def med5(x):
    return np.array([np.median(x[max(0, i - 2):i + 3])
                     for i in range(len(x))])


fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.4), sharex=True)

ax = axes[0]
for arm, c, lab in [("l2", "tab:red", "L2 (failure)"),
                    ("mip", "tab:green", "MIP (success)")]:
    d = z[f"{arm}{SD}_dser"]
    ax.plot(np.arange(len(d)), np.maximum(d, 0.4), color=c, lw=1.8,
            label=lab)
ax.axhspan(2, 4, color="orange", alpha=0.12)
ax.axhline(2, color="k", ls=":", lw=0.8)
ax.axhline(4, color="k", ls="--", lw=0.8)
ax.text(T0 + 2, 2.1, "band [2,4): slightly off-support", fontsize=8.5)
ax.set_yscale("log")
ax.set_ylim(0.4, 60)
ax.set_ylabel("d(t): distance to nearest\ntraining state (log)")
ax.legend(fontsize=9, loc="upper left")
ax.set_title("A  state deviation — L2 enters the reorientation phase at "
             "d$\\approx$2.5 and returns to the tube $\\approx$40 steps "
             "after MIP; from $\\approx$205 it leaves and does not return",
             fontsize=10.5)
ax.grid(alpha=0.25)

ax = axes[1]
for arm, c in [("l2", "tab:red"), ("mip", "tab:green")]:
    e = med5(z[f"{arm}{SD}_erot"])
    ax.plot(np.arange(len(e)), e, color=c, lw=1.8)
ax.set_ylim(0, 3.1)
ax.set_ylabel("rotation-action error vs expert\n(local reconstruction, "
              "5-step median)")
ax.set_title("B  the fork — L2's rotation actions deviate from the expert "
             "schedule for $\\approx$50 steps (error 1.5; the "
             "reorientation lags); MIP matches it within 3 steps "
             "($\\approx$0.05)", fontsize=10.5)
ax.grid(alpha=0.25)

ax = axes[2]
for arm, c in [("l2", "tab:red"), ("mip", "tab:green")]:
    a = z[f"{arm}{SD}_ang"]
    ax.plot(np.arange(len(a)), a, color=c, lw=1.8)
    if int(z[f"{arm}{SD}_asm"]):
        ax.plot(len(a) - 1, a[-1], marker="*", color=c, ms=14, ls="none")
ax.axhline(10, color="k", ls="--", lw=0.8)
ax.text(T0 + 2, 11, "$\\approx$10$^\\circ$ insertion tolerance",
        fontsize=8.5)
ax.set_yscale("log")
ax.set_ylim(3, 200)
ax.set_ylabel("frame misorientation\n(deg, log)")
ax.set_xlabel("episode step")
ax.set_title("C  consequence — MIP aligns by 150 and inserts at 193 (*); "
             "L2 aligns 30 steps later, does not complete, the alignment "
             "degrades from $\\approx$205, the late attempt misses",
             fontsize=10.5)
ax.grid(alpha=0.25)
ax.set_xlim(T0, T1)

fig.suptitle(
    "Script data, one natural episode pair (seed 21025, identical "
    "placement, no perturbation): trajectory-level view of the mechanism",
    fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig("analysis/paper/mechanism_pair.png", dpi=150,
            bbox_inches="tight")
print("saved analysis/paper/mechanism_pair.png")
