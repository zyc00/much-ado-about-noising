"""Per-input-dimension Jacobian gain profiles: isotropic or concentrated?"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = np.load("analysis/failvids/dimprof.npz")
ARMS = ["L2-200", "MIP-200", "HT-200", "HG-200", "L2-2k", "MIP-2k"]
C = {"L2-200": "#d62728", "MIP-200": "#7f7f7f", "HT-200": "#2ca02c",
     "HG-200": "#1f77b4", "L2-2k": "#e8927c", "MIP-2k": "#bbbbbb"}
GRP = [("object", 0, 44), ("eef pos", 44, 47), ("eef quat", 47, 51),
       ("grip", 51, 53)]
fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.6))

# --- A: raw per-dim profile (frame 2 of the window), on-support -------------
for band, ax, ttl in [("on", axes[0][0], "A. on-support (d<2)"),
                      ("far", axes[0][1], "B. far off-support (d>10)")]:
    for a in ARMS:
        p = d[f"{a}|{band}|prof"][53:]           # most recent frame
        ax.plot(np.arange(53), p / p.sum(), lw=1.3, color=C[a],
                alpha=0.85, label=a)
    for g, lo, hi in GRP:
        ax.axvline(hi - 0.5, color="0.75", lw=0.8, ls=":")
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.93, g, fontsize=8,
                ha="center", color="0.45")
    ax.axhline(1 / 53, color="k", ls="--", lw=1)
    ax.text(1, 1 / 53 * 1.15, "isotropic (1/53)", fontsize=8)
    ax.set_xlabel("observation dimension (current frame)")
    ax.set_ylabel("share of Jacobian gain")
    ax.set_title(f"{ttl}: gain per input dimension", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0][0].legend(frameon=False, fontsize=8, ncol=2)

# --- C: rank-ordered (Lorenz-style) ---------------------------------------
axC = axes[1][0]
for a in ARMS:
    s = d[f"{a}|far|sorted"]
    axC.plot(np.arange(1, 107), np.cumsum(s), lw=1.8, color=C[a], label=a)
axC.plot([1, 106], [1 / 106, 1.0], "k--", lw=1, label="isotropic")
axC.set_xlabel("number of input dimensions (rank-ordered)")
axC.set_ylabel("cumulative share of Jacobian gain")
axC.set_title("C. concentration curve, far off-support:\n"
              "how many dims carry the response", fontsize=10.5)
axC.legend(frameon=False, fontsize=8, ncol=2)
axC.spines[["top", "right"]].set_visible(False)

# --- D: top-1 share and dims-for-50% --------------------------------------
axD = axes[1][1]
x = np.arange(len(ARMS))
t_on = [d[f"{a}|on|sorted"][0] for a in ARMS]
t_far = [d[f"{a}|far|sorted"][0] for a in ARMS]
axD.bar(x - 0.2, t_on, 0.38, color=[C[a] for a in ARMS], alpha=0.55,
        label="on-support")
axD.bar(x + 0.2, t_far, 0.38, color=[C[a] for a in ARMS], label="far")
axD.axhline(1 / 106, color="k", ls="--", lw=1)
axD.text(-0.4, 1 / 106 * 1.3, "isotropic (1/106)", fontsize=8)
axD.set_xticks(x); axD.set_xticklabels(ARMS, rotation=20, fontsize=8.5)
axD.set_ylabel("share carried by the single top dimension")
axD.set_title("D. top-dimension share: every policy is 3–6× from isotropic;\n"
              "MSE concentrates further off-support, the others spread",
              fontsize=10.5)
axD.legend(frameon=False, fontsize=8.5)
axD.spines[["top", "right"]].set_visible(False)

fig.suptitle("Jacobian gain per observation dimension — far from isotropic, "
             "object-state dominated, and objective-dependent", fontsize=12.5)
fig.tight_layout()
fig.savefig("analysis/paper/dim_gain_profile.png", dpi=150,
            bbox_inches="tight")
print("wrote analysis/paper/dim_gain_profile.png")
