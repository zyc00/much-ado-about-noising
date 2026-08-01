import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("analysis/manifold/jacmap.npz")
ARMS = ["L2", "HG", "CND2B", "MIP"]
CATS = ["approach", "pocket", "annulus"]
SR = {"L2": 63, "HG": 87, "CND2B": 82, "MIP": 94}

fig, axes = plt.subplots(4, 4, figsize=(21, 12),
                         gridspec_kw={"width_ratios": [3, 3, 3, 1.6]})
blocks = [(0, 44, "object"), (44, 47, "eef pos"), (47, 51, "quat"),
          (51, 53, "grip")]
for r, arm in enumerate(ARMS):
    for c, cat in enumerate(CATS):
        J = z[f"{arm}_{cat}_J"]
        cn = np.linalg.norm(J, axis=1)
        cn2 = np.stack([cn[:, :53], cn[:, 53:]], axis=1)
        m = cn2.mean(0)
        m = m / (m.max() + 1e-12)          # per-panel shape normalization
        ax = axes[r, c]
        ax.imshow(m, aspect="auto", cmap="inferno", vmin=0, vmax=1)
        ax.set_yticks([0, 1], ["frame t-1", "frame t"])
        for b0, b1, lab in blocks:
            ax.axvline(b0 - 0.5, color="cyan", lw=0.7)
            if r == 0 and c == 0:
                ax.text((b0 + b1) / 2, -0.9, lab, ha="center", fontsize=8)
        if r == 0:
            ax.set_title(cat, fontsize=12)
        if c == 0:
            ax.set_ylabel(f"{arm} (SR {SR[arm]})", fontsize=11)
        if r == 3:
            ax.set_xlabel("obs dim (0-52)")
    ax = axes[r, 3]
    for cat, col in zip(CATS, ["#2ca02c", "#d62728", "#9467bd"]):
        sv = z[f"{arm}_{cat}_sv"]
        med = np.median(sv, axis=0)
        ax.semilogy(np.arange(1, 25), med / med[0], color=col, lw=1.5,
                    label=cat if r == 0 else None)
    ax.set_ylim(1e-3, 1.2)
    if r == 0:
        ax.set_title("sv spectrum (normalized)", fontsize=10)
        ax.legend(fontsize=8)
    if r == 3:
        ax.set_xlabel("sv index")
fig.suptitle("Deployed Jacobian maps: input-sensitivity SHAPE (per-panel normalized column norms) + singular spectra",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("analysis/paper/jacmap_fig2.png", dpi=140)
print("PLOT saved")
