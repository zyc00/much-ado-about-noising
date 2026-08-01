"""Figure: per-region loss and gradient balance over L2 training
(probe_grad_balance.py data; companion to boundary_retreat_fig)."""
import matplotlib.pyplot as plt
import numpy as np

S = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
L_SET = np.array([1.561e-3, 2.138e-3, 2.110e-3, 1.571e-3, 3.069e-3, 4.652e-3,
                  3.753e-3, 1.722e-3, 5.760e-3, 4.884e-3, 6.285e-3, 6.370e-3,
                  6.360e-3, 6.365e-3, 6.369e-3])
L_STR = np.array([2.354e-2, 2.498e-2, 2.419e-2, 2.264e-2, 2.307e-2, 2.296e-2,
                  2.285e-2, 2.287e-2, 2.318e-2, 2.323e-2, 2.301e-2, 2.296e-2,
                  2.301e-2, 2.295e-2, 2.299e-2])
G_SET = np.array([1.265e-1, 6.311e-1, 8.072e-1, 3.930e-1, 2.618, 2.974, 4.114,
                  5.979e-1, 1.222, 6.930, 1.702e-1, 1.814e-1, 3.100e-1,
                  1.498e-1, 1.345e-1])
G_STR = np.array([1.164, 9.694e-1, 8.237e-1, 7.186e-1, 6.881e-1, 6.590e-1,
                  6.408e-1, 6.416e-1, 6.389e-1, 6.377e-1, 6.393e-1, 6.416e-1,
                  6.461e-1, 6.488e-1, 6.496e-1])
CEIL = 6.37e-3

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True,
                               gridspec_kw={"hspace": 0.12})

# phase shading (both panels)
for ax in (ax1, ax2):
    ax.axvspan(10, 70, color="#2ca02c", alpha=0.05, zorder=0)
    ax.axvspan(70, 210, color="#ff7f0e", alpha=0.06, zorder=0)
    ax.axvspan(210, 310, color="0.5", alpha=0.08, zorder=0)
    ax.axvline(60, color="0.55", lw=1, ls=":")

ax1.semilogy(S, L_STR, "-o", color="#7f3f9f", lw=2, ms=4)
ax1.semilogy(S, L_SET, "-o", color="#d62728", lw=2, ms=4)
ax1.axhline(CEIL, color="0.4", lw=1, ls="--")
ax1.text(302, L_STR[-1], "stroke windows", color="#7f3f9f", fontsize=10,
         va="center", ha="left")
ax1.text(302, L_SET[-1] * 0.78, "settle-core\nwindows", color="#d62728",
         fontsize=10, va="center", ha="left")
ax1.text(146, CEIL * 1.18, "settle-cluster variance ceiling "
         r"($6.4\times10^{-3}$: output = cluster mean)",
         fontsize=8.5, color="0.3", ha="center")
ax1.annotate("fit ERODES: residual rises 4$\\times$\nto the ceiling",
             xy=(220, 6.29e-3), xytext=(215, 2.1e-3), fontsize=9,
             color="#d62728", ha="center",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
ax1.set_ylabel("region loss (batch MSE)", fontsize=10)
ax1.set_ylim(8e-4, 4.2e-2)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title("L2: per-region loss and gradient over training "
              "(128 settle-core / 128 stroke windows)", fontsize=11)

R = G_SET / G_STR
ax2.semilogy(S, R, "-o", color="#1f1f1f", lw=2, ms=4)
ax2.axhline(1.0, color="0.55", lw=1, ls="--")
ax2.text(302, 1.0, "parity", fontsize=9, color="0.35", va="center")
ax2.set_ylabel(r"gradient ratio  $\|g_{settle}\|/\|g_{stroke}\|$", fontsize=10)
ax2.set_xlabel("training step ($\\times 10^3$)", fontsize=10)
ax2.set_ylim(0.07, 15)
ax2.spines[["top", "right"]].set_visible(False)

ax2.annotate("balance reached at the\nboundary peak (0.98 @60k)",
             xy=(60, 0.98), xytext=(95, 0.16), fontsize=9, color="#2ca02c",
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
ax2.text(150, 6.5, "conflict: settle gradient large & volatile,\n"
         "aggregate descent overrides it (residual rises)",
         fontsize=9, color="#e07000", ha="center")
ax2.annotate("extinguished\n(0.2–0.5)", xy=(280, 0.23), xytext=(240, 0.105),
             fontsize=9, color="0.3",
             arrowprops=dict(arrowstyle="->", color="0.4", lw=1.2))

lab_y = 12.4
ax2.text(40, lab_y, "I. formation", fontsize=9, ha="center", color="#2ca02c",
         fontweight="bold")
ax2.text(140, lab_y, "II. crowding (boundary recedes)", fontsize=9,
         ha="center", color="#e07000", fontweight="bold")
ax2.text(258, lab_y, "III. capitulation", fontsize=9, ha="center",
         color="0.35", fontweight="bold")

fig.savefig("analysis/paper/grad_balance_fig.png", dpi=200,
            bbox_inches="tight")
fig.savefig("analysis/paper/grad_balance_fig.pdf", bbox_inches="tight")
print("saved analysis/paper/grad_balance_fig.png/.pdf")
