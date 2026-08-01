"""Figure: L2 boundary retreat in the settle region (PART CCCXXXIV data).

Panel A: schematic of the settle->stroke interpolation axis with measured
flip positions (L2 @60k peak, L2 @300k, HG @300k) — shows the boundary
being pushed back toward the settle anchor under continued MSE training.
Panel B: flip position alpha* vs training step for L2 / HG / HT (+ MIP
at 300k), the measured formation-then-retreat trajectories.
"""
import matplotlib.pyplot as plt
import numpy as np

STEPS = np.array([20, 60, 100, 140, 180, 220, 260, 300])  # x1000
L2 = np.array([0.615, 0.656, 0.562, 0.526, 0.491, 0.469, 0.492, 0.489])
HG = np.array([0.618, 0.715, 0.740, 0.738, 0.745, 0.730, 0.722, 0.720])
HT = np.array([0.608, 0.664, 0.701, 0.698, 0.685, 0.681, 0.680, 0.680])
MIP_300 = 0.674

C = {"L2": "#d62728", "HG": "#1f77b4", "HT": "#2ca02c", "MIP": "#7f7f7f"}

# L2 global training loss, geometric mean per 10k window (metrics.jsonl)
LSTEP = np.arange(10, 301, 10)
LLOSS = np.array([1.726e-3, 1.286e-3, 1.109e-3, 9.658e-4, 8.630e-4, 7.997e-4,
                  7.169e-4, 6.426e-4, 5.292e-4, 4.194e-4, 3.142e-4, 2.332e-4,
                  1.818e-4, 1.483e-4, 1.204e-4, 9.139e-5, 7.554e-5, 5.615e-5,
                  4.559e-5, 2.904e-5, 2.411e-5, 1.555e-5, 9.723e-6, 7.242e-6,
                  4.458e-6, 3.635e-6, 3.016e-6, 2.794e-6, 2.816e-6, 2.916e-6])

fig = plt.figure(figsize=(11.5, 6.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.6], height_ratios=[1.35, 1.0],
                      hspace=0.12, wspace=0.22)
axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 1], sharex=axB)

# ---- Panel A: interpolation-axis schematic -------------------------------
axA.set_xlim(0, 1)
axA.set_ylim(0, 1)
axA.axis("off")
y0 = 0.40
axA.annotate("", xy=(0.99, y0), xytext=(0.01, y0),
             arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"))
axA.text(0.03, y0 - 0.08, "settle\nanchor", ha="center", va="top", fontsize=10)
axA.text(0.95, y0 - 0.08, "stroke\nanchor", ha="center", va="top", fontsize=10)
axA.text(0.5, y0 - 0.26, r"interpolation coordinate $\alpha$",
         ha="center", va="top", fontsize=10)

# boundary marks, staggered label heights to avoid collisions
marks = [("L2 @ 300k", 0.489, C["L2"], 0.56, "-", "right"),
         ("L2 @ 60k", 0.656, C["L2"], 0.72, "--", "center"),
         ("HG @ 300k", 0.720, C["HG"], 0.88, "-", "left")]
for label, x, c, ytop, ls, ha in marks:
    axA.plot([x, x], [y0 - 0.04, ytop - 0.02], color=c, lw=2, ls=ls)
    xt = x - 0.02 if ha == "right" else (x + 0.02 if ha == "left" else x)
    axA.text(xt, ytop, f"{label}:  " + rf"$\alpha^*={x:.3f}$",
             ha=ha, va="bottom", fontsize=9.5, color=c)

axA.annotate("", xy=(0.489 + 0.008, y0 + 0.07), xytext=(0.656 - 0.008, y0 + 0.07),
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color=C["L2"]))
axA.text(0.572, y0 + 0.105, "pushed $-0.17$", ha="center", va="bottom",
         fontsize=9.5, color=C["L2"], fontweight="bold")
axA.fill_betweenx([y0 - 0.030, y0 + 0.030], 0.0, 0.489,
                  color="0.87", zorder=0)
axA.text(0.24, y0 + 0.042, "settle basin @300k (L2)", fontsize=8.5,
         color="0.35", ha="center")
axA.set_title("A. settle$\\to$stroke axis: where the boundary sits",
              fontsize=11, pad=18)

# ---- Panel B: formation-then-retreat trajectories ------------------------
for name, y in [("HG", HG), ("HT", HT), ("L2", L2)]:
    axB.plot(STEPS, y, "-o", color=C[name], lw=2, ms=4.5)
    axB.text(STEPS[-1] + 6, y[-1], name, color=C[name], fontsize=11,
             va="center", fontweight="bold")
axB.plot([300], [MIP_300], "s", color=C["MIP"], ms=6)
axB.text(306, MIP_300 - 0.012, "MIP", color=C["MIP"], fontsize=10, va="top",
         fontweight="bold")

axB.annotate("common formation\n(~0.61 at 20k, all arms)",
             xy=(22, 0.610), xytext=(55, 0.470), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1), color="0.25")
axB.annotate("", xy=(225, 0.478), xytext=(68, 0.650),
             arrowprops=dict(arrowstyle="-|>", lw=1.6, color=C["L2"],
                             linestyle=(0, (4, 3)), shrinkA=2, shrinkB=4,
                             connectionstyle="arc3,rad=-0.12"))
axB.text(150, 0.600, "retreat $-0.17$\n(residual-proportional gradient dies)",
         fontsize=9, color=C["L2"], ha="center")
axB.text(185, 0.760, "hold for 200k+ steps (equalized gradients)",
         fontsize=9, color=C["HG"], ha="center")

axB.set_ylabel(r"flip position $\alpha^*$ (settle$\to$stroke)", fontsize=10)
axB.set_xlim(10, 335)
axB.set_ylim(0.44, 0.78)
axB.spines[["top", "right"]].set_visible(False)
axB.tick_params(labelbottom=False)
axB.set_title("B. boundary position vs training step (f2i, 30 pairs)",
              fontsize=11)

# ---- Panel C: L2 global training loss, aligned with panel B --------------
axC.semilogy(LSTEP, LLOSS, "-", color=C["L2"], lw=2)
axC.set_ylabel("L2 training loss\n(global batch MSE)", fontsize=10)
axC.set_xlabel("training step ($\\times 10^3$)", fontsize=10)
axC.spines[["top", "right"]].set_visible(False)
axC.set_ylim(1.5e-6, 3e-3)

for ax in (axB, axC):
    ax.axvline(60, color="0.55", lw=1, ls=":")
    ax.axvspan(240, 335, color="0.5", alpha=0.08, zorder=0)
axC.annotate("$8\\times 10^{-4}$ at the\nboundary peak (60k)",
             xy=(60, 8.0e-4), xytext=(95, 1.3e-3), fontsize=8.5, color="0.25",
             arrowprops=dict(arrowstyle="->", color="0.4", lw=1))
axC.text(287, 5.5e-6, "floor $2.9\\times 10^{-6}$\n(boundary fully\nrelaxed)",
         fontsize=8.5, color="0.25", ha="center")
axC.text(160, 1.6e-4,
         "separating gradient $\\propto$ residual $\\to$ 0:\n"
         "retreat proceeds as the loss collapses $\\sim$100$\\times$",
         fontsize=9, color=C["L2"], ha="center")

fig.suptitle("MSE forms the settle boundary, then pushes it back as its "
             "loss (and gradient) collapses to the floor", fontsize=12, y=0.99)
out = "analysis/paper/boundary_retreat_fig"
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png/.pdf")
