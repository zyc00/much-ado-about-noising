"""Tube-funnel setting figure in the human_setting_fig A/B style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import toytube as T

YS = np.linspace(-0.02, 1.4, 1000)
YLIM = (-0.05, 1.72)
CS = np.array([T.center(y) for y in YS])
HW = np.array([T.half_width(y) for y in YS])
XLIM = (-0.22, 0.30)


def draw_world(ax):
    ax.axhspan(T.Y_TUBE_LO, T.Y_TUBE_TOP, color="tab:red", alpha=0.10, zorder=0)
    ax.axhspan(T.Y_FUN_LO + 0.06, T.Y_STR_LO, color="tab:red", alpha=0.10, zorder=0)
    ax.axhspan(T.Y_STR_LO, T.Y_TUBE_LO, color="tab:purple", alpha=0.07, zorder=0)
    ax.axhspan(T.Y_DOCK, T.Y_FUN_LO, color="tab:green", alpha=0.08, zorder=0)
    zb = (YS > T.Y_STR_LO) & (YS <= T.Y_TUBE_LO)
    lo = np.where(zb, -T.Z_BOX, CS - HW)
    hi = np.where(zb, T.X_OFF + T.Z_BOX, CS + HW)
    ax.fill_betweenx(YS, lo, hi, color="grey", alpha=0.25, zorder=1)
    ax.plot(lo, YS, color="dimgrey", lw=1.5, zorder=2)
    ax.plot(hi, YS, color="dimgrey", lw=1.5, zorder=2)
    cz = np.array([T.x_target(y, T.T1_FIX, T.T2_FIX) for y in YS])
    ax.plot(cz, YS, "k--", lw=0.7, alpha=0.5, zorder=2)
    ax.plot([0], [0], "k*", ms=16, zorder=6)
    ax.add_patch(plt.Circle((0, 0), T.DOCK_TOL, fill=False, color="k", lw=1.0, zorder=6))
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("x")


def demos(noise_type, n, seed=0):
    T.NOISE_TYPE = noise_type
    rng = np.random.RandomState(seed)
    return [np.array(T.gen_episode(rng)[0]) for _ in range(n)]


fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 7.6))

# --- A: geometry + clean expert paths -------------------------------------
draw_world(axA)
for s in demos("none", 8):
    axA.plot(s[:, 0], s[:, 1], "-", color="tab:blue", lw=1.1, alpha=0.85, zorder=4)
axA.set_title("A  geometry + clean expert paths (eval conditions; expert SR 1.00)",
              fontsize=11, loc="left")
axA.set_ylabel("height $y$")
axA.text(0.028, 0.82, "tube (2cm; 0.01 = 1cm):\nspeed sessions put the\npace mixture "
         r"$\eta\!\cdot\!v_y$, $\eta\sim$" + "\nclip(1+1.2 t(2)), pauses 0.12",
         fontsize=9, color="tab:red", va="center")
axA.text(0.028, 0.45, "funnel (2cm$\\to$5mm):\ndir sessions put 0.07 t(2)\n"
         "lateral tremor here", fontsize=9, color="tab:red", va="center")
axA.text(0.028, 0.15, "insert band: clean dogleg\ncenter-line, SCRAPE walls\n"
         "5mm$\\to$2mm, dock tol 2mm", fontsize=9, color="tab:green",
         va="center")

# --- B: training demonstrations (both noises, disjoint bands) -------------
draw_world(axB)
for s in demos("zturn", 14, seed=1):
    axB.plot(s[:, 0], s[:, 1], "-", lw=0.9, alpha=0.8, zorder=4)
    axB.plot(s[:, 0], s[:, 1], ".", ms=2.2, alpha=0.55, zorder=4)
axB.set_title("B  zturn demos: each picks its own out/back turn heights in the\n"
              "wide Z band; funnel/insert below unchanged",
              fontsize=11, loc="left")
axB.text(0.028, 0.82, "pace noise: spatially\ninvisible, shows as uneven\n"
         "step spacing / pauses", fontsize=9, color="tab:red", va="center")
axB.text(0.028, 0.45, "tremor: wall-bounded\nposition wiggle; action\nlabels keep "
         "raw t(2) kicks", fontsize=9, color="tab:red", va="center")
axB.text(0.028, 0.15, "noise ends above the\nprecision band; demos\nre-center, "
         "track the dogleg", fontsize=9, color="tab:green", va="center")

fig.suptitle("Tube-funnel toy, the setting: speed noise in a thin tube, direction "
             "noise in the funnel it feeds,\nboth upstream of a clean precision "
             "insert with a tight dock", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("tube_setting_AB.png", dpi=160)
print("wrote tube_setting_AB.png")
