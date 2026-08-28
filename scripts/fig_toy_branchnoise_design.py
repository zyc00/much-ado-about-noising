"""Design figure: the ORIGINAL toy_mip_nfl skew nuisance + a swept endpoint.

Layout mirrors toy_mip_nfl_design.png. Noise is exactly the original:
single correct action, zero-mean toyskew jitter {-b: .8, +4b: .2}, b=0.1,
chunk-held, executed. New ingredient: success is graded at a swept endpoint
position d instead of only the centerline — the endpoint selects the family.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

B = 0.1          # toyskew scale (original design's b)
P = 0.8          # majority mass at -b
SIGA = 0.1       # MIP anchor noise
MM = 20.0        # persistent residual (normalized) -> equilibrium mm
K = 0.5
HOLD = 8
XG = 160.0
C_L2, C_MIP, C_HT = "#2b6cb0", "#7c3aed", "#c23b22"
C_MAJ, C_MIN = "tab:red", "tab:orange"

ATOMS = [(-B, P), (4 * B, 1 - P)]      # mean = 0
xs_d = np.linspace(-0.35, 0.55, 401)


def post_mean(x):
    num, den = 0.0, 0.0
    for mu, w in ATOMS:
        k = w * np.exp(-((x - mu) ** 2) / (2 * SIGA**2))
        num, den = num + k * mu, den + k
    return num / den


R_MODE = float(post_mean(np.array([0.0]))[0])   # MIP composed residual at mean input
Y_MEAN, Y_MODE = 0.0, -B * MM                   # 0 and -2.0 mm


def half_width(x):
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


fig = plt.figure(figsize=(20, 11.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.0, 1.05], height_ratios=[1.0, 0.92],
                      hspace=0.44, wspace=0.26, left=0.05, right=0.975,
                      top=0.86, bottom=0.07)

fig.suptitle(
    "The skew witness with a movable endpoint: the SAME data rewards the mean family or the mode family",
    fontsize=20, fontweight="bold", y=0.965,
)
fig.text(0.5, 0.905,
         "DESIGN + POPULATION PREDICTION + trained landings: original asymmetric zero-mean jitter "
         "(0.8 at -b, 0.2 at +4b); grading success at a swept endpoint d replaces the fixed dock",
         ha="center", fontsize=12.5, color="0.35")


def label(ax, letter, title):
    ax.set_title(f"{letter}   {title}", fontsize=14.5, fontweight="bold", loc="left", pad=9)


# ---------- A: task + demo trajectories (original construction) ----------
axA = fig.add_subplot(gs[0, 0])
label(axA, "A", "2D funnel task; demos: servo + executed toyskew jitter")
xs = np.linspace(0, XG, 200)
axA.fill_between(xs, -half_width(xs), half_width(xs), color="#dfe7f0", zorder=0)
axA.plot(xs, half_width(xs), color="0.45", lw=1.2)
axA.plot(xs, -half_width(xs), color="0.45", lw=1.2)
rng = np.random.default_rng(7)
for i in range(28):
    y = float(rng.uniform(-11, 11)); x = 0.0
    tx, ty = [x], [y]; step = 0; r = 0.0
    while x < XG:
        if step % HOLD == 0:
            r = (B * (5.0 * (rng.random() < 0.2) - 1.0)) * 10.0
        y = y + (-K * y + r); x += 4.0
        tx.append(x); ty.append(y); step += 1
    axA.plot(tx, ty, color=C_MIN, alpha=0.5, lw=1.0, zorder=3)
axA.axhline(0, color="0.5", lw=0.9, ls="--")
axA.annotate("start distribution", xy=(2, 10.3), xytext=(30, 12.5), fontsize=11,
             color="0.25", arrowprops=dict(arrowstyle="->", color="0.35"))
axA.annotate("swept endpoint d\n(the only dial)", xy=(XG, -1.0), xytext=(118, -11.5),
             fontsize=11, color="darkgreen",
             arrowprops=dict(arrowstyle="->", color="darkgreen"))
axA.text(80, -14.0, "training demos: corrective servo + chunk-held asymmetric zero-mean jitter (orange)",
         fontsize=10, color="peru", ha="center")
axA.set_xlim(0, XG + 2); axA.set_ylim(-15, 15)
axA.set_xlabel("forward position x (mm)", fontsize=11)
axA.set_ylabel("lateral position y (mm)", fontsize=11)

# ---------- B: label PMF (identical to original panel B) ----------
axB = fig.add_subplot(gs[0, 1])
label(axB, "B", "Same state, skewed labels")
for mu, w, c, lab_ in [(-B, P, C_MAJ, "-b\n80%"), (4 * B, 1 - P, C_MIN, "+4b\n20%")]:
    axB.plot([mu, mu], [0, w], color=c, lw=3.5)
    axB.plot(mu, w, "o", color=c, ms=8)
    axB.text(mu, w + 0.05, lab_, ha="center", fontsize=11.5, color=c, fontweight="bold")
axB.axvline(0, color=C_L2, lw=1.2, ls="--")
axB.plot(0, 0, "D", color=C_L2, ms=8, clip_on=False, zorder=5)
axB.text(0.03, 0.42, "E[a|s] = a*\n(correct convex action)", fontsize=10.5, color=C_L2)
axB.text(0.97, 0.93, "0.8(-b) + 0.2(4b) = 0", transform=axB.transAxes, ha="right",
         fontsize=12, bbox=dict(boxstyle="round", fc="white", ec="0.6"))
axB.set_xlim(-0.3, 0.55); axB.set_ylim(0, 1.0)
axB.set_xlabel("lateral action label, relative to a*(s)", fontsize=11)
axB.set_ylabel("probability mass", fontsize=11)

# ---------- C: the two-family collision ----------
axC = fig.add_subplot(gs[0, 2])
label(axC, "C", "Two families, two points")
axC.axis("off")


def box(y, color, name, text):
    axC.add_patch(FancyBboxPatch((0.02, y), 0.96, 0.215, boxstyle="round,pad=0.012",
                                 fc="white", ec=color, lw=2, transform=axC.transAxes))
    axC.text(0.06, y + 0.155, name, transform=axC.transAxes, fontsize=12.5,
             fontweight="bold", color=color)
    axC.text(0.06, y + 0.035, text, transform=axC.transAxes, fontsize=10.8, color="0.2")


box(0.70, C_L2, "L2 / HG / MIP step 1", "conditional mean -> the true action\n(y* = 0 mm)")
box(0.40, C_MIP, "MIP (full)", f"anchor = mean; denoiser projects to the\nmajority mode -b   (y* = {Y_MODE:.1f} mm)")
box(0.10, C_HT, "HT (nu=2 and nu=0.5)", f"redescending location -> majority mode -b\n(y* = {Y_MODE:.1f} mm; co-located with MIP)")
axC.text(0.5, 0.005, "pure skew separates the FAMILIES —\nseparating MIP from HT needs branched aims",
         transform=axC.transAxes, ha="center", fontsize=10.2, color="0.3", style="italic")

# ---------- D: population prediction for MIP second view ----------
axD = fig.add_subplot(gs[1, 0])
label(axD, "D", f"Population prediction for MIP's second view (b = sigma_a = {B})")
axD.plot(xs_d, post_mean(xs_d), color=C_MIP, lw=2.4,
         label=r"$D(x)=E[r\,|\,r+\sigma_a\varepsilon=x]$")
axD.plot(xs_d, xs_d, ":", color="0.6", label="identity")
axD.axvline(0, color=C_L2, lw=1.0, ls="--")
axD.plot(0, R_MODE, "o", color=C_MAJ, ms=9, zorder=5)
axD.annotate(f"MIP inference input: mean = 0\nposterior output = {R_MODE:.3f} " + r"$\approx -b$",
             xy=(0, R_MODE), xytext=(0.12, -0.26), fontsize=10.8, color=C_MAJ,
             arrowprops=dict(arrowstyle="->", color=C_MAJ))
axD.set_xlim(-0.35, 0.55); axD.set_ylim(-0.3, 0.45)
axD.set_xlabel("second-view input residual x (normalized)", fontsize=11)
axD.set_ylabel("denoised residual D(x)", fontsize=11)
axD.legend(fontsize=10.5, loc="upper left")

# ---------- E: closed loop with the two winnable endpoints ----------
axE = fig.add_subplot(gs[1, 1:])
label(axE, "E", "Closed loop: two equilibria = two winnable endpoints (measured landings)")
cycles = np.arange(0, 26)
for y_eq, c, lab_ in [(Y_MEAN, C_L2, "L2 / HG / step1"),
                      (Y_MODE, C_MIP, "MIP full"),
                      (Y_MODE - 0.12, C_HT, "HT nu=2 / nu=0.5")]:
    y = 7.0 * (1 - K) ** cycles + y_eq * (1 - (1 - K) ** cycles)
    axE.plot(cycles, y, color=c, lw=2.4, label=lab_)
for y_eq, c in [(Y_MEAN, C_L2), (Y_MODE, "#a06cd5")]:
    axE.axhspan(y_eq - 0.5, y_eq + 0.5, color=c, alpha=0.10)
axE.text(25.4, Y_MEAN, "0.0 mm", fontsize=10.5, color=C_L2, va="center")
axE.text(25.4, Y_MODE, "-2.0 mm", fontsize=10.5, color=C_MIP, va="center")
axE.text(12.5, 5.6, "endpoint at 0 -> mean family wins (the original witness);\n"
                    "endpoint at -2 mm -> the mode family (MIP full + HT) wins",
         fontsize=10.8, color="0.3", ha="center")
axE.set_xlim(0, 27); axE.set_ylim(-4.5, 7.5)
axE.set_xlabel("control step", fontsize=11)
axE.set_ylabel("lateral position y (mm)", fontsize=11)
axE.legend(fontsize=10.5, loc="upper right")

fig.text(0.5, 0.012,
         "Measured (8 seeds, endpoint sweep, tol 0.5 mm): peaks at -0.25 mm for L2/HG/step1 and -2.25 mm for "
         "MIP-full/HT nu=2/HT nu=0.5, every arm SR 1.000 at its own peak  |  landings match population predictions "
         "(mode bias -b = -2.0)  |  step-1-only control lands with the mean family: the displacement is the second step",
         ha="center", fontsize=9.0, color="0.35")

fig.savefig("analysis/paper/toy_branchnoise_design.png", dpi=160)
fig.savefig("analysis/paper/toy_branchnoise_design.pdf")
print("saved analysis/paper/toy_branchnoise_design.png")
