"""Two-row trajectory figure: collected demonstrations (top) vs trained-policy
rollouts (bottom) for the panel-A-exact skew task with the movable endpoint.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = 0.2
SIG = 0.1
K = 0.5
HOLD = 8
XG = 160.0

d = json.load(open("analysis/toy2d_skewwide_b02.json"))

fig, (axT, axB) = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True, sharey=True)
fig.suptitle(
    "Collected data (top) vs trained policies (bottom): wide-majority skew (b=0.2, sigma=0.1) separates all three",
    fontsize=17, fontweight="bold",
)


def draw_funnel(ax):
    xs = np.linspace(0, XG, 200)
    hw = 5.0 + 9.0 * (1.0 - np.clip(xs / 130.0, 0.0, 1.0))
    ax.fill_between(xs, -hw, hw, color="#dfe7f0", zorder=0)
    ax.plot(xs, hw, color="0.45", lw=1.2)
    ax.plot(xs, -hw, color="0.45", lw=1.2)
    for dk, c, lab in [(0.0, "#2b6cb0", "mean (0)"),
                       (-2.0, "#7c3aed", "MIP posterior (-2)"),
                       (-4.0, "#c23b22", "mode (-4)")]:
        ax.axhline(dk, color=c, lw=0.8, ls="--", zorder=1, alpha=0.7)
        ax.text(161, dk, lab, fontsize=10, color=c, va="center")
    ax.set_xlim(0, XG + 2)
    ax.set_ylim(-15, 15)
    ax.set_ylabel("lateral position y (mm)", fontsize=11.5)


# ---- top: collected demonstrations ----
draw_funnel(axT)
rng = np.random.default_rng(7)
for i in range(30):
    y = float(rng.uniform(-11, 11))
    x = 0.0
    tx, ty = [x], [y]
    step = 0
    r = 0.0
    while x < XG:
        if step % HOLD == 0:
            if rng.random() < 0.2:
                r = (4 * B) * 10.0
            else:
                r = (-B + rng.normal(0, SIG)) * 10.0
        y = y + (-K * y + r)
        x += 4.0
        tx.append(x)
        ty.append(y)
        step += 1
    axT.plot(tx, ty, color="tab:orange", alpha=0.55, lw=1.1, zorder=3)
axT.set_title("A   Collected demonstrations: majority draws -b + N(0, sigma) (a BAND around -4 mm), minority +4b kicks; zero-mean labels",
              fontsize=12.5, fontweight="bold", loc="left")

# ---- bottom: trained rollouts ----
draw_funnel(axB)
for k, n, c in [("l2", "Regression (mean family)", "#2b6cb0"),
                ("mip_full", "MIP full", "#7c3aed"),
                ("ht2", "HT nu=2", "#c23b22")]:
    for t in d["arms"][k]["seed0"]["traces"]:
        axB.plot(t["x"], t["y"], color=c, alpha=0.6, lw=1.3, zorder=3)
    axB.plot([], [], color=c, lw=2.4, label=n)
axB.set_title("B   Trained policies (seed 0): L2 to the mean (0), MIP full to the posterior edge (~-2), HT to the mode (~-4) — three distinguishable bundles",
              fontsize=12.5, fontweight="bold", loc="left")
axB.set_xlabel("forward position x (mm)", fontsize=11.5)
axB.legend(fontsize=10.5, loc="upper right")

fig.tight_layout(rect=[0, 0.01, 1, 0.97])
fig.savefig("analysis/paper/toy_skew_data_vs_trained_wide.png", dpi=160)
fig.savefig("analysis/paper/toy_skew_data_vs_trained_wide.pdf")
print("saved analysis/paper/toy_skew_data_vs_trained_wide.png")
