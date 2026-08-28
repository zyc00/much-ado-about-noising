"""Trained-results figure for the panel-A-exact skew task + endpoint sweep,
in the style of toy_mip_nfl_results.png.

A: dock success at the two endpoints (0 = original dock, -2 = mode endpoint)
B: landing position per arm vs population predictions
C: representative rollouts (seed 0) in the original funnel geometry

Usage: python scripts/fig_toy_skew_endpoint_results.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("analysis/toy2d_skew_endpoint_orig.json"))
docks = np.array(d["config"]["docks"])

ARMS = [
    ("l2", "Regression", "#2b6cb0"),
    ("hg", "HG", "#63b3ed"),
    ("mip_step1", "MIP step 1", "#b794f4"),
    ("mip_full", "MIP full", "#7c3aed"),
    ("ht2", "HT nu=2", "#c23b22"),
    ("ht05", "HT nu=0.5", "#f6ad55"),
]
D_MEAN, D_MODE = 0.0, -2.0
i_mean = int(np.argmin(np.abs(docks - D_MEAN)))
i_mode = int(np.argmin(np.abs(docks - D_MODE)))

fig = plt.figure(figsize=(20, 10.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.22,
                      left=0.055, right=0.975, top=0.84, bottom=0.08)
fig.suptitle(
    "Trained result on the original skew task: moving the endpoint from 0 to -2 mm reverses every verdict",
    fontsize=20, fontweight="bold", y=0.96,
)
fig.text(0.5, 0.895,
         "TRAINED MODELS — 8 seeds per arm, original panel-A geometry (tight 0.5 mm tolerance), "
         "original toyskew nuisance {-b:.8, +4b:.2}; only the graded endpoint moves",
         ha="center", fontsize=12.5, color="0.35")


def label(ax, letter, title):
    ax.set_title(f"{letter}   {title}", fontsize=14.5, fontweight="bold", loc="left", pad=9)


# ---------- A: success at the two endpoints ----------
axA = fig.add_subplot(gs[0, 0])
label(axA, "A", "Dock success at the two endpoints")
xs = np.arange(len(ARMS))
w = 0.38
for off, idx, hatch, lab in [(-w / 2, i_mean, None, "endpoint at 0 (original dock)"),
                             (w / 2, i_mode, "//", "endpoint at -2 mm (mode)")]:
    vals = [d["arms"][k]["sr_curve_mean"][idx] for k, _, _ in ARMS]
    errs = [d["arms"][k]["sr_curve_std"][idx] for k, _, _ in ARMS]
    cols = [c for _, _, c in ARMS]
    axA.bar(xs + off, vals, w, color=cols, alpha=0.95 if hatch is None else 0.55,
            hatch=hatch, yerr=errs, capsize=3,
            error_kw=dict(ecolor="0.35", lw=1),
            label=lab, edgecolor="0.3", lw=0.6)
axA.set_xticks(xs)
axA.set_xticklabels([n for _, n, _ in ARMS], fontsize=10.5)
axA.set_ylim(0, 1.12)
axA.set_ylabel("dock success (|y - d| < 0.5 mm)", fontsize=11.5)
leg = axA.legend(fontsize=10.5, loc="upper center", ncols=2)
for t, txt in zip(leg.get_texts(), ["endpoint at 0 (original dock)", "endpoint at -2 mm (mode)"]):
    t.set_text(txt)

# ---------- B: landings vs population predictions ----------
axB = fig.add_subplot(gs[0, 1])
label(axB, "B", "Landing position: each arm sits on its estimator's point")
for i, (k, n, c) in enumerate(ARMS):
    meds = d["arms"][k]["landing_median_per_seed"]
    axB.scatter([i] * len(meds), meds, color="0.25", s=18, zorder=4, alpha=0.8)
    axB.bar(i, np.median(meds), 0.6, color=c, alpha=0.9, zorder=2)
for y, lab_, c in [(0.0, "population mean prediction (0)", "#2b6cb0"),
                   (-2.0, "population mode prediction (-b x gain = -2.0)", "#c23b22")]:
    axB.axhline(y, color=c, lw=1.0, ls="--", zorder=1)
    axB.text(5.45, y + 0.08, lab_, fontsize=9.5, color=c, ha="right")
axB.set_xticks(xs)
axB.set_xticklabels([n for _, n, _ in ARMS], fontsize=10.5)
axB.set_ylabel("landing y at goal (mm, median/seed)", fontsize=11.5)
axB.set_ylim(-3.0, 1.0)

# ---------- C: representative rollouts ----------
axC = fig.add_subplot(gs[1, :])
label(axC, "C", "Representative rollouts (seed 0), original funnel: both endpoints shown")
xs_f = np.linspace(0, 160, 200)
hw = 5.0 + 9.0 * (1.0 - np.clip(xs_f / 130.0, 0.0, 1.0))
axC.fill_between(xs_f, -hw, hw, color="#dfe7f0", zorder=0)
axC.plot(xs_f, hw, color="0.45", lw=1.2)
axC.plot(xs_f, -hw, color="0.45", lw=1.2)
for dk, c, lab_ in [(0.0, "#2b6cb0", "endpoint 0"), (-2.0, "#c23b22", "endpoint -2")]:
    axC.axhspan(dk - 0.5, dk + 0.5, xmin=0.93, color=c, alpha=0.18, zorder=1)
    axC.text(160.7, dk, lab_, fontsize=10, color=c, va="center")
for k, n, c in [("l2", "Regression", "#2b6cb0"), ("mip_full", "MIP full", "#7c3aed"),
                ("ht2", "HT nu=2", "#c23b22")]:
    for t in d["arms"][k]["seed0"]["traces"]:
        axC.plot(t["x"], t["y"], color=c, alpha=0.6, lw=1.3, zorder=3)
    axC.plot([], [], color=c, lw=2.4, label=n)
axC.set_xlim(0, 166)
axC.set_ylim(-15, 15)
axC.set_xlabel("forward position x (mm)", fontsize=11.5)
axC.set_ylabel("lateral position y (mm)", fontsize=11.5)
axC.legend(fontsize=10.5, loc="upper right")

m = {k: d["arms"][k] for k, _, _ in ARMS}
fig.text(0.5, 0.012,
         f"Landings (median over seeds): Regression {m['l2']['landing_median']:+.2f} / HG {m['hg']['landing_median']:+.2f} / "
         f"step1 {m['mip_step1']['landing_median']:+.2f} / MIP full {m['mip_full']['landing_median']:+.2f} / "
         f"HT nu=2 {m['ht2']['landing_median']:+.2f} / HT nu=0.5 {m['ht05']['landing_median']:+.2f} mm  |  "
         "the original witness graded only the left group of panel A; the right group is the same checkpoints at d = -2",
         ha="center", fontsize=10, color="0.35")

fig.savefig("analysis/paper/toy_skew_endpoint_results.png", dpi=160)
fig.savefig("analysis/paper/toy_skew_endpoint_results.pdf")
print("saved analysis/paper/toy_skew_endpoint_results.png")
