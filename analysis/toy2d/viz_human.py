"""Exhibit figure for the story-proof toy (PART CCCXIX, canonical cell =
round 3). Reads toyhuman_r3.json. Panels: A setting schematic, B SR bands
with per-seed dots (the story ordering), C repricing ledger, D insert-band
fit error, T component checklist vs robot facts."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("toyhuman_r3.json"))
COL = {"l2": "#d62728", "cauchy": "#e8a13c", "hg": "#ff9896", "ht": "#9467bd",
       "globalt": "#8c564b", "mip": "#1f77b4", "flow8": "#17becf"}
NAME = {"l2": "MSE\n(fixed, Gauss)", "cauchy": "Cauchy\n(fixed, heavy)",
        "hg": "hetero-G\n(learned, Gauss)", "ht": "hetero-t\n(learned, heavy)",
        "globalt": "global-t\n(global, heavy)", "mip": "MIP\n(2-step)",
        "flow8": "Flow\n(8-step)"}
ARMS = ("l2", "cauchy", "hg", "ht", "globalt", "mip", "flow8")
SEEDS = range(8)


def vals(arm, key):
    return [R[f"{arm}_s{s}"][key] for s in SEEDS if f"{arm}_s{s}" in R]


fig = plt.figure(figsize=(13.5, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.9], hspace=0.45, wspace=0.32,
                      left=0.06, right=0.98, top=0.86, bottom=0.06)
fig.suptitle("Story-proof toy (canonical cell): heavy-tailed align tremor on the critical "
             "path; fine dogleg + tight dock in the clean band\n"
             "σ = capability, tail = stability; state-conditional σ > global; "
             "repriced regression ≈ anchored flow-map",
             fontsize=11.5, y=0.97)

# --- A: setting schematic --------------------------------------------------
axA = fig.add_subplot(gs[0, 0])
yy = np.linspace(0.05, 0.5, 100)
hw = np.array([0.03 + 0.07 * np.clip((y - 0.05) / 0.45, 0, 1) for y in yy])
cc = np.array([0.02 * np.sin(np.pi * (0.25 - y) / 0.2) if y < 0.25 else 0.0
               for y in yy])
axA.fill_betweenx(yy, cc - hw, cc + hw, color="0.9")
axA.plot(cc - hw, yy, "0.5", lw=1.2)
axA.plot(cc + hw, yy, "0.5", lw=1.2)
axA.plot(cc, yy, "k--", lw=1.0)
axA.axhspan(0.25, 0.5, color="#d62728", alpha=0.12)
axA.axhspan(0.05, 0.25, color="#2ca02c", alpha=0.10)
axA.scatter([0], [0], marker="*", s=150, color="k", zorder=5)
axA.text(0.12, 0.40, "ALIGN band:\nlabel tremor\n$0.08\\cdot t(\\nu{=}2)$\n(critical path)",
         fontsize=8.5, color="#a00000")
axA.text(0.12, 0.13, "clean band:\ndogleg fine\nstructure,\ndock tol .008",
         fontsize=8.5, color="#1a7a1a")
axA.set_xlim(-0.22, 0.30)
axA.set_ylim(-0.03, 0.62)
axA.set_title("A  setting (expert anchor 1.00;\ndemos wiggle, eval clean)",
              fontsize=10, loc="left")
axA.set_xticks([])
axA.set_yticks([])

# --- B: SR bands (the story ordering) --------------------------------------
axB = fig.add_subplot(gs[0, 1:])
for i, arm in enumerate(ARMS):
    v = vals(arm, "sr")
    axB.bar(i, np.mean(v), 0.6, color=COL[arm], alpha=0.75)
    axB.errorbar(i, np.mean(v), yerr=np.std(v), color="k", capsize=4, lw=1.2)
    axB.scatter([i + np.random.RandomState(1).uniform(-0.15, 0.15) for _ in v],
                v, s=16, color="k", alpha=0.55, zorder=5)
    axB.text(i, 0.02, f"{np.mean(v):.2f}±{np.std(v):.2f}", ha="center",
             fontsize=7.5, color="white", weight="bold")
axB.set_xticks(range(len(ARMS)), [NAME[a] for a in ARMS], fontsize=8)
axB.set_ylabel("success rate (8 seeds)")
axB.set_ylim(0, 1.1)
axB.axhline(1.0, color="0.6", lw=0.8, ls=":")
axB.set_title("B  the ordering: MSE low+wide | fixed-robust below | hetero-G peaked-but-"
              "volatile | hetero-t top | global-σ below ht | anchored ≈ ht", fontsize=9.5,
              loc="left")

# --- C: repricing ledger ---------------------------------------------------
axC = fig.add_subplot(gs[1, 0])
for i, arm in enumerate(ARMS):
    v = vals(arm, "led_align")
    axC.bar(i, np.mean(v), 0.6, color=COL[arm], alpha=0.75)
axC.set_xticks(range(len(ARMS)), [a for a in ARMS], fontsize=7.5, rotation=30)
axC.set_ylabel("align share of gradient weight")
axC.set_title("C  repricing ledger: hetero losses\nsuppress the noisy pocket to ~0",
              fontsize=10, loc="left")

# --- D: insert fit error ---------------------------------------------------
axD = fig.add_subplot(gs[1, 1])
for i, arm in enumerate(("l2", "cauchy", "hg", "ht", "globalt")):
    v = vals(arm, "ins_err")
    axD.bar(i, np.mean(v), 0.6, color=COL[arm], alpha=0.75)
    axD.text(i, np.mean(v) + 0.0004, f"{np.mean(v):.4f}", ha="center", fontsize=7)
axD.set_xticks(range(5), ["l2", "cauchy", "hg", "ht", "globalt"], fontsize=8)
axD.set_ylabel("clean-band fit error (vs clean labels)")
axD.set_title("D  the damage is in the fit:\nMSE 4x hetero-t on the clean band",
              fontsize=10, loc="left")

# --- T: component checklist ------------------------------------------------
axT = fig.add_subplot(gs[1, 2])
axT.axis("off")
rows = [
    ("MSE low + seed-split", "✓  .61±.26"),
    ("fixed robust below hetero", "✓  .78±.29"),
    ("hetero-G volatile band", "✓  .57–1.00"),
    ("hetero-t top", "✓  .92 (7/8 ≥ .90)"),
    ("global-σ below ht", "✓ in-cell (real cells decisive)"),
    ("anchored ≈ ht", "✓  .85/.83 (flow8 rep.)"),
    ("σ-map / ledger / fit instruments", "✓ all"),
]
tbl = axT.table(cellText=[[a, b] for a, b in rows],
                colLabels=["robot fact", "toy (canonical cell)"],
                cellLoc="left", bbox=[0.0, 0.05, 1.0, 0.9])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
axT.set_title("T  component checklist", fontsize=10, loc="left")

fig.savefig("human_story_fig.png", dpi=140)
print("wrote human_story_fig.png")
