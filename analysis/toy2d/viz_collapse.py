"""Figure for the collapse-note toy (PART CCCXIV): task schematic, PR
pruning bars (note Fig-1 analog), survivor column-share contrast, kick-dose
curves with curated rescue, factor-isolation table. Reads toycollapse.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import toycollapse as T

COL = {"l2": "#d62728", "hg": "#ff9896", "ht": "#9467bd", "mip": "#1f77b4"}
NAME = {"l2": "MSE", "hg": "hetero-G", "ht": "hetero-t", "mip": "MIP"}
ARMS = ("l2", "hg", "ht", "mip")
R = json.load(open("toycollapse.json"))
SEEDS = (0, 1)


def m(cell, arm, key):
    return float(np.mean([R[f"{cell}_{arm}_s{s}"][key] for s in SEEDS]))


fig = plt.figure(figsize=(13.5, 10.5))
gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.9,
                      left=0.055, right=0.985, top=0.90, bottom=0.24)
axT = fig.add_axes([0.08, 0.03, 0.86, 0.13])
fig.suptitle("Collapse-note toy: ambiguous-survivor commitment → off-support misread → "
             "zero-tolerance break; cured by repricing / anchoring / curation",
             fontsize=12.5, y=0.965)

# --- A: task + expert demos ------------------------------------------------
axA = fig.add_subplot(gs[0, 0:3])
PHC = {"approach": "0.55", "settle": "tab:orange", "stroke": "tab:blue"}
rng = np.random.RandomState(5)
for _ in range(5):
    states, h = T.gen_episode(rng)
    for k in range(len(states) - 1):
        ph = T.phase_of(states[k][1], states[k][0])
        p0, p1 = states[k][0], states[k + 1][0]
        axA.plot([p0[0], p1[0]], [p0[1], p1[1]], color=PHC[ph], lw=1.2, alpha=0.75)
axA.add_patch(Rectangle((-T.XTOL, -0.09), 2 * T.XTOL, 0.16, fill=False,
                        ec="tab:orange", lw=1.6))
axA.scatter(*T.A, marker="*", s=130, color="k", zorder=6)
axA.scatter(*T.B, marker="*", s=130, color="tab:blue", zorder=6)
axA.annotate("grasp window: latch g 0→1 (6 steps < H=8),\nseat y↓0.03; "
             "$|x|>0.03$ while grasping = BREAK", (T.XTOL + 0.01, -0.05),
             fontsize=8.5, color="tab:orange")
axA.annotate("stroke to B rises back through\nthe settle/approach height band",
             (0.28, 0.16), fontsize=8.5, color="tab:blue")
axA.annotate("eval: pure y-kicks here", xy=(-0.005, 0.04), xytext=(-0.29, 0.30),
             fontsize=8.5, color="#c00000",
             arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.1))
hnd = [plt.Line2D([0], [0], color=PHC[p], lw=2) for p in PHC]
axA.legend(hnd, list(PHC), fontsize=8.5, loc="upper right")
axA.set_xlim(-0.30, 0.75)
axA.set_ylim(-0.12, 1.02)
axA.set_aspect("equal")
axA.set_title("A  task (clean, deterministic; expert demos phase-colored)",
              fontsize=10, loc="left")

# --- B: PR pruning bars (note Fig-1 analog) --------------------------------
axB = fig.add_subplot(gs[0, 3:6])
groups = [("l2", 5000), ("hg", 5000), ("l2", 40000), ("hg", 40000),
          ("ht", 40000), ("mip", 40000)]
xs = np.arange(len(groups))
w = 0.36
for off, (qk, lab, al) in enumerate((("prq", "quiet (grasp window)", 1.0),
                                     ("prl", "loud (stroke)", 0.45))):
    vals = [m("DISTR", a, f"{qk}_{st}") for a, st in groups]
    axB.bar(xs + (off - 0.5) * w, vals, w,
            color=[COL[a] for a, _ in groups], alpha=al, label=lab)
axB.set_xticks(xs, [f"{NAME[a]}\n@{st // 1000}k" for a, st in groups], fontsize=8)
axB.set_ylabel("local feature-Jacobian PR")
axB.legend(fontsize=8.5)
axB.set_title("B  chart pruning (DISTR): MSE deepest floor; repriced/anchored higher;\n"
              "NOT quiet-localized in the toy (quiet ≈ loud — differs from real)",
              fontsize=10, loc="left")

# --- C: survivor composition ----------------------------------------------
axC = fig.add_subplot(gs[1, 0:2])
bars = [("l2", "DISTR"), ("mip", "DISTR"), ("hg", "DISTR"), ("l2", "CUR")]
xs = np.arange(len(bars))
yd = [m(c, a, "cg_y") + m(c, a, "cg_d") for a, c in bars]
gg = [m(c, a, "cg_g") for a, c in bars]
axC.bar(xs - 0.19, yd, 0.36, color="#c00000", alpha=0.75,
        label="height family (y + d copies) — AMBIGUOUS")
axC.bar(xs + 0.19, gg, 0.36, color="#2ca02c", alpha=0.75,
        label="latch g — unambiguous")
axC.set_xticks(xs, [f"{NAME[a]}\n{c}" for a, c in bars], fontsize=8.5)
axC.set_ylabel("settle-window column-gain share")
axC.legend(fontsize=7.8, loc="upper right")
axC.set_title("C  the survivor's identity:\nMSE commits to the ambiguous family; "
              "MIP keeps the latch", fontsize=10, loc="left")

# --- D: kick-dose curves ---------------------------------------------------
axD = fig.add_subplot(gs[1, 2:4])
kk = [0.0, 0.02, 0.04]
for arm in ARMS:
    axD.plot(kk, [m("DISTR", arm, f"sr_k{k}_as8") for k in kk], "-o",
             color=COL[arm], label=NAME[arm], lw=1.9, ms=4.5)
axD.plot(kk, [m("CUR", "l2", f"sr_k{k}_as8") for k in kk], "--s",
         color=COL["l2"], lw=1.6, ms=4.5, alpha=0.65, label="MSE, curated obs")
axD.set_xticks(kk, ["0", "0.02", "0.04"])
axD.set_xlabel("y-kick dose")
axD.set_ylabel("SR (AS=8)")
axD.set_ylim(0, 1.05)
axD.legend(fontsize=7.6, loc="lower left")
axD.set_title("D  kick dose–response:\nMSE breaks; repricing/anchor/curation rescue",
              fontsize=10, loc="left")

# --- E: basin bars ---------------------------------------------------------
axE = fig.add_subplot(gs[1, 4:6])
xs = np.arange(len(ARMS))
axE.bar(xs, [m("DISTR", a, "basin") for a in ARMS], 0.55,
        color=[COL[a] for a in ARMS])
for x_, a in zip(xs, ARMS):
    axE.text(x_, m("DISTR", a, "basin") + 0.02, f"{m('DISTR', a, 'basin'):.2f}",
             ha="center", fontsize=8.5)
axE.set_xticks(xs, [NAME[a] for a in ARMS], fontsize=8.5)
axE.set_ylabel("basin snap α (settle→stroke)")
axE.set_ylim(0, 1.0)
axE.set_title("E  basin boundary — INVERTED vs real\n(MSE snaps latest here, "
              "not earliest)", fontsize=10, loc="left")

# --- T: factor-isolation table --------------------------------------------
axT.axis("off")
rows = [
    ("MSE full AS=8", f"{m('DISTR', 'l2', 'sr_k0.04_as8'):.2f}",
     f"{m('DISTR', 'l2', 'br_k0.04_as8'):.2f}", "the failure cell"),
    ("MSE curated", f"{m('CUR', 'l2', 'sr_k0.04_as8'):.2f}",
     f"{m('CUR', 'l2', 'br_k0.04_as8'):.2f}", "ambiguity removed → rescued (real: 100)"),
    ("MSE AS=1", f"{m('DISTR', 'l2', 'sr_k0.04_as1'):.2f}",
     f"{m('DISTR', 'l2', 'br_k0.04_as1'):.2f}",
     "INVERTED vs real: replanning re-queries the off-support state"),
    ("hetero-G", f"{m('DISTR', 'hg', 'sr_k0.04_as8'):.2f}",
     f"{m('DISTR', 'hg', 'br_k0.04_as8'):.2f}", "repricing rescues"),
    ("MIP", f"{m('DISTR', 'mip', 'sr_k0.04_as8'):.2f}",
     f"{m('DISTR', 'mip', 'br_k0.04_as8'):.2f}", "anchoring rescues"),
]
tbl = axT.table(cellText=[[r[1], r[2], r[3]] for r in rows],
                rowLabels=[r[0] for r in rows],
                colLabels=["SR @kick .04", "break rate", "reading"],
                cellLoc="center", bbox=[0.13, 0.0, 0.87, 1.0])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for r_ in range(1, len(rows) + 1):
    tbl[r_, 2].get_text().set_ha("left")
axT.set_title("T  factor isolation (kick .04): the note's Table-5 analog — "
              "curation and repricing/anchoring rescue; AS=1 inverts in the toy",
              fontsize=10, loc="left")

fig.savefig("collapse_fig.png", dpi=140)
print("wrote collapse_fig.png", flush=True)
