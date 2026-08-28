"""Figure for the support-geometry toy: task schematic, delicate-vs-robust
support faces (actual NEP=5 band samples), scarcity curves (SR / F-share /
band error vs NEP), and metric tables. Reads toysupport.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import toysupport as T

COL = {"l2": "#d62728", "mip": "#1f77b4", "flow8": "#17becf", "l2nf": "#8c564b", "hg": "#ff9896", "ht": "#9467bd"}
NAME = {"l2": "MSE", "mip": "MIP (2-step)", "flow8": "Flow (8-step)",
        "l2nf": "MSE, force masked", "hg": "hetero-G", "ht": "hetero-t"}
ARMS = ("l2", "hg", "ht", "mip", "flow8", "l2nf")
R = json.load(open("toysupport.json"))
NEPS = sorted({int(k.split("_")[0][3:]) for k in R})
SEEDS = sorted({int(k.split("_s")[1]) for k in R})


def mean(nep, arm, key):
    return float(np.mean([R[f"nep{nep}_{arm}_s{s}"][key] for s in SEEDS]))


def vals(nep, arm, key):
    return [R[f"nep{nep}_{arm}_s{s}"][key] for s in SEEDS]


fig = plt.figure(figsize=(13.5, 12.0))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.85], hspace=0.38,
                      wspace=0.30, left=0.06, right=0.98, top=0.92, bottom=0.04)
fig.suptitle("Support-geometry toy (clean data, no noise): dominant modality M1=(p,h,k) "
             "vs sparse modality M2=force; scarcity dial NEP",
             fontsize=12.5, y=0.97)

# --- A1: expert episode timeline -------------------------------------------
axA = fig.add_subplot(gs[0, 0])
rng = np.random.RandomState(3)
st, ac = T.gen_episode(rng)
p, h, k, f = st[:, 0], st[0, 1], st[0, 2], st[:, 3]
tt = np.arange(len(st))
axA.plot(tt, p - h, color="0.3", lw=1.8, label="height above surface  $p-h$")
axA.axhline(0, color="0.6", lw=1.0)
axA.fill_between(tt, -0.03, 0, color="tab:brown", alpha=0.15)
axA2 = axA.twinx()
axA2.plot(tt, f, color="tab:green", lw=1.8, label="force $f=k\\,(h-p)^+$")
axA2.axhspan(T.FSTAR - T.FTOL, T.FSTAR + T.FTOL, color="tab:green", alpha=0.15)
axA2.axhline(T.FBREAK, color="#c00000", lw=1.5, ls="--")
axA2.text(1, T.FBREAK + 0.003, "break $2f^*$", fontsize=8, color="#c00000")
axA2.text(len(tt) * 0.5, T.FSTAR + T.FTOL + 0.004, "hold band $f^*\\pm$tol",
          fontsize=8, color="tab:green")
axA2.set_ylim(0, 0.14)
axA2.set_ylabel("force", fontsize=9)
axA.set_ylim(-0.05, 1.1)
axA.set_xlabel("step")
axA.set_ylabel("height above surface")
axA.legend(fontsize=7.5, loc="upper right")
axA.set_title(f"A1  expert episode (k={k:.1f}): descend, servo to $f^*$, hold",
              fontsize=9.5, loc="left")

# --- A2/A3: delicate vs robust support face (NEP=5 band samples) ----------
drng = np.random.RandomState(1005)
X5, _ = T.build_dataset(5, drng)
band = X5[X5[:, 3] > 0]
d5 = band[:, 1] - band[:, 0]
axB = fig.add_subplot(gs[0, 1])
axB.scatter(band[:, 2], d5, s=14, color="#c00000", alpha=0.7, zorder=5)
kk = np.linspace(1, 10, 100)
axB.plot(kk, T.FSTAR / kk, color="0.4", lw=1.4, ls="--",
         label="required depth $d^*=f^*/k$")
axB.fill_between(kk, (T.FSTAR - T.FTOL) / kk, (T.FSTAR + T.FTOL) / kk,
                 color="tab:green", alpha=0.15)
axB.fill_between(kk, T.FBREAK / kk, 0.13, color="#c00000", alpha=0.08)
axB.text(6.2, 0.055, "break region", fontsize=8, color="#c00000")
axB.set_xlabel("stiffness k (M1 coordinate)")
axB.set_ylabel("penetration depth $d$ (M1)")
axB.set_ylim(0, 0.13)
axB.legend(fontsize=8, loc="upper right")
axB.set_title("A2  M1 chart: NEP=5 in-band support =\nthin per-episode slices "
              "of the $1/k$ surface", fontsize=9.5, loc="left")

axC = fig.add_subplot(gs[0, 2])
axC.hist(band[:, 3], bins=24, color="tab:green", alpha=0.75)
axC.axvline(T.FSTAR, color="0.2", lw=1.5)
axC.axvspan(T.FSTAR - T.FTOL, T.FSTAR + T.FTOL, color="tab:green", alpha=0.18)
axC.axvline(T.FBREAK, color="#c00000", lw=1.5, ls="--")
axC.set_xlabel("force f (M2 coordinate)")
axC.set_ylabel("in-band sample count")
axC.set_title("A3  M2 chart: same NEP=5 samples =\ndense 1-D interval; "
              "target = one point $f^*$", fontsize=9.5, loc="left")

# --- Row 2: scarcity curves ------------------------------------------------
specs = [("sr1", "B  closed-loop SR (AS=1) vs data amount", "success rate", None),
         ("fshare", "C  in-band Jacobian F-share vs data amount", "F-share", None),
         ("banderr", "D  band action error (off-diagonal grid)", "mean |a - a*|", "log")]
for j, (key, title, ylab, yscale) in enumerate(specs):
    ax = fig.add_subplot(gs[1, j])
    arms_j = tuple(a for a in ARMS if not (key == "fshare" and a == "l2nf"))
    for arm in arms_j:
        ms = [mean(n, arm, key) for n in NEPS]
        ax.plot(NEPS, ms, "-o", color=COL[arm], label=NAME[arm], ms=4.5, lw=1.8)
        for n in NEPS:
            ax.scatter([n] * len(SEEDS), vals(n, arm, key), s=10,
                       color=COL[arm], alpha=0.45)
    if key == "fshare":
        ax.set_ylim(-0.005, 0.06)
        ax.text(0.5, 0.75, "NO arm reads force: F-share ≈ 0\nat every data amount —\n"
                "the scarcity advantage (panel B)\nis NOT modal re-weighting",
                transform=ax.transAxes, fontsize=9, ha="center", color="0.25")
    ax.set_xscale("log")
    ax.set_xticks(NEPS, [str(n) for n in NEPS])
    if yscale:
        ax.set_yscale(yscale)
    if key == "sr1":
        ax.axhline(1.0, color="0.5", lw=1.0, ls=":")
        ax.text(NEPS[0], 1.015, "expert", fontsize=7.5, color="0.4")
        ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("demonstrations (NEP)")
    ax.set_ylabel(ylab)
    if j == 0:
        ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(title, fontsize=9.5, loc="left")

# --- Row 3: tables ---------------------------------------------------------
tspecs = [("sr1", "SR (AS=1)"), ("br1", "break rate"), ("srab", "SR, force ablated")]
for j, (key, title) in enumerate(tspecs):
    ax = fig.add_subplot(gs[2, j])
    ax.axis("off")
    cells = [[f"{mean(n, a, key):.2f}" for n in NEPS] for a in ARMS]
    tbl = ax.table(cellText=cells, rowLabels=[NAME[a] for a in ARMS],
                   colLabels=[f"NEP={n}" for n in NEPS], cellLoc="center",
                   bbox=[0.30, 0.05, 0.68, 0.85])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for i, a in enumerate(ARMS):
        tbl[i + 1, -1].get_text().set_color(COL[a])
    ax.set_title(f"T{j + 1}  {title}", fontsize=9.5, loc="left")

fig.savefig("support_fig.png", dpi=140)
print("wrote support_fig.png", flush=True)
