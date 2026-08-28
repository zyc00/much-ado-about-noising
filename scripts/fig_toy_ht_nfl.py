"""Design + results figures for the heavy-tail (HT-only-wins) witness,
matching the style of toy_mip_nfl_design/results."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "scripts")
from toy2d_mip_nfl import (ACTION_MM, B, DOCK_TOL_MM, FORWARD_MM, K_SERVO,
                           X_GOAL_MM, half_width_mm, make_dataset, predict,
                           train, train_nll)

C = {"regression": "tab:blue", "mip_step1": "#8c6bb1", "mip_full": "tab:red",
     "hg": "tab:orange", "ht": "tab:green"}
LBL = {"regression": "Regression", "mip_step1": "MIP step 1",
       "mip_full": "MIP full", "hg": "HG", "ht": "HT"}

# ---------------- design figure ----------------
fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)

ax = fig.add_subplot(gs[0, :2])
xs = np.linspace(0, 165, 200)
ax.fill_between(xs, -half_width_mm(xs), half_width_mm(xs), color="#dbe4ee",
                alpha=0.8)
ax.plot(xs, half_width_mm(xs), color="dimgray", lw=1.5)
ax.plot(xs, -half_width_mm(xs), color="dimgray", lw=1.5)
rng = np.random.RandomState(3)
for y0 in np.linspace(-9, 9, 7):
    y = y0
    tr = [(0.0, y)]
    for _ in range(40):
        y = y + (-K_SERVO * y)
        tr.append((tr[-1][0] + FORWARD_MM, y))
        if tr[-1][0] >= 160:
            break
    tr = np.array(tr)
    ax.plot(tr[:, 0], tr[:, 1], color="tab:blue", lw=1.1, alpha=0.6)
y = -7.0
xpos = 0.0
for k in range(24):
    a_clean = -K_SERVO * y
    spike = rng.standard_t(1.05) * B
    spike = float(np.clip(spike, -40 * B, 40 * B))
    ax.annotate("", xy=(xpos + 4, y + a_clean + spike * ACTION_MM * 0.6),
                xytext=(xpos, y),
                arrowprops=dict(arrowstyle="->", color="tab:orange",
                                lw=1.1, alpha=0.8))
    if abs(spike) > 10 * B:
        ax.annotate(f"recorded spike {spike/B:+.0f}$\\times$ scale\n"
                    "(never executed)",
                    xy=(xpos + 4, y + a_clean + spike * ACTION_MM * 0.6),
                    xytext=(xpos + 22, y + np.sign(spike) * 6.5),
                    fontsize=9, color="tab:orange",
                    arrowprops=dict(arrowstyle="->", lw=0.9,
                                    color="tab:orange"))
    y = y + a_clean
    xpos += FORWARD_MM
    if xpos > 150:
        break
ax.axvline(X_GOAL_MM, color="k", lw=1)
ax.add_patch(plt.Rectangle((X_GOAL_MM - 0.8, -DOCK_TOL_MM), 1.6,
                           2 * DOCK_TOL_MM, color="tab:green", alpha=0.7))
ax.text(X_GOAL_MM + 2, -0.4, "tight dock\n|y| < 0.5 mm", fontsize=10,
        color="tab:green")
ax.set_xlim(-3, 178)
ax.set_ylim(-15, 15)
ax.set_xlabel("forward position x (mm)")
ax.set_ylabel("lateral position y (mm)")
ax.set_title("A   Same funnel-to-dock task; demonstrator EXECUTES the clean "
             "servo (oracle SR = 1)\nbut the RECORDED lateral labels carry "
             "symmetric heavy-tailed corruption (Student-t, df = 1.05)",
             fontsize=12, loc="left")

ax = fig.add_subplot(gs[0, 2])
s = np.clip(np.random.RandomState(0).standard_t(1.05, 200000) * B,
            -40 * B, 40 * B)
ax.hist(s, bins=160, density=True, color="tab:orange", alpha=0.75)
ax.set_yscale("log")
ax.axvline(0, color="k", lw=1.4)
ax.text(0.3, 2.0, "mean = median = location = 0\n(no bias to exploit —\n"
        "the attack is variance)", fontsize=9)
ax.set_xlabel("recorded lateral residual (action units)")
ax.set_title("B   Label distribution at every state\nsymmetric, zero-mean, "
             "heavy tails", fontsize=12, loc="left")

ax = fig.add_subplot(gs[1, 0])
r = np.linspace(-8, 8, 400)
ax.plot(r, r, color="tab:blue", lw=2, label="L2 / HG:  $\\psi(r) = r$ "
        "(unbounded)")
nu = 2.0
ax.plot(r, (nu + 1) * r / (nu + r ** 2), color="tab:green", lw=2,
        label="HT ($\\nu$=2):  $\\psi(r) = \\frac{(\\nu+1)r}{\\nu+r^2}$")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlabel("residual r (scales)")
ax.set_ylabel("influence on the location")
ax.set_title("C   The mechanism: influence functions\na single spike drags "
             "the mean; the t-location ignores it", fontsize=12, loc="left")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
rngE = np.random.RandomState(1)
means, tlocs = [], []
for _ in range(2000):
    x = np.clip(rngE.standard_t(1.05, 30) * B, -40 * B, 40 * B)
    means.append(x.mean())
    mu = 0.0
    for _ in range(60):
        w = (nu + 1) / (nu + ((x - mu) / B) ** 2)
        mu = (w * x).sum() / w.sum()
    tlocs.append(mu)
ax.hist(np.array(means) / B, bins=80, range=(-3, 3), alpha=0.65,
        color="tab:blue", label=f"empirical mean (std {np.std(means)/B:.2f})")
ax.hist(np.array(tlocs) / B, bins=80, range=(-3, 3), alpha=0.65,
        color="tab:green",
        label=f"t-location (std {np.std(tlocs)/B:.2f})")
ax.set_xlabel("location estimate at n=30 samples (scales)")
ax.legend(fontsize=9)
ax.set_title("D   Finite-sample prediction (pure statistics)\nmean scatters; "
             "robust location concentrates", fontsize=12, loc="left")

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
rows = [
    ("L2 / HG", "empirical conditional mean:\ndragged by recorded spikes",
     "tab:blue"),
    ("MIP full", "step 1 = the same mean fit;\nstep-2 keeps heavy-tail-"
     "consistent\ndeviations instead of shrinking", "tab:red"),
    ("HT", "bounded-influence location:\nmatches the noise hypothesis\n"
     "$\\Rightarrow$ efficient, docks", "tab:green"),
]
for i, (name, txt, col) in enumerate(rows):
    yy = 0.93 - i * 0.33
    ax.add_patch(plt.Rectangle((0.02, yy - 0.26), 0.96, 0.30, fill=False,
                               edgecolor=col, lw=2,
                               transform=ax.transAxes))
    ax.text(0.06, yy - 0.03, name, fontsize=12, weight="bold", color=col,
            transform=ax.transAxes, va="top")
    ax.text(0.06, yy - 0.09, txt, fontsize=9.5, transform=ax.transAxes,
            va="top")
ax.set_title("E   The inductive-bias collision", fontsize=12, loc="left")

fig.suptitle("A heavy-tail no-free-lunch witness: recorded-only symmetric "
             "corruption + small data — only the estimator whose noise "
             "hypothesis matches the data docks",
             fontsize=15, y=0.99)
fig.savefig("analysis/paper/toy_ht_nfl_design.png", dpi=140,
            bbox_inches="tight")
print("design saved")

# ---------------- results figure ----------------
d_small = json.load(open("analysis/toy2d_mip_nfl_heavy.json"))
d_big = json.load(open("analysis/toy2d_mip_nfl_heavy30k.json"))
METH = ["regression", "mip_step1", "mip_full", "hg", "ht"]


def collect(d):
    agg = {m: [] for m in METH}
    for c in d["cells"]:
        for m in METH:
            v = c["methods"][m]
            agg[m].append((v["success_rate"], v["abs_y_goal_p50"]))
    return {m: np.array(v) for m, v in agg.items()}


small, big = collect(d_small), collect(d_big)

fig2 = plt.figure(figsize=(19, 10))
gs = fig2.add_gridspec(2, 2, hspace=0.4, wspace=0.25)

ax = fig2.add_subplot(gs[0, 0])
w = 0.38
xi = np.arange(len(METH))
for off, data, lab, al in ((-w / 2, small, "3k chunks (main)", 0.9),
                           (w / 2, big, "30k chunks (control)", 0.45)):
    ax.bar(xi + off, [data[m][:, 0].mean() for m in METH], w,
           color=[C[m] for m in METH], alpha=al,
           edgecolor="k" if al > 0.5 else "none",
           label=lab)
    for i, m in enumerate(METH):
        for v in data[m][:, 0]:
            ax.plot(xi[i] + off + np.random.uniform(-0.07, 0.07), v, "o",
                    color="k", ms=3.5, alpha=0.65)
ax.set_xticks(xi)
ax.set_xticklabels([LBL[m] for m in METH], fontsize=10)
ax.set_ylabel("dock success (|y| < 0.5 mm)")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.set_title("A   Only HT docks in the small-data regime; everyone passes "
             "at 10$\\times$ data\n(8 seeds main, 3 seeds control; oracle "
             "expert = 1.00 by construction)", fontsize=12, loc="left")

ax = fig2.add_subplot(gs[0, 1])
ax.bar(xi, [small[m][:, 1].mean() for m in METH], 0.55,
       color=[C[m] for m in METH], alpha=0.9)
for i, m in enumerate(METH):
    for v in small[m][:, 1]:
        ax.plot(xi[i] + np.random.uniform(-0.09, 0.09), v, "o", color="k",
                ms=3.5, alpha=0.65)
ax.axhline(DOCK_TOL_MM, color="tab:green", ls="--", lw=1.6)
ax.text(3.6, DOCK_TOL_MM + 0.02, "dock tolerance", fontsize=9,
        color="tab:green")
ax.set_xticks(xi)
ax.set_xticklabels([LBL[m] for m in METH], fontsize=10)
ax.set_ylabel("median |y| at dock (mm)")
ax.set_title("B   Dock error, 3k-chunk cell\nHT 0.24 mm vs 0.55$-$0.83 mm "
             "for every mean/mode method", fontsize=12, loc="left")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state, action = make_dataset("heavy", 300, 10, 8, 20260805)
nets = {}
nets["regression"], _ = train("regression", state, action, 0, 128, 6000,
                              512, 5e-4, device)
mip, _ = train("mip", state, action, 0, 128, 6000, 512, 5e-4, device)
nets["mip_full"] = mip
nets["hg"], _ = train_nll("hg", state, action, 0, 128, 6000, 512, 5e-4,
                          device)
nets["ht"], _ = train_nll("ht", state, action, 0, 128, 6000, 512, 5e-4,
                          device)

ax = fig2.add_subplot(gs[1, 0])
xx = np.repeat(np.linspace(0.0, 150.0, 41), 41)
yy = np.tile(np.linspace(-6.0, 6.0, 41), 41)
grid = np.stack([xx, yy], axis=1).astype(np.float32)
clean = (-K_SERVO * yy) / ACTION_MM
for m in ["regression", "mip_full", "ht"]:
    samp = {"regression": "regression", "mip_full": "mip_full",
            "hg": "hg", "ht": "ht"}[m]
    out = predict(nets[m], grid, samp, device)
    res = np.abs(out[:, 1] - clean) * ACTION_MM
    ax.hist(res, bins=np.logspace(-2.3, 0.9, 60), histtype="step", lw=2,
            color=C[m], label=f"{LBL[m]} (p90 {np.percentile(res,90):.2f} mm)")
ax.set_xscale("log")
ax.axvline(DOCK_TOL_MM, color="tab:green", ls="--", lw=1.4)
ax.set_xlabel("|fitted lateral action $-$ clean servo| (mm), state grid")
ax.set_ylabel("states")
ax.legend(fontsize=9)
ax.set_title("C   The fitted field (seed 0): spike-bent for mean/mode "
             "methods,\ntight for HT", fontsize=12, loc="left")

ax = fig2.add_subplot(gs[1, 1])
xs = np.linspace(0, 165, 200)
ax.fill_between(xs, -half_width_mm(xs), half_width_mm(xs), color="#dbe4ee",
                alpha=0.8)
for m in ["regression", "mip_full", "ht"]:
    samp = "mip_full" if m == "mip_full" else m
    x = np.zeros(9)
    y = np.linspace(-8, 8, 9)
    for _ in range(70):
        st = np.stack([x, y], axis=1).astype(np.float32)
        a = predict(nets[m], st, samp, device)
        x = x + np.clip(a[:, 0] * ACTION_MM, -2, 8)
        y = y + np.clip(a[:, 1] * ACTION_MM, -12, 12)
        keep = x < X_GOAL_MM
        if not keep.any():
            break
    ax.plot([X_GOAL_MM] * 9, y, "o", color=C[m], ms=6, alpha=0.85,
            label=f"{LBL[m]} dock positions")
ax.axhline(DOCK_TOL_MM, color="tab:green", ls="--", lw=1.2)
ax.axhline(-DOCK_TOL_MM, color="tab:green", ls="--", lw=1.2)
ax.set_xlim(150, 168)
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y at dock (mm)")
ax.legend(fontsize=9, loc="upper left")
ax.set_title("D   Where the 9 eval starts dock (seed 0, zoom at the gate)",
             fontsize=12, loc="left")

fig2.suptitle("Heavy-tail witness, trained models: oracle 1.00; HT 7/8 "
              "seeds (0.24 mm); Regression 0/8 (0.83 mm); MIP full 1/8 "
              "(0.67 mm); HG 2/8 (0.55 mm)", fontsize=15, y=0.99)
fig2.savefig("analysis/paper/toy_ht_nfl_results.png", dpi=140,
             bbox_inches="tight")
print("results saved")
