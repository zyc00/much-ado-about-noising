"""Why is clean-session l2 low at v2.1? Trajectory + death-site visualization."""
import os

for k, v in [("HW_DOCK", "0.004"), ("WIDTH", "256"), ("STEPS", "50000"),
             ("NEP", "160"), ("AS", "1"), ("NOISE_TYPE", "none")]:
    os.environ[k] = v

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import toytube as T

YS = np.linspace(-0.02, 1.4, 1000)
CZ = np.array([T.x_target(y, T.T1_FIX, T.T2_FIX) for y in YS])
HW = np.array([T.half_width(y) for y in YS])


def draw_world(ax):
    zb = (YS > T.Y_STR_LO) & (YS <= T.Y_TUBE_LO)
    lo = np.where(zb, -T.Z_BOX, CZ - HW)
    hi = np.where(zb, T.X_OFF + T.Z_BOX, CZ + HW)
    ax.fill_betweenx(YS, lo, hi, color="grey", alpha=0.2, zorder=1)
    ax.plot(lo, YS, color="dimgrey", lw=1.2, zorder=2)
    ax.plot(hi, YS, color="dimgrey", lw=1.2, zorder=2)
    ax.plot(CZ, YS, "k--", lw=0.6, alpha=0.5, zorder=2)
    ax.plot([0], [0], "k*", ms=12, zorder=6)
    ax.set_ylim(-0.03, 1.45)


def rollouts(net, norm, seed, n=25):
    rng = np.random.RandomState(9000 + seed)
    trajs = []
    for _ in range(n):
        p = np.array([rng.uniform(-0.2, 0.2), rng.uniform(1.6, 1.7)])
        vel = np.zeros(2)
        pts, out = [p.copy()], None
        for _ in range(T.ROLL_BUDGET):
            with torch.no_grad():
                ob = np.concatenate([p, vel])[None, :]
                yn = net(torch.tensor(norm.nx(ob), device=T.DEV)).cpu().numpy()[0]
            a = norm.uy(yn).reshape(T.H, 2)[0]
            pn = T.wall(p[:2] + np.clip(a, -T.CLIP, T.CLIP) + T.MOM * vel
                        + T.PROC * rng.randn(2))
            vel = pn - p[:2]
            p = pn
            pts.append(p.copy())
            if np.isnan(p[0]):
                out = "scrape"
                break
            if np.linalg.norm(p - T.G) < T.DOCK_TOL:
                out = "success"
                break
        trajs.append((np.array(pts), out or "timeout"))
    return trajs


fig, axes = plt.subplots(1, 4, figsize=(17, 7))
alldead = {}
for col, seed in enumerate((0, 1)):
    drng = np.random.RandomState(1000 + seed)
    X, Y, Yc, AL = T.build_dataset(drng)
    norm = T.Norm(X, Y)
    net = T.train("l2", seed, norm.nx(X), norm.ny(Y))
    trajs = rollouts(net, norm, seed)
    ax = axes[col]
    draw_world(ax)
    dead = []
    for pts, out in trajs:
        c = {"success": "tab:green", "scrape": "tab:red",
             "timeout": "tab:orange"}[out]
        ax.plot(pts[:-1, 0], pts[:-1, 1], "-", color=c, lw=0.7, alpha=0.6)
        if out == "scrape":
            ax.plot(pts[-2, 0], pts[-2, 1], "x", color="k", ms=6, zorder=7)
            dead.append(pts[-2, 1])
    alldead[seed] = dead
    sr = np.mean([o == "success" for _, o in trajs])
    ax.set_title(f"l2 clean seed {seed}: rollouts (SR {sr:.2f})\n"
                 f"green=success red=scrape x=death", fontsize=9)
    ax.set_xlim(-0.1, 0.15)

ax = axes[2]
for seed, c in ((0, "tab:blue"), (1, "tab:red")):
    if alldead[seed]:
        ax.hist(alldead[seed], bins=np.arange(0.05, 0.3, 0.01), alpha=0.6,
                color=c, label=f"seed {seed} (n={len(alldead[seed])})")
ax.axvline(T.Y_FUN_LO, color="k", ls=":", lw=0.8)
ax.set_xlabel("y at scrape")
ax.set_title("death sites (insert band 0.05-0.25;\ndogleg apex ~0.15)", fontsize=9)
ax.legend(fontsize=8)

# lateral fit error along demo states, insert band, both seeds
ax = axes[3]
for seed, c in ((0, "tab:blue"), (1, "tab:red")):
    drng = np.random.RandomState(1000 + seed)
    X, Y, Yc, AL = T.build_dataset(drng)
    norm = T.Norm(X, Y)
    net = T.train("l2", seed, norm.nx(X), norm.ny(Y))
    with torch.no_grad():
        pred = norm.uy(net(torch.tensor(norm.nx(X), device=T.DEV)).cpu().numpy())
    e0 = np.abs(pred[:, 0] - Y[:, 0]) * 1000.0
    ys = X[:, 1]
    bins = np.arange(0.05, 0.55, 0.025)
    mids, med = [], []
    for lo in bins:
        m = (ys >= lo) & (ys < lo + 0.025)
        if m.sum() > 10:
            mids.append(lo + 0.0125)
            med.append(np.median(e0[m]))
    ax.plot(mids, med, "-o", ms=3, color=c, label=f"seed {seed}")
hwmm = [1000 * T.half_width(y) for y in np.arange(0.06, 0.25, 0.01)]
ax.plot(np.arange(0.06, 0.25, 0.01), hwmm, "k--", lw=0.8, label="wall half-width (mm)")
ax.set_xlabel("y")
ax.set_ylabel("median |a0 error| (mm)")
ax.set_title("lateral action fit error vs walls\n(insert+funnel)", fontsize=9)
ax.legend(fontsize=8)

fig.suptitle("v2.1 clean-session l2 diagnosis: good seed (0) vs bad seed (1)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("v21_none_diag.png", dpi=150)
print("wrote v21_none_diag.png")
