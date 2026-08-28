"""zturn setting under v2.2: demos, l2 vs ht rollouts, mean-path anatomy."""
import os

for k, v in [("HW_DOCK", "0.004"), ("WIDTH", "256"), ("STEPS", "50000"),
             ("NEP", "160"), ("AS", "1"), ("NOISE_TYPE", "zturn")]:
    os.environ[k] = v

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import toytube as T

YS = np.linspace(-0.02, 1.4, 1000)
CS = np.array([T.center(y) for y in YS])
HW = np.array([T.half_width(y) for y in YS])


def draw_world(ax):
    zb = (YS > T.Y_STR_LO) & (YS <= T.Y_TUBE_LO)
    lo = np.where(zb, -T.Z_BOX, CS - HW)
    hi = np.where(zb, T.X_OFF + T.Z_BOX, CS + HW)
    ax.fill_betweenx(YS, lo, hi, color="grey", alpha=0.2, zorder=1)
    ax.plot(lo, YS, color="dimgrey", lw=1.2, zorder=2)
    ax.plot(hi, YS, color="dimgrey", lw=1.2, zorder=2)
    ax.plot(CS, YS, "k--", lw=0.6, alpha=0.5, zorder=2)
    ax.plot([0], [0], "k*", ms=12, zorder=6)
    ax.set_ylim(-0.03, 1.45)
    ax.set_xlim(-0.1, 0.15)


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


seed = 0
drng = np.random.RandomState(1000 + seed)
X, Y, Yc, AL = T.build_dataset(drng)
norm = T.Norm(X, Y)

fig, axes = plt.subplots(1, 4, figsize=(17, 7))

ax = axes[0]
draw_world(ax)
rng = np.random.RandomState(5)
for _ in range(12):
    st, _ = T.gen_episode(rng)
    s = np.array(st)
    ax.plot(s[:, 0], s[:, 1], "-", lw=0.8, alpha=0.7)
ax.plot([0, 0], [T.Y_STR_LO, T.Y_TUBE_LO], "k--", lw=1.2, alpha=0.8)
ax.set_title("zturn demos: detour at random heights\n(dashed = straight clean reference)",
             fontsize=9)

for col, arm in ((1, "l2"), (2, "ht")):
    net = T.train(arm, seed, norm.nx(X), norm.ny(Y))
    trajs = rollouts(net, norm, seed)
    ax = axes[col]
    draw_world(ax)
    dead = []
    for pts, out in trajs:
        c = {"success": "tab:green", "scrape": "tab:red",
             "timeout": "tab:orange"}[out]
        ax.plot(pts[:-1, 0], pts[:-1, 1], "-", color=c, lw=0.7, alpha=0.6)
        if out != "success":
            ax.plot(pts[-2, 0], pts[-2, 1], "x", color="k", ms=6, zorder=7)
            dead.append(pts[-2, 1])
    sr = np.mean([o == "success" for _, o in trajs])
    ax.set_title(f"{arm} rollouts (seed 0, SR {sr:.2f})", fontsize=9)

# mean lateral position of demo states by y (the label-average path) + policies
ax = axes[3]
ys = X[:, 1]
bins = np.arange(0.5, 0.92, 0.02)
mids, mx = [], []
for lo in bins:
    m = (ys >= lo) & (ys < lo + 0.02)
    if m.sum() > 5:
        mids.append(lo + 0.01)
        mx.append(X[m, 0].mean())
ax.plot(mx, mids, "k-o", ms=3, label="demo state mean E[x|y]")
ax.axvline(0, color="grey", ls="--", lw=0.8, label="straight reference")
ax.axvline(T.X_OFF, color="grey", ls=":", lw=0.8, label="detour column")
ax.set_ylim(0.48, 0.92)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Z band: mean demo occupancy\n(the path l2's average implies)", fontsize=9)
ax.legend(fontsize=7)

fig.suptitle("v2.2 zturn anatomy: detour-as-uncertainty vs straight clean law (seed 0)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("v22_zturn_diag.png", dpi=150)
print("wrote v22_zturn_diag.png")
