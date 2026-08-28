"""Failure-mode comparison on z-turn: traced rollouts for L2 / HG / HT
(standard v2.3b config) and flow8 (tuned EMA recipe), scrapes marked."""
import os

os.environ.setdefault("NOISE_TYPE", "zturn")
os.environ.setdefault("NEP", "160")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import toytube as T

YS = np.linspace(-0.02, 1.45, 1000)
CS = np.array([T.center(y) for y in YS])
HW = np.array([T.half_width(y) for y in YS])


def draw_world(ax):
    zb = (YS > T.Y_STR_LO) & (YS <= T.Y_TUBE_LO)
    lo = np.where(zb, -T.Z_BOX, CS - HW)
    hi = np.where(zb, T.X_OFF + T.Z_BOX, CS + HW)
    ax.fill_betweenx(YS, lo, hi, color="grey", alpha=0.22, zorder=1)
    ax.plot(lo, YS, color="dimgrey", lw=1.2, zorder=2)
    ax.plot(hi, YS, color="dimgrey", lw=1.2, zorder=2)
    ax.plot([0], [0], "k*", ms=12, zorder=6)
    ax.set_xlim(-0.16, 0.24)
    ax.set_ylim(-0.05, 1.5)
    ax.set_xlabel("x")


def rollout_trace(net, norm, rng):
    p = np.array([rng.uniform(-0.2, 0.2), rng.uniform(1.6, 1.7)])
    vel = np.zeros(2)
    tr = [p.copy()]
    for _ in range(T.ROLL_BUDGET // T.AS_DEF):
        with torch.no_grad():
            ob = np.concatenate([p, vel])[None, :]
            xn = torch.tensor(norm.nx(ob), device=T.DEV)
            yn = net(xn).cpu().numpy()[0]
        chunk = norm.uy(yn).reshape(T.H, T.ACT_D)[:T.AS_DEF]
        for a in chunk:
            p_new = T.wall(p[:2] + np.clip(a[:2], -T.CLIP, T.CLIP)
                           + T.MOM * vel + T.PROC * rng.randn(2))
            vel = p_new - p[:2] if not np.isnan(p_new[0]) else vel
            p = p_new
            tr.append(p.copy())
            if np.isnan(p[0]):
                return np.array(tr), "scrape"
            if np.linalg.norm(p - T.G) < T.DOCK_TOL:
                return np.array(tr), "success"
    return np.array(tr), "timeout"


drng = np.random.RandomState(1000)
X, Y, Yc, AL = T.build_dataset(drng)
norm = T.Norm(X, Y)

CMAP = {"success": "tab:green", "scrape": "tab:red", "timeout": "0.55"}
fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2))

ARMS = [("HG", "hg"), ("HT", "ht"), ("flow32", "flow8")]
for ax, (label, arm) in zip(axes, ARMS):
    if arm == "flow8":
        T.WIDTH, T.STEPS, T.EMA_R, T.FLOW_NS = 512, 150000, 0.0, 32
    else:
        T.WIDTH, T.STEPS, T.EMA_R = 256, 50000, 0.0
    print(f"training {arm}...", flush=True)
    net = T.train(arm, 0, norm.nx(X), norm.ny(Y))
    net.eval()
    draw_world(ax)
    rng = np.random.RandomState(7)
    n = {"success": 0, "scrape": 0, "timeout": 0}
    for k in range(30):
        tr, out = rollout_trace(net, norm, rng)
        ax.plot(tr[:, 0], tr[:, 1], "-", color=CMAP[out], lw=0.9, alpha=0.6,
                zorder=4)
        if out == "scrape":
            ax.plot(tr[-2, 0], tr[-2, 1], "x", color="tab:red", ms=7, mew=1.8,
                    zorder=6)
        n[out] += 1
    ax.set_title(f"{label}   SR {n['success']}/30  "
                 f"(scrape {n['scrape']}, timeout {n['timeout']})",
                 fontsize=11, loc="left")
axes[0].set_ylabel("height $y$")

fig.suptitle("z-turn rollouts (canonical env): HG / HT / flow32 (30 traced rollouts each; "
             "red = scrape with final contact marked, grey = timeout)",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("arms_zturn_hg_ht_flow.png", dpi=160)
print("wrote arms_zturn_hg_ht_flow.png", flush=True)
