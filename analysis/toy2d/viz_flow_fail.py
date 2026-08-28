"""How flow fails on z-turn: zero-init deterministic integration blends the
route modes; integrating from a source draw commits. One net, three panels:
(A) rollouts with the toy's zero-init inference, (B) rollouts with randn-init,
(C) the output fan at a decision state (60 draws vs the single zero path)."""
import os

os.environ.setdefault("NOISE_TYPE", "zturn")
os.environ.setdefault("WIDTH", "512")
os.environ.setdefault("STEPS", "150000")
os.environ.setdefault("NEP", "160")
os.environ.setdefault("FLOW_NS", "32")
os.environ.setdefault("EMA", "0.999")

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
    ax.plot([0], [0], "k*", ms=13, zorder=6)
    ax.set_xlim(-0.16, 0.24)
    ax.set_ylim(-0.05, 1.5)
    ax.set_xlabel("x")


def integrate(net, xn, y0):
    y = y0
    for j in range(net.nsteps):
        t = torch.full((len(xn), 1), j / net.nsteps, device=xn.device)
        y = y + (1.0 / net.nsteps) * net.vel(xn, y, t)
    return y


def rollout_trace(net, norm, rng, rand_init):
    p = np.array([rng.uniform(-0.2, 0.2), rng.uniform(1.6, 1.7)])
    vel = np.zeros(2)
    tr = [p.copy()]
    for _ in range(T.ROLL_BUDGET // T.AS_DEF):
        with torch.no_grad():
            ob = np.concatenate([p, vel])[None, :]
            xn = torch.tensor(norm.nx(ob), device=T.DEV)
            y0 = (torch.randn(1, T.ACT_D * T.H, device=T.DEV) if rand_init
                  else torch.zeros(1, T.ACT_D * T.H, device=T.DEV))
            yn = integrate(net, xn, y0).cpu().numpy()[0]
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


print("training flow8 (zturn, EMA recipe)...", flush=True)
drng = np.random.RandomState(1000)
X, Y, Yc, AL = T.build_dataset(drng)
norm = T.Norm(X, Y)
net = T.train("flow8", 0, norm.nx(X), norm.ny(Y))
net.eval()
print("trained.", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
CMAP = {"success": "tab:green", "scrape": "tab:red", "timeout": "0.6"}

for ax, rand_init, title in [
        (axes[0], False, "A  zero-init inference (the toy default):\n"
                         "one deterministic path per state = route BLEND"),
        (axes[1], True, "B  randn-init inference (proper draw):\n"
                        "each rollout COMMITS to a route")]:
    draw_world(ax)
    rng = np.random.RandomState(7)
    n_s = 0
    for k in range(30):
        tr, out = rollout_trace(net, norm, rng, rand_init)
        ax.plot(tr[:, 0], tr[:, 1], "-", color=CMAP[out], lw=0.9,
                alpha=0.65, zorder=4)
        if out == "scrape":
            ax.plot(tr[-2, 0], tr[-2, 1], "x", color="tab:red", ms=7,
                    mew=1.8, zorder=6)
        n_s += out == "success"
    ax.set_title(title + f"   (SR {n_s}/30)", fontsize=10.5, loc="left")
axes[0].set_ylabel("height $y$")

# ---- C: output fan at a decision state -----------------------------------
ax = axes[2]
draw_world(ax)
# anchor: demo state just above the Z band, moving down
ii = np.argmin(np.abs(X[:, 1] - 1.02) + np.abs(X[:, 0]) * 3)
anchor = X[ii]
xn = torch.tensor(norm.nx(anchor[None, :]), device=T.DEV).repeat(60, 1)
with torch.no_grad():
    fan = integrate(net, xn, torch.randn(60, T.ACT_D * T.H, device=T.DEV))
    zero = integrate(net, xn[:1], torch.zeros(1, T.ACT_D * T.H, device=T.DEV))
for yn in fan.cpu().numpy():
    ch = norm.uy(yn).reshape(T.H, T.ACT_D)[:, :2]
    path = anchor[None, :2] + np.cumsum(np.clip(ch, -T.CLIP, T.CLIP), 0)
    ax.plot(np.r_[anchor[0], path[:, 0]], np.r_[anchor[1], path[:, 1]],
            "-", color="tab:blue", lw=0.8, alpha=0.45, zorder=4)
ch = norm.uy(zero.cpu().numpy()[0]).reshape(T.H, T.ACT_D)[:, :2]
path = anchor[None, :2] + np.cumsum(np.clip(ch, -T.CLIP, T.CLIP), 0)
ax.plot(np.r_[anchor[0], path[:, 0]], np.r_[anchor[1], path[:, 1]],
        "-", color="black", lw=2.5, zorder=6)
ax.plot(anchor[0], anchor[1], "ko", ms=7, zorder=7)
ax.set_ylim(0.35, 1.15)
ax.set_title("C  predicted chunks at one decision state:\n60 randn draws "
             "(blue) vs the single zero-init output (black)", fontsize=10.5,
             loc="left")

fig.suptitle("Why toy flow fails on z-turn: training transports NOISE to "
             "routes, but zero-init inference transports one point — the "
             "averaged path", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("flow_fail_fig.png", dpi=160)
print("wrote flow_fail_fig.png", flush=True)
