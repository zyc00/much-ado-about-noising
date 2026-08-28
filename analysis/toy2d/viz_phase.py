"""Trajectory figure for the phase/distractor toy (v4 Cell B config:
GSAT=0.6 SEAT=0.05, DISTR obs). Panels: A task schematic + expert demos
(phase-colored), D latch-progress timelines, B closed-loop rollouts per
arm, C hold-region zooms. Caches seed-0 checkpoints in phase_ck_*.pt."""
import os

os.environ.setdefault("GSAT", "0.6")
os.environ.setdefault("SEAT", "0.05")
os.environ.setdefault("STEPS", "60000")
os.environ.setdefault("WIDTH", "128")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

import toyphase as P

COL = {"l2": "#d62728", "mip": "#1f77b4", "flow8": "#17becf"}
NAME = {"l2": "MSE", "mip": "MIP (2-step)", "flow8": "Flow (8-step)"}
ARMS = ("l2", "mip", "flow8")
PHC = {"approach": "0.55", "settle": "tab:orange", "transit": "tab:blue"}
CELL = "DISTR"

drng = np.random.RandomState(0)
X, Y, PH = P.build_dataset(CELL, drng)
norm = P.Norm(X, Y)
Xn, Yn = norm.nx(X), norm.ny(Y)
print(f"dataset {len(X)}", flush=True)

nets = {}
for arm in ARMS:
    ck = f"phase_ck_{arm}.pt"
    if os.path.exists(ck):
        nets[arm] = torch.load(ck, map_location=P.DEV, weights_only=False)
        print(f"{arm}: loaded cache", flush=True)
    else:
        nets[arm] = P.train(arm, 0, Xn, Yn, X.shape[1])
        torch.save(nets[arm], ck)
        print(f"{arm}: trained + cached", flush=True)


def rollout_rec(net, rng):
    h = 0.3 + 0.4 * rng.rand(P.NDIST) + rng.uniform(-0.05, 0.05, P.NDIST)
    p = np.array([rng.uniform(-0.1, 0.1), rng.uniform(0.7, 0.9)])
    c = 0.0
    path, cs = [p.copy()], [0.0]
    ok = False
    budget = 320
    while budget > 0 and not ok:
        state = np.concatenate([p, [c], h])
        chunk = P.policy_chunk(net, norm, state, CELL, rng)
        for a in chunk:
            p = p + np.clip(a, -P.CLIP, P.CLIP)
            c = P.step_c(p, c)
            path.append(p.copy())
            cs.append(c)
            budget -= 1
            if c >= 1.0 and np.linalg.norm(p - P.B) < 0.05:
                ok = True
                break
            if budget <= 0:
                break
    return np.array(path), np.array(cs), ok


rec = {arm: [rollout_rec(nets[arm], np.random.RandomState(500 + i))
             for i in range(14)] for arm in ARMS}
for arm in ARMS:
    print(f"{arm}: SR {np.mean([r[2] for r in rec[arm]]):.2f}", flush=True)

fig = plt.figure(figsize=(13.5, 12.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.0], hspace=0.32,
                      wspace=0.26, left=0.06, right=0.98, top=0.93, bottom=0.05)
fig.suptitle("Phase/distractor toy (GSAT=0.6, SEAT=0.05, distractor obs): "
             "MSE drifts out of the ambiguous wait; MIP/Flow hold and go",
             fontsize=12.5, y=0.975)

# --- A: schematic + expert demos ------------------------------------------
axA = fig.add_subplot(gs[0, 0:2])
grng = np.random.RandomState(11)
for _ in range(5):
    st, ac = P.gen_episode(grng)
    ph = [P.phase_of(s) for s in st]
    for k in range(len(st) - 1):
        axA.plot(st[k:k + 2, 0], st[k:k + 2, 1], color=PHC[ph[k]],
                 lw=1.2, alpha=0.7)
axA.add_patch(Circle(P.A, P.HOLD_R, fill=False, ec="tab:orange", lw=1.5))
axA.add_patch(Circle(P.A, 0.15, fill=False, ec="tab:orange", lw=1.0, ls=":"))
axA.scatter(*P.A, marker="*", s=140, color="k", zorder=6)
axA.scatter(*P.B, marker="*", s=140, color="tab:blue", zorder=6)
axA.text(-0.23, 0.92, "A: hold K=15 steps; latch c 0→1\n(obs latch saturates at c=0.6);\ntool SEATS: y creeps −0.05",
         fontsize=8.5, ha="left", va="top")
axA.annotate("B: go target\n(c≥1 required)", P.B + np.array([0.02, -0.02]),
             fontsize=8.5, color="tab:blue", va="top")
axA.text(-0.12, 0.55, "approach", fontsize=8.5, color="0.4", rotation=78)
axA.text(0.16, 0.28, "transit (rises back through\nall settle depths)",
         fontsize=8.5, color="tab:blue")
hnd = [plt.Line2D([0], [0], color=PHC[p], lw=2) for p in PHC]
axA.legend(hnd, list(PHC), fontsize=8.5, loc="upper right")
axA.set_xlim(-0.25, 0.72)
axA.set_ylim(-0.16, 0.95)
axA.set_aspect("equal")
axA.set_title("A  expert demos, phase-colored (5 episodes, exec-noise tube)",
              fontsize=10, loc="left")

# --- D: latch-progress timelines ------------------------------------------
axD = fig.add_subplot(gs[0, 2])
for arm in ARMS:
    for path, cs, ok in rec[arm]:
        axD.plot(np.arange(len(cs)), cs, color=COL[arm], alpha=0.35,
                 lw=1.0 if ok else 0.8, ls="-" if ok else "--")
axD.axhline(1.0, color="k", lw=0.8, ls=":")
axD.text(2, 1.02, "seated (go allowed)", fontsize=7.5)
axD.set_xlabel("env step")
axD.set_ylabel("latch progress c")
axD.set_xlim(0, 140)
axD.set_ylim(-0.03, 1.12)
hnd = [plt.Line2D([0], [0], color=COL[a], lw=2) for a in ARMS]
axD.legend(hnd, [NAME[a] for a in ARMS], fontsize=8, loc="lower right")
axD.set_title("D  latch c(t): dashed = failed episode", fontsize=10, loc="left")

# --- B: closed-loop rollouts per arm --------------------------------------
for i, arm in enumerate(ARMS):
    ax = fig.add_subplot(gs[1, i])
    nok = 0
    for path, cs, ok in rec[arm]:
        ax.plot(path[:, 0], path[:, 1], color=COL[arm], lw=1.0,
                alpha=0.75 if ok else 0.45, ls="-" if ok else "--")
        if not ok:
            ax.scatter(path[-1, 0], path[-1, 1], marker="x", s=40,
                       color=COL[arm], zorder=6)
        nok += ok
    ax.add_patch(Circle(P.A, P.HOLD_R, fill=False, ec="tab:orange", lw=1.2))
    ax.scatter(*P.A, marker="*", s=90, color="k", zorder=6)
    ax.scatter(*P.B, marker="*", s=90, color="tab:blue", zorder=6)
    ax.set_xlim(-0.3, 0.75)
    ax.set_ylim(-0.25, 0.95)
    ax.set_aspect("equal")
    t = f"B{i + 1}  {NAME[arm]} rollouts ({nok}/14)"
    if i == 0:
        t += "; × = failed end"
    ax.set_title(t, fontsize=9.5, loc="left")

# --- C: hold-region zooms --------------------------------------------------
for i, arm in enumerate(ARMS):
    ax = fig.add_subplot(gs[2, i])
    for path, cs, ok in rec[arm]:
        m = (cs > 0) & (cs < 1.0)
        idx = np.where(m[1:])[0] + 1  # steps where the latch is engaged
        if len(idx) == 0:
            continue
        lo, hi = idx[0], min(idx[-1] + 12, len(path) - 1)
        seg = path[lo:hi]
        ax.plot(seg[:, 0], seg[:, 1], color=COL[arm], lw=1.0,
                alpha=0.8 if ok else 0.5, ls="-" if ok else "--")
        if not ok:
            ax.scatter(seg[-1, 0], seg[-1, 1], marker="x", s=36,
                       color=COL[arm], zorder=6)
    ax.add_patch(Circle(P.A, P.HOLD_R, fill=False, ec="tab:orange", lw=1.5))
    ax.add_patch(Circle(P.A, 0.15, fill=False, ec="tab:orange", lw=1.0, ls=":"))
    ax.scatter(*P.A, marker="*", s=90, color="k", zorder=6)
    ax.annotate("seat depth", (0.005, -P.SEAT), fontsize=7.5, color="0.35",
                xytext=(0.06, -0.13),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9))
    ax.set_xlim(-0.2, 0.3)
    ax.set_ylim(-0.18, 0.2)
    ax.set_aspect("equal")
    t = f"C{i + 1}  {NAME[arm]} hold + go (zoom at A)"
    if i == 0:
        t += "; solid ring = hold radius"
    ax.set_title(t, fontsize=9.5, loc="left")

fig.savefig("phase_traj.png", dpi=140)
print("wrote phase_traj.png", flush=True)
