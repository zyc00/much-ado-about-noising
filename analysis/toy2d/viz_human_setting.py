"""Setting visualization for the story-proof toy (canonical round-3 cell).
A: geometry + clean expert paths. B: actual training demos (tremor visible).
C: the labels the regressor sees (a_x vs y scatter). D: closed-loop rollouts
of trained MSE vs hetero-t (seed 0, cached)."""
import os

os.environ.setdefault("S_HI", "0.08")
os.environ.setdefault("NEP", "40")
os.environ.setdefault("DOCK_TOL", "0.008")
os.environ.setdefault("STEPS", "25000")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import toyhuman as T

fig = plt.figure(figsize=(13.5, 10.5))
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22,
                      left=0.06, right=0.98, top=0.90, bottom=0.05)
fig.suptitle("Story-proof toy, the setting: funnel corridor with heavy-tailed align tremor "
             "and a fine dogleg into a tight dock", fontsize=12.5, y=0.96)


def draw_geometry(ax):
    yy = np.linspace(T.Y_DOCK, T.Y_TOP, 200)
    cc = np.array([T.center(y) for y in yy])
    hw = np.array([T.half_width(y) for y in yy])
    ax.fill_betweenx(yy, cc - hw, cc + hw, color="0.92", zorder=0)
    ax.plot(cc - hw, yy, color="0.45", lw=1.4)
    ax.plot(cc + hw, yy, color="0.45", lw=1.4)
    ax.plot(cc, yy, "k--", lw=0.9, alpha=0.7)
    ax.axhspan(T.Y_ALIGN_LO, T.Y_TOP, color="#d62728", alpha=0.10)
    ax.axhspan(T.Y_DOCK, T.Y_ALIGN_LO, color="#2ca02c", alpha=0.08)
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(T.DOCK_TOL * np.cos(th), T.DOCK_TOL * np.sin(th), color="k", lw=1.0)
    ax.scatter([0], [0], marker="*", s=120, color="k", zorder=6)


# --- A: geometry + clean expert ---------------------------------------------
axA = fig.add_subplot(gs[0, 0])
draw_geometry(axA)
rng = np.random.RandomState(3)
for _ in range(6):
    p = np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.8, 1.0)])
    tr = [p.copy()]
    for _ in range(80):
        p = T.wall(p + T.expert_action(p))
        tr.append(p.copy())
        if np.linalg.norm(p - T.G) < T.DOCK_TOL:
            break
    tr = np.array(tr)
    axA.plot(tr[:, 0], tr[:, 1], color="tab:blue", lw=1.3, alpha=0.8)
axA.text(0.09, 0.38, "align band:\ndemos carry\n$0.08\\,t(\\nu{=}2)$\nlateral tremor",
         fontsize=9, color="#a00000")
axA.text(0.09, 0.14, "clean band:\ndogleg center-line,\ndock tol 0.008",
         fontsize=9, color="#1a7a1a")
axA.set_xlim(-0.25, 0.3)
axA.set_ylim(-0.05, 1.02)
axA.set_title("A  geometry + clean expert paths (eval conditions; expert SR 1.00)",
              fontsize=10, loc="left")

# --- B: actual training demos ----------------------------------------------
axB = fig.add_subplot(gs[0, 1])
draw_geometry(axB)
drng = np.random.RandomState(0)
for _ in range(10):
    st, ac = T.gen_episode(drng)
    tr = np.array(st)
    axB.plot(tr[:, 0], tr[:, 1], lw=0.9, alpha=0.75)
axB.set_xlim(-0.25, 0.3)
axB.set_ylim(-0.05, 1.02)
axB.set_title("B  training demonstrations: heavy-tailed wiggle in the align band,\n"
              "clean dogleg tracking below (the human-demo anatomy)", fontsize=10,
              loc="left")

# --- C: the labels the regressor sees --------------------------------------
axC = fig.add_subplot(gs[1, 0])
X, Y, Yc, AL = T.build_dataset(np.random.RandomState(0))
sel = X[:, 1] < 0.75
axC.scatter(Y[sel][:, 0], X[sel][:, 1], s=4, alpha=0.25,
            c=np.where(AL[sel], "#d62728", "#2ca02c"))
axC.axhspan(T.Y_ALIGN_LO, T.Y_TOP, color="#d62728", alpha=0.06)
axC.axhspan(T.Y_DOCK, T.Y_ALIGN_LO, color="#2ca02c", alpha=0.06)
axC.set_xlabel("lateral action label $a_x$ (first step of chunk)")
axC.set_ylabel("height $y$")
axC.set_xlim(-0.09, 0.09)
axC.set_title("C  what the regressor sees: align labels are heavy-tailed around the\n"
              "true signal (red); clean-band labels trace the fine dogleg law (green)",
              fontsize=10, loc="left")

# --- D: closed-loop rollouts l2 vs ht --------------------------------------
axD = fig.add_subplot(gs[1, 1])
draw_geometry(axD)
norm = T.Norm(X, Y)
Xn, Yn = norm.nx(X), norm.ny(Y)
nets = {}
MEDIAN_SEED = {"l2": 7, "ht": 2}  # median-SR seeds from toyhuman_r3.json
for arm in ("l2", "ht"):
    ck = f"human_ck_{arm}_med.pt"
    if os.path.exists(ck):
        nets[arm] = torch.load(ck, map_location=T.DEV, weights_only=False)
    else:
        nets[arm], _ = T.train(arm, MEDIAN_SEED[arm], Xn, Yn)
        torch.save(nets[arm], ck)
    print(f"{arm} ready", flush=True)
CO = {"l2": "#d62728", "ht": "#9467bd"}
for arm in ("l2", "ht"):
    err = np.random.RandomState(555 + 8)
    rr = np.random.RandomState(999)
    n_ok = 0
    for i in range(12):
        p = np.array([rr.uniform(-0.2, 0.2), rr.uniform(0.8, 1.0)])
        tr = [p.copy()]
        ok = False
        for _ in range(150 // T.H):
            on = torch.tensor(norm.nx(p[None].astype(np.float32)), device=T.DEV)
            with torch.no_grad():
                chunk = norm.uy(nets[arm](on)[0].cpu().numpy()).reshape(T.H, 2)
            for a in chunk:
                p = T.wall(p + np.clip(a, -T.CLIP, T.CLIP))
                tr.append(p.copy())
                if np.linalg.norm(p - T.G) < T.DOCK_TOL:
                    ok = True
                    break
                if p[1] < -0.05:
                    break
            if ok or p[1] < -0.05:
                break
        n_ok += ok
        tr = np.array(tr)
        axD.plot(tr[:, 0], tr[:, 1], color=CO[arm], lw=1.0,
                 alpha=0.8 if ok else 0.45, ls="-" if ok else "--")
        if not ok:
            axD.scatter(tr[-1, 0], tr[-1, 1], marker="x", s=40, color=CO[arm])
    print(f"{arm}: {n_ok}/12 dock", flush=True)
hnd = [plt.Line2D([0], [0], color=CO[a], lw=2) for a in ("l2", "ht")]
axD.legend(hnd, ["MSE", "hetero-t"], fontsize=9, loc="upper right")
axD.set_xlim(-0.25, 0.3)
axD.set_ylim(-0.09, 1.02)
axD.set_title("D  trained policies, clean closed loop (median seeds): MSE inherits the\n"
              "corrupted align fit and misses the dock; hetero-t docks", fontsize=10,
              loc="left")

fig.savefig("human_setting_fig.png", dpi=140)
print("wrote human_setting_fig.png", flush=True)
