"""Visualization of the tube-funnel setting (geometry + demos per session)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import toytube as T

BANDS = [("tube\n(speed noise)", T.Y_TUBE_LO, 1.0, "#fde0dd"),
         ("straight\n(clean)", T.Y_STR_LO, T.Y_TUBE_LO, "#e5f5e0"),
         ("funnel\n(dir noise)", T.Y_FUN_LO, T.Y_STR_LO, "#fff3c4"),
         ("insert\n(clean, scrape)", T.Y_DOCK, T.Y_FUN_LO, "#deebf7"),
         ("dock", -0.01, T.Y_DOCK, "#d9d9d9")]


def draw_geometry(ax):
    ys = np.linspace(0.0, 1.0, 600)
    cs = np.array([T.center(y) for y in ys])
    hw = np.array([T.half_width(y) for y in ys])
    for name, lo, hi, col in BANDS:
        ax.axhspan(lo, hi, color=col, zorder=0)
        ax.text(0.0345, (lo + hi) / 2, name, fontsize=7, va="center")
    ax.plot(cs - hw, ys, "k-", lw=1.2)
    ax.plot(cs + hw, ys, "k-", lw=1.2)
    ax.plot(cs, ys, "k--", lw=0.6, alpha=0.6)
    ax.plot([T.G[0]], [T.G[1]], "r*", ms=12, zorder=5)
    ax.add_patch(plt.Circle((T.G[0], T.G[1]), T.DOCK_TOL, fill=False,
                            color="r", lw=0.8))
    ax.set_xlim(-0.034, 0.056)
    ax.set_ylim(-0.02, 1.02)


def demos(noise_type, n=8, seed=0):
    os.environ["NOISE_TYPE"] = noise_type
    T.NOISE_TYPE = noise_type
    rng = np.random.RandomState(seed)
    return [T.gen_episode(rng) for _ in range(n)]


fig, axes = plt.subplots(1, 5, figsize=(16, 6.2))

ax = axes[0]
draw_geometry(ax)
ax.set_title("geometry (x stretched)", fontsize=9)
ax.set_xlabel("x")
ax.set_ylabel("y")

for ax, nt, title in ((axes[1], "none", "demos: none (clean)"),
                      (axes[2], "dir", "demos: dir (funnel tremor)"),
                      (axes[3], "speed", "demos: speed (tube pace)")):
    draw_geometry(ax)
    for st, ac in demos(nt):
        s = np.array(st)
        ax.plot(s[:, 0], s[:, 1], "-", lw=0.7, alpha=0.75)
        ax.plot(s[:, 0], s[:, 1], ".", ms=1.6, alpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_yticklabels([])

ax = axes[4]
for nt, col in (("none", "tab:green"), ("speed", "tab:red")):
    for i, (st, ac) in enumerate(demos(nt, n=6)):
        s = np.array(st)
        ax.plot(np.arange(len(s)), s[:, 1], "-", lw=0.8, alpha=0.8,
                color=col, label=nt if i == 0 else None)
ax.axhspan(T.Y_TUBE_LO, 1.0, color="#fde0dd", zorder=0)
ax.axhspan(T.Y_STR_LO, T.Y_TUBE_LO, color="#e5f5e0", zorder=0)
ax.set_title("pace: y vs step (speed mixture\nin tube, uniform below)", fontsize=9)
ax.set_xlabel("step")
ax.set_ylabel("y")
ax.legend(fontsize=8, loc="upper right")

for ax in axes[:4]:
    ax.tick_params(labelsize=7)
axes[4].tick_params(labelsize=7)
fig.suptitle("tube-funnel toy: speed noise in tube, direction noise in funnel, "
             "clean precision insert (scrape walls), dock tol 0.008", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("tube_setting_fig.png", dpi=160)
print("wrote tube_setting_fig.png")
