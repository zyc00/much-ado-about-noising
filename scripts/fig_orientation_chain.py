"""Real-trajectory evidence for the precision mechanism: frame
misorientation over the episode (carry reorientation -> residual at
descent entry), natural episodes, no perturbation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.load("analysis/orientpack.npz", allow_pickle=True)
arms, asm, entry = z["arms"], z["asm"], z["entry"]
n = len(arms)
angs = [z[f"ang_{i}"] for i in range(n)]
zn = np.load("analysis/natpack.npz", allow_pickle=True)
kn = {(str(zn["arms"][i]), int(zn["seeds"][i])): i for i in range(n)}
minres = np.full(n, np.nan)
for i in range(n):
    if entry[i] < 0:
        continue
    eef = zn[f"eef_{kn[(str(arms[i]), int(z['seeds'][i]))]}"]
    zz = eef[:, 2]
    te = min(entry[i], len(zz) - 1)
    tb = te + int(np.argmin(zz[te:te + 40]))
    hi = np.where(zz > 1.06)[0]
    t_lift = hi[0] if len(hi) else 0
    if asm[i]:
        minres[i] = angs[i][-1]
    else:
        minres[i] = angs[i][min(tb, len(angs[i]) - 1)]

order = ["l2", "hgcbest", "mip"]
titles = {"l2": "L2", "hgcbest": "HT+condreg", "mip": "MIP"}

fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.4),
                         gridspec_kw={"width_ratios": [1, 1, 1, 0.75]})
for j, tg in enumerate(order):
    ax = axes[j]
    for i in range(n):
        if arms[i] != tg or entry[i] < 0:
            continue
        te = min(entry[i], len(angs[i]) - 1)
        cut = min(te + 5, len(angs[i]))
        c, lw, al, zo = (("tab:green", 1.0, 0.5, 1) if asm[i] else
                         ("tab:red", 1.7, 0.95, 3))
        ax.plot(np.arange(cut), angs[i][:cut], color=c, lw=lw, alpha=al,
                zorder=zo)
        ax.plot(te, angs[i][te], marker="o", color=c, ms=5, zorder=zo + 1,
                ls="none")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 130)
    ax.axhline(4.5, color="k", ls="--", lw=0.9)
    if j == 0:
        ax.text(3, 4.8, "4.5$^\\circ$", fontsize=8.5)
        ax.set_ylabel("frame misorientation vs inserted\norientation "
                      "(deg, log)")
    ax.set_xlabel("episode step")
    nf = sum(1 for i in range(n) if arms[i] == tg and not asm[i])
    ax.set_title(f"{titles[tg]} — {30 - nf}/30 success", fontsize=11)
    ax.grid(alpha=0.25)

ax = axes[3]
np.random.seed(0)
for j, tg in enumerate(order):
    for i in range(n):
        if arms[i] != tg or entry[i] < 0:
            continue
        if not np.isfinite(minres[i]):
            continue
        c = "tab:green" if asm[i] else "tab:red"
        ax.plot(minres[i], j + np.random.uniform(-0.16, 0.16), marker="o",
                color=c, ms=6, alpha=0.85, ls="none")
ax.axvline(4.5, color="k", ls="--", lw=0.9)
ax.set_xscale("log")
ax.set_xlim(0.4, 200)
ax.set_yticks(range(3))
ax.set_yticklabels([titles[t] for t in order], fontsize=10)
ax.set_xlabel("residual at attempt bottom (deg, log)")
ax.set_title("success: at completion; failure: at first-attempt\n"
             "bottom (green success, red failure)", fontsize=10)
ax.grid(alpha=0.25, axis="x")

fig.suptitle(
    "Script data, natural episodes (no kick): the frame is reoriented "
    "$\\approx$100$^\\circ$ during carry and the residual must converge "
    "before the first insertion attempt (dots: descent entry). Successes "
    "finish at 0.5\u201310$^\\circ$ (median 2\u20134); failures commit "
    "at 9\u2013160$^\\circ$. All 70 successes are single-attempt; no "
    "successful retry exists in 90 episodes.",
    fontsize=11.5, y=1.0)
fig.tight_layout()
fig.savefig("analysis/paper/real_orientation_chain.png", dpi=150,
            bbox_inches="tight")
print("saved analysis/paper/real_orientation_chain.png")
