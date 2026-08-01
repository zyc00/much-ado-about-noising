"""Plot |bias|(distance-to-clean-support) for all 5 models on the SAME test states,
into analysis/recovery/mip_saves_mse_fails.png. Bias = ||E[a_pred - a_gt]|| of the
executed (first) action, per distance bin. Shaded band = bootstrap 90% CI on |bias|."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ref = np.load("analysis/recovery/err_vectors.npz")
c20 = np.load("analysis/recovery/err_clean20k.npz")
p20 = np.load("analysis/recovery/err_mip20k.npz")
d6k = np.load("analysis/recovery/err_dart6k.npz")

# (name, dist, exec-err (N,6), color, linestyle)
SERIES = [
    ("MSE-clean-2k",  ref["dist"], ref["eMSE"][:, 0, :],  "tab:red",    "-"),
    ("MSE-clean-20k", c20["dist"], c20["e20"][:, 0, :],   "darkred",    "--"),
    ("MIP-2k",        ref["dist"], ref["eMIP"][:, 0, :],  "tab:blue",   "-"),
    ("MIP-20k",       p20["dist"], p20["eP20"][:, 0, :],  "navy",       "--"),
    ("DART-6k (MSE)", d6k["dist"], d6k["eDART"][:, 0, :], "tab:green",  "-"),
]

EDGES = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8]
centers = [0.5 * (EDGES[i] + EDGES[i + 1]) for i in range(len(EDGES) - 1)]
rng = np.random.RandomState(0)

fig, ax = plt.subplots(figsize=(10, 6.2))
for name, dist, e, col, ls in SERIES:
    xs, ys, lo, hi = [], [], [], []
    for i in range(len(EDGES) - 1):
        m = (dist >= EDGES[i]) & (dist < EDGES[i + 1]); n = int(m.sum())
        if n < 15:
            continue
        X = e[m]
        b = np.linalg.norm(X.mean(0))
        boots = [np.linalg.norm(X[rng.randint(0, n, n)].mean(0)) for _ in range(300)]
        l, h = np.percentile(boots, [5, 95])
        xs.append(centers[i]); ys.append(b); lo.append(l); hi.append(h)
    ax.plot(xs, ys, ls, color=col, lw=2.2, marker="o", ms=5, label=name)
    ax.fill_between(xs, lo, hi, color=col, alpha=0.12)

# in/off-support reference lines (clean expert -> training cloud: p95=1.79, p99=2.45)
ax.axvspan(0, 1.79, color="gray", alpha=0.06)
ax.axvline(1.79, color="gray", ls=":", lw=1)
ax.text(1.82, ax.get_ylim()[1] * 0.93, "p95 in-domain", fontsize=8, color="gray")
ax.axvline(4.0, color="k", ls="--", lw=1)
ax.text(4.05, ax.get_ylim()[1] * 0.93, "PNR (d≈4)", fontsize=8)

ax.set_xlabel("distance to CLEAN expert support  (1-NN, z-scored 53-dim obs)", fontsize=11)
ax.set_ylabel("|bias| = ‖E[a_pred − a_gt]‖  (executed action)", fontsize=11)
ax.set_title("Off-support action bias vs distance — MSE blows up, MIP halves it & scales,\n"
             "DART (support coverage) eliminates it  |  same 41k held-out states, same metric",
             fontsize=11)
ax.grid(alpha=0.25); ax.legend(fontsize=10, loc="upper left")
ax.set_xlim(0, 7)
plt.tight_layout()
plt.savefig("analysis/recovery/mip_saves_mse_fails.png", dpi=120)
print("saved analysis/recovery/mip_saves_mse_fails.png")
