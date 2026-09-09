import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gr1_norm_utils import lohi, unnorm, load_stats
dump = sys.argv[1] if len(sys.argv) > 1 else "fit_dump_gr1_v2.npz"; out = sys.argv[2] if len(sys.argv) > 2 else "fig_gr1_mechanism.png"
z = np.load(dump, allow_pickle=True); T, D = 8, 29; GT = z["gt"].reshape(-1, T, D); N = len(GT); A = load_stats("statsA.json"); lo, hi = lohi(A)
models = [m for m in ["mse", "flow", "ht464"] if m + "_pred" in z.files]; P = {m: z[m + "_pred"].mean(1).reshape(N, T, D) for m in models}
col = {"mse": "#d62728", "flow": "#2ca02c", "ht464": "#1f77b4", "l1": "#9467bd"}; lab = {"mse": "MSE head", "flow": "flow head (mean of 4 draws)", "ht464": "HT head (nu=464)"}
ARM = slice(0, 14); HAND = slice(14, 26)
fig, axs = plt.subplots(1, 4, figsize=(19, 4.4)); 
# (a) residual by |GT| bin, arm, in rad
gphys = unnorm(GT, lo, hi); edges = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 10]; g = np.abs(GT[:, :, ARM]).ravel(); b = np.digitize(g, edges[1:-1]); x = np.arange(6)
for i, m in enumerate(models):
    r = (unnorm(P[m], lo, hi) - gphys)[:, :, ARM].ravel(); vals = [np.sqrt(np.mean(r[b == k] ** 2)) for k in range(6)]
    axs[0].bar(x + (i - 1) * 0.27, vals, 0.27, color=col[m], label=lab[m])
axs[0].set_xticks(x); axs[0].set_xticklabels([f"[{edges[k]},{edges[k+1]})\n{np.mean(b==k)*100:.0f}%" for k in range(6)], fontsize=8); axs[0].set_xlabel("|demo arm offset| bin (normalized units), share of elements"); axs[0].set_ylabel("residual rms (rad)"); axs[0].set_title("(a) arm residual by action magnitude", fontsize=10); axs[0].legend(fontsize=7)
# (b) per-sample residual: HT vs MSE scatter (arm, rad)
rp = {m: np.sqrt(np.mean(((unnorm(P[m], lo, hi) - gphys)[:, :, ARM]) ** 2, axis=(1, 2))) for m in models}
axs[1].scatter(rp["mse"], rp["ht464"], s=8, alpha=0.6, color="#1f77b4"); lim = max(rp["mse"].max(), rp["ht464"].max()) * 1.05; axs[1].plot([0, lim], [0, lim], "k--", lw=1); axs[1].set_xlim(0, lim); axs[1].set_ylim(0, lim)
axs[1].set_xlabel("MSE head: per-state arm residual rms (rad)"); axs[1].set_ylabel("HT head: per-state arm residual rms (rad)"); axs[1].set_title(f"(b) same 400 training states: HT lower on {np.mean(rp['ht464']<rp['mse'])*100:.0f}%", fontsize=10)
# (c) wrist endpoint error distribution (cm) from saved FK results
try:
    ee = np.load("fit_v2_ee.npz"); egt = ee["ee_gt"]
    for m in models:
        err = np.linalg.norm(ee[m][:, 0] - egt[:, 0], axis=1) * 100; axs[2].hist(err, bins=np.linspace(0, 6, 31), histtype="step", lw=1.8, color=col[m], label=f"{lab[m]} (median {np.median(err):.2f} cm)")
    axs[2].set_xlabel("right-wrist endpoint error at chunk step 7 (cm)"); axs[2].set_ylabel("states"); axs[2].set_title("(c) end-effector error of the predicted chunk", fontsize=10); axs[2].legend(fontsize=7)
except Exception as ex: axs[2].text(0.1, 0.5, f"no FK file: {ex}")
# (d) hand channels: residual and in-chunk flip commitment
H = GT[:, :, HAND]; jump = np.abs(np.diff(H, axis=1)).max(axis=(1, 2)); has = jump > 0.3
vals = []; 
for m in models:
    pr = P[m][:, :, HAND]; vals.append((np.sqrt(np.mean((pr - H) ** 2)), np.median(np.abs(np.diff(pr[has], axis=1)).max(axis=(1, 2))) / np.median(jump[has])))
xx = np.arange(len(models)); axs[3].bar(xx - 0.18, [v[0] for v in vals], 0.36, color=[col[m] for m in models], alpha=0.9); axs[3].set_ylabel("hand residual rms (normalized)"); ax2 = axs[3].twinx(); ax2.bar(xx + 0.18, [v[1] for v in vals], 0.36, color=[col[m] for m in models], alpha=0.4, hatch="//"); ax2.set_ylabel("predicted / demo hand-flip size (hatched)")
axs[3].set_xticks(xx); axs[3].set_xticklabels([lab[m].split(" (")[0] for m in models], fontsize=8); axs[3].set_title(f"(d) hand channels: residual (solid), flip commitment (hatched, n={has.sum()})", fontsize=10)
fig.suptitle(f"GR1 training-set fit on identical states (N={N}, {len(set(zip(z['ident_ds'], z['ident_ep'])))} episodes): where the HT head differs from MSE / flow", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
