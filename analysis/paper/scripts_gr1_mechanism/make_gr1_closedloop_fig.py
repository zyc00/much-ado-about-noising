"""Closed-loop concrete example: HT464 success vs MSE multi-attempt failure on CuttingboardToBasket (env 2, first episode; scenes differ)."""
import numpy as np, io, contextlib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_rollouts_gr1 as A
from gr1_fk_lib import ee_traj
task = "PosttrainPnPNovelFromCuttingboardToBasketSplitA"; fr = np.load("ep_frames_basket_e2.npz", allow_pickle=True)
R = [6, 8, 7, 9]; thr = 0.686; col = {"ht464": "#1f77b4", "mse": "#d62728"}; lab = {"ht464": "HT head (nu=464): success in 26 chunks", "mse": "MSE head: timeout (90 chunks), 5 grasp attempts"}
E = {}
for label in ["ht464", "mse"]:
    z = np.load(f"trajdump/{label}_{task}.npz", allow_pickle=True); eps = [e for e in A.episodes(z) if e["complete"]]; cnt = {}
    for e in eps:
        k = cnt.get(e["env"], 0); cnt[e["env"]] = k + 1
        if e["env"] == 2 and k == 0: E[label] = e; e["waist"] = z["obs.state.waist"][e["t0"]:e["t1"] + 1, 2, 0]
fig = plt.figure(figsize=(18, 9.5)); gs = fig.add_gridspec(4, 8, height_ratios=[1.1, 1.1, 1, 1], hspace=0.55, wspace=0.3)
for row, label in enumerate(["ht464", "mse"]):
    frames = fr[f"{label}_{task}|frames"]; ts = fr[f"{label}_{task}|t"]; n = len(ts); pick = np.linspace(0, n - 1, 8).astype(int)
    for j, i in enumerate(pick):
        ax = fig.add_subplot(gs[row, j]); ax.imshow(frames[i]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"chunk {ts[i]} ({ts[i]*0.4:.1f} s)", fontsize=8)
        if j == 0: ax.set_ylabel(lab[label].split(":")[0], fontsize=9, color=col[label])
ax1 = fig.add_subplot(gs[2, :]); ax2 = fig.add_subplot(gs[3, :])
for label in ["ht464", "mse"]:
    e = E[label]; c = e["st_hand"][:, R].mean(1); t = np.arange(e["n"])
    ax1.plot(t, c, color=col[label], lw=1.6, label=lab[label]); 
    cl = np.where(np.diff(np.concatenate([[0], (c > thr).astype(int)])) == 1)[0]; ax1.scatter(cl, c[cl], color=col[label], s=40, marker="v", zorder=5)
    off = np.sqrt(((e["arm"] - e["st_arm"][:, None, :]) ** 2).mean((1, 2))); _, ee = ee_traj(e["st_arm"], e["waist"])
    ax2.plot(t, off, color=col[label], lw=1.6, label=f"{label}: commanded arm offset rms (rad), mean {off.mean():.3f}")
    ax2.plot(t, (ee[:, 2] - ee[0, 2]) , color=col[label], lw=1.2, ls="--", label=f"{label}: right wrist height change (m)")
ax1.axhline(thr, color="gray", ls=":", lw=1); ax1.set_ylabel("right-hand closure (measured)"); ax1.set_title("hand closure over the episode (triangles = closure events)", fontsize=10); ax1.legend(fontsize=8, loc="upper right"); ax1.set_xlim(0, 90)
ax2.set_xlabel("chunk index (0.4 s each)"); ax2.set_title("commanded arm amplitude and wrist height", fontsize=10); ax2.legend(fontsize=7, loc="upper right", ncol=2); ax2.set_xlim(0, 90); ax2.grid(alpha=0.3)
fig.suptitle("Closed-loop example, CuttingboardToBasket (env 2, first episode of each run; scenes are not identical across runs)", fontsize=11)
fig.savefig("fig_gr1_closedloop_basket.png", dpi=120, bbox_inches="tight"); print("saved")
