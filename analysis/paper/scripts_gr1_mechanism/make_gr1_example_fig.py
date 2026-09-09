"""Concrete example: one training state; GT chunk vs MSE vs HT predictions (arm joints in rad, wrist path via FK), with dataset frames."""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gr1_norm_utils import lohi, unnorm, load_stats
from gr1_fk_lib import fk
dump = sys.argv[1] if len(sys.argv) > 1 else "fit_dump_gr1_v2_partial.npz"; idx = int(sys.argv[2]) if len(sys.argv) > 2 else 220; out = sys.argv[3] if len(sys.argv) > 3 else "fig_gr1_example.png"
z = np.load(dump, allow_pickle=True); T, D = 8, 29; A = load_stats("statsA.json"); lo, hi = lohi(A)
keys = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]; lo_s = np.concatenate([np.array(A["state"][k]["q01"]) for k in keys]); hi_s = np.concatenate([np.array(A["state"][k]["q99"]) for k in keys])
GT = unnorm(z["gt"].reshape(-1, T, D), lo, hi); st = unnorm(z["state"][:, :29], lo_s, hi_s)
models = [m for m in ["mse", "ht464", "flow"] if m + "_pred" in z.files]; P = {m: unnorm(z[m + "_pred"].mean(1).reshape(-1, T, D), lo, hi) for m in models}
ds, ep, step = str(z["ident_ds"][idx]), int(z["ident_ep"][idx]), int(z["ident_step"][idx])
g = GT[idx]; q0 = st[idx]
JN = ["shoulder pitch", "shoulder roll", "shoulder yaw", "elbow pitch", "wrist yaw", "wrist roll", "wrist pitch"]
col = {"gt": "k", "mse": "#d62728", "ht464": "#1f77b4", "flow": "#2ca02c"}; lab = {"gt": "demonstration", "mse": "MSE head", "ht464": "HT head (nu=464)", "flow": "flow head (mean of 4)"}
def wrist_path(off):  # (T,29) offsets -> right wrist positions (T+1,3) incl. current
    pts = [fk(q0[:7], q0[7:14], q0[26:29])[1]]
    for t in range(T):
        q = q0 + off[t]; pts.append(fk(q[:7], q[7:14], q0[26:29])[1])
    return np.array(pts) * 100
paths = {"gt": wrist_path(g), **{m: wrist_path(P[m][idx]) for m in models}}
try:
    fr = np.load("frames_cand.npz"); frames = [(k, fr[k]) for k in fr.files if k.startswith(f"{ds}|{ep}|")]; frames.sort(key=lambda kv: int(kv[0].split("|")[2]))
except Exception: frames = []
fig = plt.figure(figsize=(17, 10.5)); gs = fig.add_gridspec(3, 8, height_ratios=[1.05, 1, 1], hspace=0.5, wspace=0.55)
# row 0: frames + wrist path
for i, (k, im) in enumerate(frames[:3]):
    ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); f = int(k.split("|")[2]); ax.set_title(f"frame {f} (chunk step {f - step})", fontsize=10)
ax = fig.add_subplot(gs[0, 6:8]); 
for m in ["gt"] + models:
    p = paths[m]; ax.plot(p[:, 0] - paths["gt"][0, 0], p[:, 2] - paths["gt"][0, 2], "-o", ms=3, color=col[m], label=lab[m], lw=2 if m == "gt" else 1.4)
ax.set_xlabel("forward x (cm)"); ax.set_ylabel("up z (cm)"); ax.set_title("right wrist path over the chunk (FK)", fontsize=10); ax.axis("equal"); ax.legend(fontsize=7, loc="best")
# rows 1-2: right arm joints (7) + summary panel
resid = {m: np.sqrt(((P[m][idx] - g)[:, 7:14] ** 2).mean()) for m in models}
for j in range(7):
    r, c = divmod(j, 4); ax = fig.add_subplot(gs[1 + r, c * 2:(c + 1) * 2])
    ax.plot(range(T), g[:, 7 + j], "k-o", ms=3, lw=2, label=lab["gt"])
    for m in models: ax.plot(range(T), P[m][idx][:, 7 + j], "-o", ms=3, lw=1.4, color=col[m], label=lab[m])
    ax.set_title(f"right {JN[j]}", fontsize=10); ax.set_xlabel("chunk step"); ax.set_ylabel("offset from current (rad)"); ax.grid(alpha=0.3)
    if j == 0: ax.legend(fontsize=7)
ax = fig.add_subplot(gs[2, 6:8]); ax.axis("off")
txt = [f"{ds.replace('gr1_unified.', '')}", f"episode {ep}, frame {step}", "", "right-arm residual rms (rad):"] + [f"  {lab[m]}: {resid[m]:.3f}" for m in models] + ["", "wrist endpoint error at step 7 (cm):"] + [f"  {lab[m]}: {np.linalg.norm(paths[m][-1] - paths['gt'][-1]):.2f}" for m in models] + ["", f"demo wrist displacement: {np.linalg.norm(paths['gt'][-1] - paths['gt'][0]):.1f} cm"]
ax.text(0, 1, "\n".join(txt), va="top", fontsize=9, family="monospace")
fig.suptitle(f"GR1 training state: what each head predicts for the next 8 steps (0.4 s) — {ds.replace('gr1_unified.', '')} ep{ep} f{step}", fontsize=12)
fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out, {m: round(float(resid[m]), 4) for m in models}, "endpoint err cm", {m: round(float(np.linalg.norm(paths[m][-1] - paths['gt'][-1])), 2) for m in models})
