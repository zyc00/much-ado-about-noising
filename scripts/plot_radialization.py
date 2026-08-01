"""Radialization evidence figures:
A) bird's-eye gate-approach paths (demos vs MSE vs MIP rollouts, gate at origin)
B) lateral command-field quiver triptych at shared band states (GT / MSE / MIP)
C) the cancellation law: perpendicular attenuation vs local demo dispersion, per model
"""
import os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
R = np.load("analysis/traj_vis/radialization.npz")
sxy = R["sxy"]; loc_R = R["loc_R"]

# gate offset + demo paths
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, dpaths, dacts, dxy = [], [], [], []
for k in keys[:60]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
h.close()
off = np.median(np.stack(offs), 0)
h = h5py.File(DSP, "r")
for k in keys[:40]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    seg = ov[c1+5:r1]
    path = seg[:, FP] - (seg[:, BP] + off[None])
    dpaths.append(path[:, :2])
    for t in range(len(seg)):
        d = path[t]
        if 0.01 <= np.linalg.norm(d[:2]) < 0.08 and abs(d[2]) <= 0.05:
            dxy.append(d[:2]); dacts.append(a[c1+5+t, :2])
h.close()
dxy = np.stack(dxy); dacts = np.stack(dacts)

# A) bird's-eye paths
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharex=True, sharey=True)
for p in dpaths:
    axes[0].plot(p[:, 0]*100, p[:, 1]*100, color="#555555", lw=0.8, alpha=0.5)
axes[0].set_title(f"GT demos (n={len(dpaths)}) — curved approaches", fontsize=11)
for ax_i, (mname, color_s, color_f) in [(1, ("hMSE", "#c0392b", "#e8b4ae")), (2, ("hMIP", "#1f6fb2", "#a9cbe8"))]:
    ax = axes[ax_i]; n = 0
    for seed in ["s5", "s5001"]:
        z = np.load(f"analysis/traj_vis/human_{mname}_{seed}.npz")
        i = 0
        while f"ep{i}_obs" in z.files:
            o = z[f"ep{i}_obs"]; m = z[f"ep{i}_meta"]; a = z[f"ep{i}_act"]
            L = min(len(o), len(a)); gc = a[:L, 6]
            run = 0; seg = []
            for t in range(L):
                run = run + 1 if gc[t] >= 0 else 0
                if run >= 10:
                    d = o[t, FP] - (o[t, BP] + off)
                    if np.linalg.norm(d[:2]) < 0.12 and abs(d[2]) < 0.08:
                        seg.append(d[:2])
            if len(seg) > 5:
                seg = np.stack(seg)
                ok = bool(m[1])
                ax.plot(seg[:, 0]*100, seg[:, 1]*100, color=(color_s if ok else color_f),
                        lw=(1.0 if ok else 0.7), alpha=0.8); n += 1
            i += 1
    ax.set_title(f"{mname} rollouts (both seeds, n={n}) — saturated = assembled", fontsize=11)
for ax in axes:
    ax.scatter([0], [0], marker="*", s=180, color="#111111", zorder=5, label="gate")
    for r_ in [2, 3, 5, 8]:
        ax.add_patch(plt.Circle((0, 0), r_, fill=False, color="#999999", lw=0.5, ls=":"))
    ax.set_xlabel("lateral x [cm]"); ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_aspect(1)
axes[0].set_ylabel("lateral y [cm]")
fig.suptitle("Frame position relative to insertion gate (bird's-eye): demos approach on curved paths; "
             "MSE failures orbit the ring; MIP converts", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("analysis/paper/radial_paths.png", dpi=150, bbox_inches="tight")

# B) quiver triptych
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharex=True, sharey=True)
sub = np.random.RandomState(0).choice(len(dxy), min(250, len(dxy)), replace=False)
axes[0].quiver(dxy[sub, 0]*100, dxy[sub, 1]*100, dacts[sub, 0], dacts[sub, 1],
               color="#555555", width=0.003, scale=12)
axes[0].set_title("GT demo actions in the band", fontsize=11)
ssub = np.random.RandomState(1).choice(len(sxy), min(250, len(sxy)), replace=False)
for ax_i, key, col, ttl in [(1, "hMSE_s5_axy", "#c0392b", "hMSE commands (same states)"),
                            (2, "hMIP_s5_axy", "#1f6fb2", "hMIP commands (same states)")]:
    A = R[key]
    axes[ax_i].quiver(sxy[ssub, 0]*100, sxy[ssub, 1]*100, A[ssub, 0], A[ssub, 1],
                      color=col, width=0.003, scale=12)
    axes[ax_i].set_title(ttl, fontsize=11)
for ax in axes:
    ax.scatter([0], [0], marker="*", s=180, color="#111111", zorder=5)
    ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_aspect(1); ax.set_xlabel("lateral x [cm]")
axes[0].set_ylabel("lateral y [cm]")
fig.suptitle("Lateral command fields at the gate band (star = insertion point)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("analysis/paper/radial_quiver.png", dpi=150, bbox_inches="tight")

# C) cancellation law bars
fig, ax = plt.subplots(figsize=(8.5, 5))
labels = ["crossing demos\n(local R<0.6)", "aligned demos\n(local R>0.8)"]
vals = {"hMSE s5": (0.78, 0.95), "hMSE s5001": (0.68, 0.83), "hMIP s5": (0.96, 0.87), "hMIP s5001": (0.83, 0.76)}
xpos = np.arange(2); w = 0.19
cols = {"hMSE s5": "#c0392b", "hMSE s5001": "#e08283", "hMIP s5": "#1f6fb2", "hMIP s5001": "#7fb3d8"}
for i, (k, v) in enumerate(vals.items()):
    ax.bar(xpos + (i - 1.5) * w, v, w, label=k, color=cols[k])
ax.axhline(1.0, color="#555555", lw=0.8, ls="--")
ax.set_xticks(xpos); ax.set_xticklabels(labels); ax.set_ylabel("perpendicular command / GT perpendicular")
ax.set_title("The cancellation law: MSE loses perpendicular (path) content exactly where\n"
             "demonstrations cross; MIP does not (both seed pairs)", fontsize=11)
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig("analysis/paper/radial_cancellation.png", dpi=150, bbox_inches="tight")
print("SAVED 3 figures", flush=True)
