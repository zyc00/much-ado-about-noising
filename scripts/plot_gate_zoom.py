"""Zoomed gate-frame comparison: frame-tip trajectories in the last 12cm around the gate.
Demos (gray), MIP successes (blue), student-t FAILURES (red thick). Two views: top-down
(x-y lateral plane) and side (lateral distance vs height)."""
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, demo_paths = [], []
for k in keys[:40]:
    o = h[f"data/{k}/obs"]
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    demo_paths.append(ov[c1+5:r1, FP] - (ov[c1+5:r1, BP] + offs[-1]))
h.close()
off = np.median(np.stack(offs), 0)
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
def paths(tag, want):
    z = np.load(f"analysis/traj_vis/human_{tag}.npz")
    out, i = [], 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        if bool(int(m[0])) == want:
            g0, gend = held_range(o, a)
            if g0 is not None:
                out.append(o[g0:gend, FP] - (o[g0:gend, BP] + off))
        i += 1
    return out
mip_ok = paths("hmipL286k", True)
ht_bad = paths("hheterot_s1000", False)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
R = 0.12
def draw(ax, view):
    for p in demo_paths:
        m = np.linalg.norm(p, axis=1) < R
        if view == "top": ax.plot(p[m, 0]*1000, p[m, 1]*1000, color="#999999", lw=0.7, alpha=0.5)
        else: ax.plot(np.linalg.norm(p[m, :2], axis=1)*1000, p[m, 2]*1000, color="#999999", lw=0.7, alpha=0.5)
    for p in mip_ok:
        m = np.linalg.norm(p, axis=1) < R
        if view == "top": ax.plot(p[m, 0]*1000, p[m, 1]*1000, color="#1f6fb2", lw=1.2, alpha=0.8)
        else: ax.plot(np.linalg.norm(p[m, :2], axis=1)*1000, p[m, 2]*1000, color="#1f6fb2", lw=1.2, alpha=0.8)
    for p in ht_bad:
        m = np.linalg.norm(p, axis=1) < R
        if view == "top": ax.plot(p[m, 0]*1000, p[m, 1]*1000, color="#c0392b", lw=1.8, alpha=0.95)
        else: ax.plot(np.linalg.norm(p[m, :2], axis=1)*1000, p[m, 2]*1000, color="#c0392b", lw=1.8, alpha=0.95)
    if view == "top":
        ax.plot(0, 0, marker="*", color="k", ms=16); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title("top-down (lateral plane); star = gate")
        ax.set_xlim(-90, 90); ax.set_ylim(-90, 90); ax.set_aspect("equal")
    else:
        ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("lateral distance [mm]"); ax.set_ylabel("height above gate [mm]")
        ax.set_title("side (lat vs height)")
        ax.set_xlim(0, 90); ax.set_ylim(-20, 110)
draw(axes[0], "top"); draw(axes[1], "side")
fig.suptitle("Final 12cm at the gate: demos (gray), MIP successes (blue), student-t FAILURES (red)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("analysis/paper/gate_zoom_ht_vs_mip.png", dpi=150, bbox_inches="tight")
print("SAVED")
