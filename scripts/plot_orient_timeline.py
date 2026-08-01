"""The reorientation maneuver, aligned: in-hand orientation error vs time-to-ring-entry
for demos / passing / failing episodes."""
import os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
FQ = slice(17, 21); BP = slice(7, 10); FP = slice(21, 24)
def qn(q): return q / (np.linalg.norm(q) + 1e-9)
def qangle(q1, q2): return np.degrees(2*np.arccos(np.clip(abs(float(np.dot(q1, q2))), -1, 1)))

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
dq, offs, demos = [], [], []
for k in keys[:80]:
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
    dq.append(qn(ov[r1-20, FQ]))
    demos.append((ov, c1, r1))
h.close()
off = np.median(np.stack(offs), 0)
dq = np.stack(dq); qref = dq[0]
for i in range(1, len(dq)):
    if np.dot(dq[i], qref) < 0: dq[i] = -dq[i]
qmu = qn(dq.mean(0))
def err_series(ov):
    out = []
    for t in range(len(ov)):
        q = qn(ov[t, FQ])
        if np.dot(q, qmu) < 0: q = -q
        out.append(qangle(q, qmu))
    return np.array(out)

def ring_entry(ov, lo, hi):
    for t in range(lo, hi):
        d = ov[t, FP] - (ov[t, BP] + off)
        if np.linalg.norm(d[:2]) < 0.08 and abs(d[2]) < 0.05:
            return t
    return None

WIN = 140
def aligned_end(ov, anchor, held):
    """error over the last WIN steps BEFORE the anchor (release / episode end), held steps only"""
    e = err_series(ov)
    xs = np.arange(-WIN, 1)
    ys = np.full(len(xs), np.nan)
    for i, dt in enumerate(xs):
        t = anchor + dt
        if 0 <= t < len(e) and (held is None or (t < len(held) and held[t])):
            ys[i] = e[t]
    return ys

curves = {"demos": [], "pass": [], "fail": []}
for ov, c1, r1 in demos:
    curves["demos"].append(aligned_end(ov, r1 - 1, None))
for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        L = min(len(o), len(a)); gc = a[:L, 6]
        run = 0; g0 = None
        for t in range(L):
            run = run + 1 if gc[t] >= 0 else 0
            if run >= 15: g0 = t - 14; break
        if g0 is None: i += 1; continue
        held = np.zeros(L, dtype=bool); run = 0
        for t in range(L):
            run = run + 1 if gc[t] >= 0 else 0
            held[t] = run >= 10
        if int(m[1]):
            rel = next((t for t in range(g0 + 30, L) if held[t-1] and gc[t] < 0), None)
            anchor = rel - 1 if rel else L - 1
        else:
            anchor = L - 1
        curves["pass" if int(m[1]) else "fail"].append(aligned_end(o, anchor, held))
        i += 1

fig, ax = plt.subplots(figsize=(9.5, 5.5))
xs = np.arange(-WIN, 1)
for key, col, lbl in [("demos", "#555555", f"demonstrators (n={len(curves['demos'])})"),
                      ("pass", "#1f6fb2", f"passing episodes (n={len(curves['pass'])})"),
                      ("fail", "#c0392b", f"failing episodes (n={len(curves['fail'])})")]:
    Y = np.stack(curves[key])
    med = np.nanmedian(Y, 0); lo = np.nanpercentile(Y, 25, 0); hi = np.nanpercentile(Y, 75, 0)
    ax.plot(xs, med, color=col, lw=2, label=lbl)
    ax.fill_between(xs, lo, hi, color=col, alpha=0.15)
ax.axvline(0, color="#999999", lw=0.8, ls=":")
ax.text(-138, 33, "aligned on release (demos/passing) or episode end (failing); held-frame steps only", fontsize=8, color="#666666"); ax.axhline(10, color="#999999", lw=0.8, ls="--")
ax.text(2, 10.5, "insertable threshold (~10 deg)", fontsize=8, color="#666666")
ax.set_xlabel("steps before release (demos, passing) / episode end (failing)")
ax.set_ylabel("in-hand orientation error [deg]")
ax.set_title("The terminal reorientation maneuver: demos and passing episodes rotate the frame\n"
             "~18 deg -> ~9 deg after reaching the gate; failing episodes never do", fontsize=11)
ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, 35)
plt.tight_layout()
plt.savefig("analysis/paper/reorient_timeline.png", dpi=150, bbox_inches="tight")
print("SAVED analysis/paper/reorient_timeline.png")
