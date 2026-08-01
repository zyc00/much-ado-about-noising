"""Distance-controlled dispersion test (closing the confound in the cancellation law) +
single-path tangent test (MIP characterization). Uses radialization.npz + demo band data."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py
from scipy.spatial import cKDTree

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
R = np.load("analysis/traj_vis/radialization.npz")
sxy = R["sxy"]; loc_R = R["loc_R"]

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, dxy, dact = [], [], []
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
h.close()
off = np.median(np.stack(offs), 0)
h = h5py.File(DSP, "r")
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
    for t in range(c1 + 5, r1):
        d = ov[t, FP] - (ov[t, BP] + off)
        if 0.01 <= np.linalg.norm(d[:2]) < 0.08 and abs(d[2]) <= 0.05:
            dxy.append(d[:2]); dact.append(a[t, :2])
h.close()
dxy = np.stack(dxy); dact = np.stack(dact)
u = dact / (np.linalg.norm(dact, axis=1, keepdims=True) + 1e-9)
dtree = cKDTree(dxy)

dist = np.linalg.norm(sxy, axis=1)
# local stats per shared state
perpGT, near_dir = [], []
for p in sxy:
    _, idx = dtree.query(p, k=12)
    r_ = p / (np.linalg.norm(p) + 1e-9)
    av = dact[idx]
    perpGT.append(np.median(np.abs(av[:, 0]*r_[1] - av[:, 1]*r_[0])))
    near_dir.append(u[idx[0]])   # single nearest demo point's action direction
perpGT = np.array(perpGT); near_dir = np.stack(near_dir)

print("=== distance-controlled cancellation (perp/GTperp medians)", flush=True)
for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    A = R[f"{name}_axy"]
    r_ = sxy / (np.linalg.norm(sxy, axis=1, keepdims=True) + 1e-9)
    perp = np.abs(A[:, 0]*r_[:, 1] - A[:, 1]*r_[:, 0])
    att = perp / (perpGT + 1e-9)
    line = f"CTRL {name}:"
    for dl, dh in [(0.01, 0.03), (0.03, 0.08)]:
        for tag, m2 in [("cross", loc_R < 0.6), ("align", loc_R >= 0.6)]:
            m = (dist >= dl) & (dist < dh) & m2
            if m.sum() >= 30:
                line += f" | {int(dl*100)}-{int(dh*100)}cm/{tag}={np.median(att[m]):.2f}(n={m.sum()})"
    print(line, flush=True)

print("=== single-nearest-path tangent alignment (MIP characterization)", flush=True)
for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    A = R[f"{name}_axy"]
    ua = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    ct = (ua * near_dir).sum(1)
    m = loc_R < 0.6
    print(f"TANGENT {name}: cos-to-nearest-demo-dir @crossing p50={np.median(ct[m]):+.2f} | @aligned p50={np.median(ct[~m]):+.2f}", flush=True)
print("CTRL-DONE")
