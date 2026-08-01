"""Is the orientation drop mechanical funneling (user hypothesis)? For demos + passing
episodes: median in-hand orientation error binned by 3D distance of frame to gate.
If error stays ~18deg until distance < ~10-15mm and only drops during the final descent,
the drop is the insertion's EFFECT (hole guides the frame) — orientation is observation,
not cause. Also: orientation at FIRST 15mm-contact (insertion start) in demos = the actual
precondition tolerance."""
import os
import numpy as np, h5py
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
def err(q):
    q = qn(q)
    if np.dot(q, qmu) < 0: q = -q
    return qangle(q, qmu)

BINS = [(0, 0.005), (0.005, 0.010), (0.010, 0.015), (0.015, 0.025), (0.025, 0.04), (0.04, 0.08)]
def profile(rows):
    by = {b: [] for b in range(len(BINS))}
    for ov, lo_t, hi_t in rows:
        for t in range(lo_t, hi_t):
            d3 = np.linalg.norm(ov[t, FP] - (ov[t, BP] + off))
            for b, (lo, hi) in enumerate(BINS):
                if lo <= d3 < hi:
                    by[b].append(err(ov[t, FQ])); break
    return by

demo_by = profile([(ov, c1 + 5, r1) for ov, c1, r1 in demos])
line = "FUNNEL demos err-vs-3Ddist:"
for b, (lo, hi) in enumerate(BINS):
    if demo_by[b]:
        line += f" {int(lo*1000)}-{int(hi*1000)}mm={np.median(demo_by[b]):.1f}deg(n={len(demo_by[b])})"
print(line, flush=True)
# orientation at insertion start (first time 3D dist < 15mm)
start_err = []
for ov, c1, r1 in demos:
    for t in range(c1 + 5, r1):
        if np.linalg.norm(ov[t, FP] - (ov[t, BP] + off)) < 0.015:
            start_err.append(err(ov[t, FQ])); break
print(f"FUNNEL demos orientation at FIRST 15mm approach: p50={np.median(start_err):.1f}deg p90={np.quantile(start_err,0.9):.1f}deg (n={len(start_err)})", flush=True)

def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
import os as _o
for name in _o.environ.get("NAMES", "hMSE_s5 hMIP_s5 hMSE_s5001 hMIP_s5001").split():
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    rows = {"PASS": [], "FAIL": []}
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            rows["PASS" if int(m[1]) else "FAIL"].append((o, g0, gend))
        i += 1
    for grp, rr in rows.items():
        if not rr: continue
        by = profile(rr)
        line = f"FUNNEL {name}-{grp}:"
        for b, (lo, hi) in enumerate(BINS):
            if by[b]:
                line += f" {int(lo*1000)}-{int(hi*1000)}mm={np.median(by[b]):.1f}(n={len(by[b])})"
        print(line, flush=True)
print("FUNNEL-DONE")
