"""Episode-level adjudication: orientation as CAUSE vs OBSERVATION.
For every failing episode: closest 3D approach to the gate and orientation error
AT that moment. If failures include well-oriented (<15deg, within demo funnel
tolerance p90=21deg) episodes that still stall >15mm out, orientation is not the gate."""
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
FQ = slice(17, 21); BP = slice(7, 10); FP = slice(21, 24)
def qn(q): return q / (np.linalg.norm(q) + 1e-9)
def qangle(q1, q2): return np.degrees(2*np.arccos(np.clip(abs(float(np.dot(q1, q2))), -1, 1)))
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
dq, offs = [], []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP]); dq.append(qn(ov[r1-20, FQ]))
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
    i = 0; rows = []
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        if not int(m[1]):
            g0, gend = held_range(o, a)
            if g0 is not None:
                d3 = np.linalg.norm(o[g0:gend, FP] - (o[g0:gend, BP] + off), axis=1)
                tmin = int(np.argmin(d3))
                # median orientation over the 30 held steps around closest approach
                lo2, hi2 = max(0, tmin - 15), min(gend - g0, tmin + 15)
                oe = np.median([err(o[g0 + t, FQ]) for t in range(lo2, hi2)])
                rows.append((float(d3[tmin]) * 1000, float(oe)))
        i += 1
    rows.sort()
    ss = " | ".join(f"{d:.0f}mm/{e:.0f}d" for d, e in rows)
    ngood = sum(1 for d, e in rows if e < 15); nclose = sum(1 for d, e in rows if d < 15)
    print(f"CAUSE {name}-FAIL (n={len(rows)}): closest3D/orient@closest: {ss}")
    print(f"CAUSE {name}-FAIL: well-oriented(<15deg) failures = {ngood}/{len(rows)} | reached<15mm = {nclose}/{len(rows)}", flush=True)
print("CAUSE-DONE")
