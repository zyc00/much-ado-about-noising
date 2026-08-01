"""WHERE is the ~20deg in-hand orientation error manufactured?
Track orientation error (frame relq angle to the demo pre-gate median) at checkpoints:
grasp+15, grasp+60, ring entry (<8cm lateral), pre-gate ring (final held window).
Groups: demos / each model x {pass, fail}. Outcomes:
(a) error ~20deg already at grasp+15 -> manufactured AT THE GRASP
(b) grows en route -> transport drift
(c) demos start high and converge while fails start high and stay -> missing REORIENTATION
"""
import os
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
FQ = slice(17, 21); BP = slice(7, 10); FP = slice(21, 24)

def qangle(q1, q2):
    return np.degrees(2 * np.arccos(np.clip(abs(float(np.dot(q1, q2))), -1, 1)))
def qn(q):
    return q / (np.linalg.norm(q) + 1e-9)

# demo reference qmu (pre-gate) + gate offset
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
    dq.append(qn(ov[(r1-20), FQ]))
    demos.append((ov, c1, r1))
h.close()
off = np.median(np.stack(offs), 0)
dq = np.stack(dq); qref = dq[0]
for i in range(1, len(dq)):
    if np.dot(dq[i], qref) < 0: dq[i] = -dq[i]
qmu = qn(dq.mean(0))

def err(q):
    q = qn(q)
    return qangle(q, qmu)

def checkpoints_demo(ov, c1, r1):
    ring = next((t for t in range(c1+5, r1) if np.linalg.norm((ov[t, FP]-(ov[t, BP]+off))[:2]) < 0.08), None)
    pts = {}
    pts["grasp+15"] = err(ov[min(c1+15, r1-1), FQ])
    pts["grasp+60"] = err(ov[min(c1+60, r1-1), FQ])
    if ring: pts["ring-entry"] = err(ov[ring, FQ])
    pts["pre-gate"] = err(np.median(ov[r1-25:r1-5, FQ], axis=0))
    return pts

groups = {"DEMOS": [checkpoints_demo(ov, c1, r1) for ov, c1, r1 in demos]}
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
        ring = None
        run = 0
        for t in range(g0, L):
            run = run + 1 if gc[t] >= 0 else 0
            d = o[t, FP] - (o[t, BP] + off)
            if run >= 1 and np.linalg.norm(d[:2]) < 0.08 and abs(d[2]) < 0.05:
                ring = t; break
        ringstates = [t for t in range(g0, L) if gc[t] >= 0 and np.linalg.norm((o[t, FP]-(o[t, BP]+off))[:2]) < 0.08 and abs((o[t, FP]-(o[t, BP]+off))[2]) < 0.05]
        pts = {}
        pts["grasp+15"] = err(o[min(g0+15, L-1), FQ])
        pts["grasp+60"] = err(o[min(g0+60, L-1), FQ])
        if ring: pts["ring-entry"] = err(o[ring, FQ])
        if len(ringstates) > 10:
            pts["pre-gate"] = err(np.median(o[ringstates[-30:], FQ], axis=0))
        key = f"{name}-{'PASS' if int(m[1]) else 'FAIL'}"
        groups.setdefault(key, []).append(pts)
        i += 1

CKS = ["grasp+15", "grasp+60", "ring-entry", "pre-gate"]
for gname, rows in groups.items():
    line = f"OTRAJ {gname} (n={len(rows)}):"
    for ck in CKS:
        v = [r[ck] for r in rows if ck in r]
        line += f" {ck}={np.median(v):.1f}deg" if v else f" {ck}=--"
    print(line, flush=True)
print("OTRAJ-DONE")
