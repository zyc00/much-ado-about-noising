"""In-hand ORIENTATION: frame rel-quat (obs 17-20) angle to the demo median during the
pre-gate held segment (before first release/assembly), fail vs pass vs demos. Also eef quat
(47-50) deviation. A rotated grip blocks insertion at position-perfect states."""
import os
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
FQ = slice(17, 21); EQ = slice(47, 51); BP = slice(7, 10); FP = slice(21, 24)

def qangle(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return np.degrees(2 * np.arccos(np.clip(d, -1, 1)))

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
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    seg = ov[r1-40:r1-5, FQ]
    q = seg[len(seg)//2]; q = q / (np.linalg.norm(q) + 1e-9)
    dq.append(q)
h.close()
off = np.median(np.stack(offs), 0)
dq = np.stack(dq)
qref = dq[0]
for i in range(1, len(dq)):
    if np.dot(dq[i], qref) < 0: dq[i] = -dq[i]
qmu = dq.mean(0); qmu /= np.linalg.norm(qmu)
scat = [qangle(q, qmu) for q in dq]
print(f"DEMO in-hand orientation scatter (pre-gate): p50={np.median(scat):.1f}deg p90={np.quantile(scat,0.9):.1f}deg", flush=True)

for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    ang_pass, ang_fail = [], []
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        L = min(len(o), len(a)); gc = a[:L, 6]
        # pre-gate held ring states: held + frame within 8cm lateral of gate, pre-assembly window
        run = 0; qs = []
        for t in range(L):
            run = run + 1 if gc[t] >= 0 else 0
            if run < 15: continue
            d = o[t, FP] - (o[t, BP] + off)
            if np.linalg.norm(d[:2]) < 0.08 and abs(d[2]) < 0.05:
                q = o[t, FQ]; q = q / (np.linalg.norm(q) + 1e-9)
                if np.dot(q, qmu) < 0: q = -q
                qs.append(q)
        if len(qs) > 10:
            qmid = np.stack(qs).mean(0); qmid /= np.linalg.norm(qmid)
            (ang_pass if int(m[1]) else ang_fail).append(qangle(qmid, qmu))
        i += 1
    def s(x): return f"p50={np.median(x):.1f}deg p90={np.quantile(x,0.9):.1f}deg (n={len(x)})" if x else "none"
    print(f"ORIENT {name}: PASS {s(ang_pass)} | FAIL {s(ang_fail)}", flush=True)
print("ORIENT-DONE")
