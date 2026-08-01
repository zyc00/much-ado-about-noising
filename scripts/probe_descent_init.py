"""Descent-initiation profile (user observation: 'inserting before aligning'):
every downward crossing of alt=25mm (held, lat<80mm): record (lat, orient) at crossing.
Demos vs each policy's PASS/FAIL. Premature insertion = crossings at high lat / high orient."""
import numpy as np, h5py, os
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24); FQ = slice(17, 21)
def qn(q): return q / (np.linalg.norm(q) + 1e-9)
def qangle(q1, q2): return np.degrees(2*np.arccos(np.clip(abs(float(np.dot(q1, q2))), -1, 1)))
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, dq, drows = [], [], []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    offs.append(ov[r1-1, FP] - ov[r1-1, BP]); dq.append(qn(ov[r1-20, FQ]))
    drows.append((ov, c1 + 5, r1))
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
def crossings(ov, a0, a1):
    v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
    lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
    out = []
    for t in range(1, len(v)):
        if alt[t-1] >= 0.025 > alt[t] and lat[t] < 0.080:
            out.append((lat[t] * 1000, err(ov[a0 + t, FQ])))
    return out
def rep(tag, cs):
    if not cs: print(f"DESCINIT {tag}: none"); return
    L = np.array([c[0] for c in cs]); O = np.array([c[1] for c in cs])
    mis = np.mean((L > 20) | (O > 21))
    print(f"DESCINIT {tag} (n={len(cs)}): lat p50={np.median(L):.0f} p90={np.quantile(L,0.9):.0f}mm | orient p50={np.median(O):.1f} p90={np.quantile(O,0.9):.1f}deg | frac MISALIGNED (lat>20mm or orient>21deg)={mis:.2f}", flush=True)
rep("demos", [c for ov, a0, a1 in drows for c in crossings(ov, a0, a1)])
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
for name in os.environ.get("NAMES", "hheterot_s5 hheterot_s1000 hmipL286k").split():
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0; grp = {"PASS": [], "FAIL": []}
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            grp["PASS" if int(m[0]) else "FAIL"].extend(crossings(o, g0, gend))
        i += 1
    for gn, cs in grp.items(): rep(f"{name}-{gn}", cs)
print("DESCINIT-DONE")
