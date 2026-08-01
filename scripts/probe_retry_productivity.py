"""Are retries productive? (1) Successive pocket-entry orientation errors per episode:
delta between consecutive entries (demos should improve, fails flat). (2) During retreat
bouts: rotation-command magnitude |act[3:6]| and realized dorient/step."""
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
FQ = slice(17, 21); BP = slice(7, 10); FP = slice(21, 24)
def qn(q): return q / (np.linalg.norm(q) + 1e-9)
def qangle(q1, q2): return np.degrees(2*np.arccos(np.clip(abs(float(np.dot(q1, q2))), -1, 1)))
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, dq, drows = [], [], []
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
    drows.append((ov, a, c1 + 5, r1))
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
def analyze(ov, act, a0, a1):
    v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
    lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
    inpocket = (alt >= 0.025) & (alt < 0.060) & (lat >= 0.010) & (lat < 0.045)
    # pocket entries (gap>5 separates visits)
    entries = []
    last = -99
    for t in range(len(v)):
        if inpocket[t]:
            if t - last > 5: entries.append(t)
            last = t
    ent_err = [err(ov[a0 + t, FQ]) for t in entries]
    deltas = list(np.diff(ent_err)) if len(ent_err) >= 2 else []
    # retreat bouts: rising-alt runs in the near zone (lat<80mm), >=8 steps
    rot, dor = [], []
    t = 0
    while t < len(v) - 8:
        if lat[t] < 0.080 and alt[t + 8] - alt[t] > 0.010 and alt[t] > 0.005:
            e = t
            while e < len(v) - 1 and alt[e + 1] >= alt[e] - 0.002 and lat[e] < 0.10: e += 1
            if e - t >= 8:
                rot.extend(np.linalg.norm(act[a0 + t:a0 + e, 3:6], axis=1))
                dor.append((err(ov[a0 + e, FQ]) - err(ov[a0 + t, FQ])) / (e - t))
            t = e
        t += 1
    return ent_err, deltas, rot, dor
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
def report(tag, packs):
    ne = [len(p[0]) for p in packs]
    dl = [d for p in packs for d in p[1]]
    rot = [r for p in packs for r in p[2]]
    dor = [d for p in packs for d in p[3]]
    first = [p[0][0] for p in packs if p[0]]
    lastv = [p[0][-1] for p in packs if p[0]]
    s = f"RETRY {tag} (eps={len(packs)}): entries/ep p50={np.median(ne):.0f} | first-entry orient p50={np.median(first):.1f} -> last-entry {np.median(lastv):.1f}deg"
    if dl: s += f" | delta-per-retry p50={np.median(dl):+.1f}deg (n={len(dl)})"
    if rot: s += f" | |rotcmd|@retreat p50={np.median(rot):.3f} (n={len(rot)})"
    if dor: s += f" | dorient@retreat p50={np.median(dor)*10:+.1f}deg/10st (bouts={len(dor)})"
    print(s, flush=True)
report("demos", [analyze(ov, act, a0, a1) for ov, act, a0, a1 in drows])
import os as _o
for name in _o.environ.get("NAMES", "hMSE_s5 hMIP_s5 hMSE_s5001 hMIP_s5001").split():
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0; grp = {"PASS": [], "FAIL": []}
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            grp["PASS" if int(m[1]) else "FAIL"].append(analyze(o, a, g0, gend))
        i += 1
    for gname, packs in grp.items():
        if packs: report(f"{name}-{gname}", packs)
print("RETRY-DONE")
