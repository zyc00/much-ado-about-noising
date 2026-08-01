"""Commit-vs-retreat branch analysis at the engaged pocket (alt 25-60mm, lat 10-45mm).
1) In DEMOS: label each pocket state by realized branch (alt(t+15)-alt(t) < -10mm = COMMIT,
   > +10mm = RETREAT, else drop). Compare feature distributions per branch:
   lat, in-hand orientation error, transverse speed. => the demonstrated commit precondition.
2) In FAILING runs: label pocket states the same way; report branch mix and the fraction of
   commit-eligible states (inside demo commit p10-p90 box on lat AND orient) whose realized
   branch was retreat => wrongly-retreated states."""
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
FQ = slice(17, 21); BP = slice(7, 10); FP = slice(21, 24)
LK = 15
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
def pocket_rows(ov, a0, a1):
    v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
    lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
    out = []
    for t in range(len(v) - LK):
        if not (0.025 <= alt[t] < 0.060 and 0.010 <= lat[t] < 0.045): continue
        da = (alt[t + LK] - alt[t]) * 1000
        br = "C" if da < -10 else ("R" if da > 10 else None)
        if br is None: continue
        sp = np.mean(np.linalg.norm(np.diff(ov[a0+max(0,t-5):a0+t+1, FP][:, :2], axis=0), axis=1)) * 1000 if t >= 2 else np.nan
        out.append((br, lat[t] * 1000, err(ov[a0 + t, FQ]), sp))
    return out
demo_rows = [x for ov, a0, a1 in drows for x in pocket_rows(ov, a0, a1)]
for br, tag in [("C", "COMMIT"), ("R", "RETREAT")]:
    r = [x for x in demo_rows if x[0] == br]
    L = np.array([x[1] for x in r]); O = np.array([x[2] for x in r]); S = np.array([x[3] for x in r])
    print(f"BRANCH demos {tag} (n={len(r)}): lat p50={np.median(L):.0f} [{np.quantile(L,0.1):.0f},{np.quantile(L,0.9):.0f}]mm | orient p50={np.median(O):.1f} [{np.quantile(O,0.1):.1f},{np.quantile(O,0.9):.1f}]deg | speed p50={np.nanmedian(S):.2f}mm/st", flush=True)
C = [x for x in demo_rows if x[0] == "C"]
lat_hi = np.quantile([x[1] for x in C], 0.9); or_hi = np.quantile([x[2] for x in C], 0.9)
print(f"BRANCH demo commit box: lat<={lat_hi:.0f}mm AND orient<={or_hi:.1f}deg (p90s)", flush=True)
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
    i = 0; grp = {"PASS": [], "FAIL": []}
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            grp["PASS" if int(m[1]) else "FAIL"].extend(pocket_rows(o, g0, gend))
        i += 1
    for gname, rows in grp.items():
        if not rows: continue
        nC = sum(1 for x in rows if x[0] == "C"); nR = len(rows) - nC
        elig = [x for x in rows if x[1] <= lat_hi and x[2] <= or_hi]
        eR = sum(1 for x in elig if x[0] == "R")
        O = np.array([x[2] for x in rows])
        print(f"BRANCH {name}-{gname}: pocket n={len(rows)} commit={nC} retreat={nR} | orient@pocket p50={np.median(O):.1f}deg | commit-eligible={len(elig)} of which RETREATED={eR}", flush=True)
print("BRANCH-DONE")
