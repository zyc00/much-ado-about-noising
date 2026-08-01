"""Spiral anatomy of near-gate motion. Per group:
 (a) signed angular velocity dtheta (deg/step) around the pin axis (consistent circulation?)
 (b) azimuth occupancy: histogram of atan2(v_y,v_x) in 8 sectors — do fails hover at
     azimuths demos rarely visit (coverage hole)?
 (c) speed |d frame_xy|/step distribution — hover test."""
import numpy as np, h5py, os
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, drows = [], []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP]); drows.append((ov, a, c1 + 5, r1))
h.close()
off = np.median(np.stack(offs), 0)
def stats(rows):
    dth, spd, az = [], [], []
    for ov, act, a0, a1 in rows:
        v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
        lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
        th = np.arctan2(v[:, 1], v[:, 0])
        near = (lat >= 0.010) & (lat < 0.060) & (alt >= 0.005) & (alt < 0.120)
        for t in range(len(v) - 1):
            if not (near[t] and near[t + 1]): continue
            d = np.degrees(np.arctan2(np.sin(th[t+1]-th[t]), np.cos(th[t+1]-th[t])))
            dth.append(d)
            spd.append(np.linalg.norm(v[t+1, :2] - v[t, :2]) * 1000)
            az.append(int(((th[t] + np.pi) / (2*np.pi) * 8)) % 8)
    dth = np.array(dth); az = np.array(az)
    hist = np.bincount(az, minlength=8) / max(len(az), 1)
    hs = "/".join(f"{100*x:.0f}" for x in hist)
    return (f"dtheta p50={np.median(dth):+.2f}deg/st frac|>0|={np.mean(dth>0):.2f}"
            f" | speed p50={np.median(spd):.2f} p10={np.quantile(spd,0.1):.2f}mm/st"
            f" | azimuth% {hs} (n={len(dth)})")
print("SPIRAL demos:", stats(drows), flush=True)
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
for name in os.environ.get("NAMES", "hMSE_s5 hMIP_s5 hrotaux_s5 hrotw_s5").split():
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0; grp = {"PASS": [], "FAIL": []}
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            grp["PASS" if int(m[1]) else "FAIL"].append((o, a, g0, gend))
        i += 1
    for gn, rows in grp.items():
        if rows: print(f"SPIRAL {name}-{gn}:", stats(rows), flush=True)
print("SPIRAL-DONE")
