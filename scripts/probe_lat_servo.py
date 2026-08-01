"""Lateral-channel battery (the lateral analog of retry productivity).
Near states = held, lat in [10,60)mm, alt in [5,120)mm. Per group:
  |a_xy| p50            — lateral command magnitude
  cmd_lat p50, frac>0   — centering projection (signed)
  closure R             — -(sum dlat)/(sum |dlat|) over near states (+1 = every mm of
                          lateral motion closes; 0 = pure cancellation/orbit)
  visit min-lat p50     — best centering per pocket visit
  frac visits <10mm     — visits that complete centering
Groups: demos, per-model PASS/FAIL (+ hrotaux)."""
import numpy as np, h5py
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
    axy, cl_, dl = [], [], []
    vmin, vhit = [], 0
    for ov, act, a0, a1 in rows:
        v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
        lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
        near = (lat >= 0.010) & (lat < 0.060) & (alt >= 0.005) & (alt < 0.120)
        for t in range(len(v) - 1):
            if not near[t]: continue
            u = -v[t, :2] / (lat[t] + 1e-9)
            axy.append(float(np.linalg.norm(act[a0 + t, 0:2])))
            cl_.append(float(np.dot(act[a0 + t, 0:2], u)))
            dl.append((lat[t + 1] - lat[t]) * 1000)
        # pocket visits (alt 25-60, lat 10-45), gap>5
        inp = (alt >= 0.025) & (alt < 0.060) & (lat >= 0.010) & (lat < 0.045)
        t = 0
        while t < len(v):
            if inp[t]:
                e = t
                while e < len(v) - 1 and (inp[e + 1] or (e + 1 - t < 5)): e += 1
                vmin.append(float(np.min(lat[t:e + 1])) * 1000)
                t = e + 6
            else: t += 1
    dl = np.array(dl)
    R = -dl.sum() / (np.abs(dl).sum() + 1e-9)
    vmin = np.array(vmin)
    hit = float(np.mean(vmin < 10)) if len(vmin) else np.nan
    return (f"|a_xy|={np.median(axy):.3f} | cmd_lat={np.median(cl_):+.3f} frac>0={np.mean(np.array(cl_)>0):.2f}"
            f" | closure R={R:+.2f} | visit-minlat p50={np.median(vmin) if len(vmin) else -1:.0f}mm"
            f" | visits<10mm={100*hit:.0f}% (nvis={len(vmin)})")
print("LATSERVO demos:", stats(drows), flush=True)
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
import os
for name in os.environ.get("NAMES", "hMSE_s5 hMIP_s5 hMSE_s5001 hMIP_s5001 hrotaux_s5").split():
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0; grp = {"PASS": [], "FAIL": []}
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            grp["PASS" if int(m[1]) else "FAIL"].append((o, a, g0, gend))
        i += 1
    for gn, rows in grp.items():
        if rows: print(f"LATSERVO {name}-{gn}:", stats(rows), flush=True)
print("LATSERVO-DONE")
