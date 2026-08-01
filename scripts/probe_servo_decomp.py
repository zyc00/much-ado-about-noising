"""Why does lateral centering saturate? At matched approach states (held, alt>0,
lat banded), decompose executed commands:
  cmd_lat  = a_xy . unit(gate_xy - frame_xy)   (>0 = centering)
  cmd_z    = a_z                                (<0 = descending toward gate)
  realized dlat/step (mm), and per-band flip rate of sign(cmd_lat).
Discriminates: wrong direction (cmd_lat<=0, cmd_z<0 diagonal-through-pin) vs
attenuated/dither (cmd_lat ~0, flips) vs blocked (cmd_lat>>0 but dlat ~0).
Also approach shape: median alt per lat band."""
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
BANDS = [(0.010, 0.020), (0.020, 0.030), (0.030, 0.045), (0.045, 0.060)]
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

def collect(ov, act, a0, a1):
    """per-state rows: band, cmd_lat, cmd_z, dlat_mm, alt_mm, sign(cmd_lat)"""
    v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
    lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
    out = []
    for t in range(len(v) - 1):
        if alt[t] <= 0: continue
        for b, (lo, hi) in enumerate(BANDS):
            if lo <= lat[t] < hi:
                u = -v[t, :2] / (lat[t] + 1e-9)
                axy = act[a0 + t, 0:2]; az = float(act[a0 + t, 2])
                cl_ = float(np.dot(axy, u))
                dlat = (lat[t + 1] - lat[t]) * 1000
                out.append((b, cl_, az, dlat, alt[t] * 1000))
                break
    return out

def report(tag, rows):
    if not rows: return
    R = np.array(rows)
    for b in range(len(BANDS)):
        r = R[R[:, 0] == b]
        if len(r) < 12: continue
        lo, hi = BANDS[b]
        flips = np.mean(np.sign(r[1:, 1]) != np.sign(r[:-1, 1])) if len(r) > 1 else np.nan
        print(f"SERVO {tag} lat{int(lo*1000)}-{int(hi*1000)}mm (n={len(r)}): cmd_lat p50={np.median(r[:,1]):+.3f} frac>0={np.mean(r[:,1]>0):.2f} | cmd_z p50={np.median(r[:,2]):+.3f} | dlat p50={np.median(r[:,3]):+.2f}mm/step | alt p50={np.median(r[:,4]):.0f}mm | flip={flips:.2f}", flush=True)

report("demos", [x for ov, act, a0, a1 in drows for x in collect(ov, act, a0, a1)])
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0; grp = {"PASS": [], "FAIL": []}
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            grp["PASS" if int(m[1]) else "FAIL"].extend(collect(o, a, g0, gend))
        i += 1
    for gname, rows in grp.items(): report(f"{name}-{gname}", rows)
print("SERVO-DONE")
