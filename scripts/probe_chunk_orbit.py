"""Where does lateral cancellation happen — within chunks or between chunks?
Control executes 8-step chunks (act_steps=8, boundaries at step%8 from control start).
Over near-gate held states, per chunk c: net_c = sum(dlat), path_c = sum(|dlat|).
  R_within = sum|net_c| / sum path_c   (1 = each chunk moves coherently)
  R_across = |sum net_c| / sum|net_c|  (1 = chunks agree on direction)
  total R = R_within * R_across (approx; both reported)
Also: cos between consecutive chunks' mean xy-displacement (direction flip test).
Demos: 8-step pseudo-chunks for reference."""
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
def stats(rows, phase0=0):
    nets, paths, dirs = [], [], []
    coss = []
    for ov, act, a0, a1 in rows:
        v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
        lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
        near = (lat >= 0.010) & (lat < 0.060) & (alt >= 0.005) & (alt < 0.120)
        # chunk index in ABSOLUTE episode time so boundaries match control
        prev_dir = None
        t = 0
        while t < len(v) - 1:
            cstart = t - ((a0 + t - phase0) % 8)  # align to chunk boundary
            cend = min(cstart + 8, len(v) - 1)
            if cstart < 0: t = cend; continue
            seg = [s for s in range(max(cstart, 0), cend) if near[s]]
            if len(seg) >= 6:
                dl = np.diff(lat[max(cstart,0):cend + 1]) * 1000
                dxy = (v[cend, :2] - v[max(cstart,0), :2])
                nets.append(dl.sum()); paths.append(np.abs(dl).sum())
                n = np.linalg.norm(dxy)
                if n > 1e-6:
                    d = dxy / n
                    if prev_dir is not None: coss.append(float(np.dot(d, prev_dir)))
                    prev_dir = d
                else: prev_dir = None
            else: prev_dir = None
            t = cend
    nets = np.array(nets); paths = np.array(paths)
    Rw = np.abs(nets).sum() / (paths.sum() + 1e-9)
    Ra = -nets.sum() / (np.abs(nets).sum() + 1e-9)
    return (f"chunks={len(nets)} | R_within={Rw:.2f} | R_across={Ra:+.2f}"
            f" | consec-chunk dir cos p50={np.median(coss) if coss else np.nan:+.2f}"
            f" | net/chunk p50={np.median(nets):+.2f}mm")
print("CHUNK demos(pseudo):", stats(drows), flush=True)
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
        if rows: print(f"CHUNK {name}-{gn}:", stats(rows), flush=True)
print("CHUNK-DONE")
