"""2D pocket analysis: commands conditioned on (lat, alt) jointly.
Pockets: lat {10-20,20-30,30-45}mm x alt {5-25, 25-60, 60-120}mm.
Per group: occupancy share, cmd_lat, cmd_z, realized dlat. Tests whether failure is
(a) never entering the descend pocket (occupancy) or (b) wrong field at matched pocket."""
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
LB = [(0.010, 0.020), (0.020, 0.030), (0.030, 0.045)]
AB = [(0.005, 0.025), (0.025, 0.060), (0.060, 0.120)]
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
    v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
    lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
    out = []
    for t in range(len(v) - 1):
        li = next((i for i, (lo, hi) in enumerate(LB) if lo <= lat[t] < hi), None)
        ai = next((i for i, (lo, hi) in enumerate(AB) if lo <= alt[t] < hi), None)
        if li is None or ai is None: continue
        u = -v[t, :2] / (lat[t] + 1e-9)
        out.append((li, ai, float(np.dot(act[a0 + t, 0:2], u)), float(act[a0 + t, 2]),
                    (lat[t + 1] - lat[t]) * 1000))
    return out
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
groups = {"demos": [x for ov, act, a0, a1 in drows for x in collect(ov, act, a0, a1)]}
for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            key = f"{name}-{'PASS' if int(m[1]) else 'FAIL'}"
            groups.setdefault(key, []).extend(collect(o, a, g0, gend))
        i += 1
# pool passes and fails across models for power; also print per-model fail rows
groups["ALL-PASS"] = sum((v for k, v in groups.items() if k.endswith("PASS")), [])
groups["ALL-FAIL"] = sum((v for k, v in groups.items() if k.endswith("FAIL")), [])
for tag in ["demos", "ALL-PASS", "ALL-FAIL", "hMSE_s5-FAIL", "hMSE_s5001-FAIL"]:
    R = np.array(groups[tag]); tot = len(R)
    print(f"--- POCKET {tag} (N={tot})")
    for ai in range(len(AB)):
        row = f"  alt{int(AB[ai][0]*1000)}-{int(AB[ai][1]*1000)}mm:"
        for li in range(len(LB)):
            r = R[(R[:, 0] == li) & (R[:, 1] == ai)]
            if len(r) < 10:
                row += f"  lat{int(LB[li][0]*1000)}-{int(LB[li][1]*1000)}: n={len(r)}<10"
                continue
            row += (f"  lat{int(LB[li][0]*1000)}-{int(LB[li][1]*1000)}: n={len(r)}({100*len(r)/tot:.0f}%)"
                    f" cl={np.median(r[:,2]):+.3f} cz={np.median(r[:,3]):+.3f} dlat={np.median(r[:,4]):+.2f}")
        print(row, flush=True)
print("POCKET-DONE")
