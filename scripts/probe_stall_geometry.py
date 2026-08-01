"""Where do failing episodes stall, in gate-frame coordinates? Compare demo
mid-approach corridor (states at 15-40mm 3D, held) vs failing-episode closest-approach
positions. Axes: gate-frame offset vec = frame_pos - (base_pos + off); report signed
components (x,y = lateral, z = vertical) to see if stalls cluster directionally."""
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
    offs.append(ov[r1-1, FP] - ov[r1-1, BP]); drows.append((ov, c1 + 5, r1))
h.close()
off = np.median(np.stack(offs), 0)
def vecs(rows, lo=0.015, hi=0.040):
    out = []
    for ov, a0, a1 in rows:
        for t in range(a0, a1):
            v = ov[t, FP] - (ov[t, BP] + off)
            if lo <= np.linalg.norm(v) < hi: out.append(v)
    return np.array(out)
dv = vecs(drows)
print(f"GEO demos corridor 15-40mm (n={len(dv)}): x p50={np.median(dv[:,0])*1000:+.0f}mm [{np.quantile(dv[:,0],0.1)*1000:+.0f},{np.quantile(dv[:,0],0.9)*1000:+.0f}] | y p50={np.median(dv[:,1])*1000:+.0f} [{np.quantile(dv[:,1],0.1)*1000:+.0f},{np.quantile(dv[:,1],0.9)*1000:+.0f}] | z p50={np.median(dv[:,2])*1000:+.0f} [{np.quantile(dv[:,2],0.1)*1000:+.0f},{np.quantile(dv[:,2],0.9)*1000:+.0f}]", flush=True)
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
    i = 0; pts = []
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        if not int(m[1]):
            g0, gend = held_range(o, a)
            if g0 is not None:
                d3 = np.linalg.norm(o[g0:gend, FP] - (o[g0:gend, BP] + off), axis=1)
                tmin = int(np.argmin(d3)) + g0
                pts.append(o[tmin, FP] - (o[tmin, BP] + off))
        i += 1
    p = np.array(pts)
    if not len(p): continue
    comp = " ".join(f"({v[0]*1000:+.0f},{v[1]*1000:+.0f},{v[2]*1000:+.0f})" for v in p)
    print(f"GEO {name}-FAIL stall offsets (x,y,z mm): {comp}", flush=True)
print("GEO-DONE")
