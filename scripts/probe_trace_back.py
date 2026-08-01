"""Trace-back probe: where does the failing approach diverge?
Gate frame: v(t) = frame_pos - (base_pos + off), off = median demo (frame-base) at release.
lat = ||v_xy||, alt = v_z. Demonstrated order: center at altitude, then descend.
Per episode:
  bestcenter_alt = min lat while alt > 20mm  (did it ever center at altitude?)
  lat@descend    = lat at first alt < 20mm   (was it centered when it descended?)
  postdrift      = max lat after descend - lat@descend (did it drift after descending?)
  neverdesc      = alt never < 20mm
  closest3D
Groups: demos, PASS/FAIL per model."""
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
ALT_TH = 0.020
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

def trace(ov, a0, a1):
    v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
    lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]; d3 = np.linalg.norm(v, axis=1)
    up = alt > ALT_TH
    bc = float(np.min(lat[up])) if up.any() else np.nan
    below = np.where(~up)[0]
    if len(below):
        td = below[0]
        latd = float(lat[td]); post = float(np.max(lat[td:]) - lat[td])
    else:
        latd = np.nan; post = np.nan
    return dict(bc=bc, latd=latd, post=post, never=not len(below), c3=float(np.min(d3)))

def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend

def report(tag, rows):
    if not rows: return
    def q(key):
        x = np.array([r[key] for r in rows if not np.isnan(r[key])])
        return f"{np.median(x)*1000:.0f}[{np.quantile(x,0.1)*1000:.0f},{np.quantile(x,0.9)*1000:.0f}]" if len(x) else "-"
    nev = sum(r["never"] for r in rows)
    print(f"TRACE {tag} (n={len(rows)}): bestcenter@alt={q('bc')}mm | lat@descend={q('latd')}mm | postdrift={q('post')}mm | neverdesc={nev} | closest3D={q('c3')}mm", flush=True)

report("demos", [trace(ov, a0, a1) for ov, a0, a1 in drows])
import os as _o
for name in _o.environ.get("NAMES", "hMSE_s5 hMIP_s5 hMSE_s5001 hMIP_s5001").split():
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0; grp = {"PASS": [], "FAIL": []}
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            grp["PASS" if int(m[1]) else "FAIL"].append(trace(o, g0, gend))
        i += 1
    for gname, rows in grp.items(): report(f"{name}-{gname}", rows)
print("TRACE-DONE")
