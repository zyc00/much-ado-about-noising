"""Sub-centimeter gate-approach analysis: what happens at the insertion gate?

Target estimation: mean over GT demos of frame_pos(r1) relative to base_pos -> per-episode
target = base_pos + offset. Per rollout episode: closest approach ||frame_pos - target||,
miss decomposition (lateral vs vertical) at closest approach, number of approach attempts
(entries into 3cm followed by retreat to >5cm), and time spent within 2cm.
Cells: chiunet pairs s5 + s5001. MSE-blocked (S4) vs MIP episodes.
"""
import os
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)

# estimate gate offset from demos
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs = []
for k in keys[:60]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1 - 1, FP] - ov[r1 - 1, BP])
h.close()
off = np.median(np.stack(offs), 0)
print(f"gate offset (frame rel base at release): {np.round(off,4)} (n={len(offs)}, demo scatter p90={np.quantile(np.linalg.norm(np.stack(offs)-off,axis=1),0.9)*1000:.1f}mm)", flush=True)

for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    f = f"analysis/traj_vis/human_{name}.npz"
    if not os.path.exists(f): continue
    z = np.load(f)
    rows = []
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; m = z[f"ep{i}_meta"]
        target = o[0, BP] + off
        d = np.linalg.norm(o[:, FP] - target[None], axis=1)
        dl = np.linalg.norm((o[:, FP] - target[None])[:, :2], axis=1)
        dv = np.abs((o[:, FP] - target[None])[:, 2])
        tmin = int(np.argmin(d))
        # approach attempts: entries below 3cm followed by retreat above 5cm
        att, inside = 0, False
        for t in range(len(d)):
            if not inside and d[t] < 0.03: inside = True; att += 1
            elif inside and d[t] > 0.05: inside = False
        rows.append(dict(succ=int(m[0]), asm=int(m[1]), dmin=d.min(),
                         lat=dl[tmin], vert=dv[tmin], att=att,
                         t2cm=(d < 0.02).sum()))
        i += 1
    for grp, sel in [("asm-fail", [r for r in rows if not r["asm"]]),
                     ("asm-pass", [r for r in rows if r["asm"]])]:
        if not sel: print(f"GATE {name} {grp}: none", flush=True); continue
        dmin = np.array([r["dmin"] for r in sel]) * 1000
        lat = np.array([r["lat"] for r in sel]) * 1000
        vert = np.array([r["vert"] for r in sel]) * 1000
        att = np.array([r["att"] for r in sel])
        t2 = np.array([r["t2cm"] for r in sel])
        print(f"GATE {name} {grp} (n={len(sel)}): closest-approach p50={np.median(dmin):.1f}mm p90={np.quantile(dmin,0.9):.1f}mm | miss lat/vert p50={np.median(lat):.1f}/{np.median(vert):.1f}mm | attempts p50={np.median(att):.0f} max={att.max()} | steps<2cm p50={np.median(t2):.0f}", flush=True)
print("GATE-DONE")
