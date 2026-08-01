"""Stage-2 (tool-hang) funnel: for episodes that assembled the frame, where does the
second leg die? Events: r1 (frame release), c2 (tool grasp), lift, approach, hang.
Stages: S2 grasp2 | S3 lifted (+20mm) | S4 approach (<80mm of hang target) | S5 precision
(closest 3D to target) | success."""
import numpy as np, h5py, os
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
FP = slice(21, 24); TP = slice(35, 38)
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
tgts = []
drows = []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if len(cl) < 2 or len(op) < 2: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if r1 is None: continue
    c2 = next((t for t in cl if t > r1 + 5), None)
    r2 = next((t for t in op if t > (c2 or 1e9)), None)
    if c2 is None or r2 is None: continue
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    tgts.append(ov[r2-1, TP] - ov[r2-1, FP])
    drows.append((ov, np.asarray(a), c2, r2))
h.close()
tgt = np.median(np.stack(tgts), 0)
print(f"hang target offset (tool - frame): {np.round(tgt,3)} | demos with full stage2: {len(drows)}")
def stage2(ov, act, T):
    g = act[:T, 6]
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: return dict(g2=False)
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if r1 is None: return dict(g2=False)
    c2 = next((t for t in cl if t > r1 + 5), None)
    if c2 is None or c2 >= T - 10: return dict(g2=False)
    T = min(T, len(ov), len(act))
    if c2 >= T - 10: return dict(g2=False)
    tz = ov[c2:T, TP.start + 2]
    lifted = bool((tz - tz[0] > 0.02).any())
    d3 = np.linalg.norm(ov[c2:T, TP] - (ov[c2:T, FP] + tgt), axis=1)
    return dict(g2=True, lifted=lifted, approach=bool((d3 < 0.080).any()), closest=float(d3.min() * 1000))
def rep(tag, rows):
    n = len(rows)
    g2 = [r for r in rows if r["g2"]]
    lif = [r for r in g2 if r["lifted"]]; app = [r for r in lif if r["approach"]]
    cl = [r["closest"] for r in app]
    print(f"STAGE2 {tag}: n={n} grasp2={len(g2)} lifted={len(lif)} approach<80mm={len(app)}"
          f" | closest p50={np.median(cl) if cl else -1:.0f}mm p90={np.quantile(cl,0.9) if cl else -1:.0f}mm", flush=True)
def held_analysis(name):
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0
    groups = {"ASSEMBLED-FAIL": [], "SUCCESS": []}
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        T = min(len(o), len(a))
        if int(m[0]): groups["SUCCESS"].append(stage2(o, a, T))
        elif int(m[1]): groups["ASSEMBLED-FAIL"].append(stage2(o, a, T))
        i += 1
    for gn, rows in groups.items():
        if rows: rep(f"{name}-{gn}", rows)
rep("demos", [stage2(ov, act, min(r2 + 10, min(len(ov), len(act)))) for ov, act, c2, r2 in drows])
for name in os.environ.get("NAMES", "hheterot_s5 hheterot_s1000 hmipL286k").split():
    held_analysis(name)
print("STAGE2-DONE")
