"""Avoidance vs robustness: for each policy's rollouts —
(a) AVOIDANCE: distribution of support-distance at near-gate states (does the policy
    generate off-support configurations at all?), per PASS/FAIL;
(b) ROBUSTNESS: cos(executed action, kNN-demo action) binned by support distance —
    how well does each policy behave AS states go off-support?"""
import numpy as np, torch, h5py, os
DSP = os.environ.get("DSP", "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
import sys
sys.path.insert(0, ".")
from mip.dataset_utils import MinMaxNormalizer
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, bank_o, bank_prev, bank_a, all_obs = [], [], [], [], []
for k in keys[:200]:
    o = h[f"data/{k}/obs"]
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    all_obs.append(ov)
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    for t in range(c1 + 5, r1, 2):
        v = ov[t, FP] - (ov[t, BP] + offs[-1])
        if np.linalg.norm(v[:2]) < 0.10:
            bank_o.append(ov[t]); bank_prev.append(ov[t-1]); bank_a.append(a[t])
h.close()
off = np.median(np.stack(offs), 0)
norm = MinMaxNormalizer(np.concatenate(all_obs))
BW = norm.normalize(np.stack([np.stack([p, c]) for p, c in zip(bank_prev, bank_o)]))
BA = np.stack(bank_a)
BKt = torch.tensor(BW.reshape(len(BW), -1), dtype=torch.float32)
print(f"bank {len(BW)}", flush=True)
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
BINS = [(0, 0.35), (0.35, 0.6), (0.6, 1.0), (1.0, 9.9)]
import os
for name in os.environ.get("NAMES", "hheterot_s5 hheterot_s1000 hmse_mlp_s5 hmipL286k").split():
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0
    dist_pass, dist_fail = [], []
    binned = {b: [] for b in range(len(BINS))}
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None:
            v = o[g0:gend, FP] - (o[g0:gend, BP] + off)
            lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
            ts = [t for t in range(1, len(v)) if 0.010 <= lat[t] < 0.080 and 0.005 <= alt[t] < 0.120][::3]
            for t in ts:
                w = norm.normalize(np.stack([o[g0 + t - 1], o[g0 + t]])[None])[0]
                q = torch.tensor(w.reshape(1, -1), dtype=torch.float32)
                dv = torch.cdist(q, BKt)[0]
                near = dv.topk(8, largest=False)
                d = float(near.values[0])
                (dist_pass if int(m[1]) else dist_fail).append(d)
                pa = a[g0 + t][:6]; ka = BA[near.indices.numpy()][:, :6].mean(0)
                c = float(np.dot(pa, ka) / (np.linalg.norm(pa) * np.linalg.norm(ka) + 1e-9))
                for b, (lo, hi) in enumerate(BINS):
                    if lo <= d < hi: binned[b].append(c); break
        i += 1
    dp = np.array(dist_pass); df = np.array(dist_fail)
    row = f"OFFSUP {name}: PASS-dist p50={np.median(dp):.2f} p90={np.quantile(dp,0.9):.2f} (n={len(dp)})"
    if len(df): row += f" | FAIL-dist p50={np.median(df):.2f} p90={np.quantile(df,0.9):.2f} (n={len(df)})"
    row += f" | frac states >0.6 off-support={np.mean(np.concatenate([dp,df]) > 0.6):.2f}"
    print(row, flush=True)
    row2 = f"OFFSUP {name} robustness cos-by-dist:"
    for b, (lo, hi) in enumerate(BINS):
        if len(binned[b]) >= 10:
            row2 += f" [{lo}-{hi}): {np.median(binned[b]):+.2f} (n={len(binned[b])})"
    print(row2, flush=True)
print("OFFSUP-DONE")
