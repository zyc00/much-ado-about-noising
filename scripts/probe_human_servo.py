"""On-support servo/dither anatomy from rollout dumps + GT demos.

Segments: for failed rollouts, the terminal 120 steps (timeout dither window); for
successful rollouts, the 60 steps before the final release; for GT demos, align1
(pre-r1 hover) and align2 (final hover).

Metrics per segment, aggregated per model:
  - progress efficiency: |net eef displacement| / sum |commanded pos action| over 20-step
    windows (dither = commands that cancel; GT tremor sets the reference level)
  - direction-flip rate: fraction of consecutive commanded xyz pairs with cos < 0
  - lag-1 autocorrelation of commanded xyz
  - command magnitude p50 (mm-scale raw units)
"""
import os, sys
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"

def seg_metrics(pos, act):
    """pos (T,3) eef positions, act (T,7) raw actions -> dict"""
    a = act[:, :3]
    out = {}
    # progress efficiency over 20-step windows
    effs = []
    for i in range(0, len(a) - 20, 10):
        net = np.linalg.norm(pos[i + 20] - pos[i])
        tot = np.abs(a[i:i + 20]).sum() + 1e-9
        effs.append(net / tot)
    out["eff"] = np.mean(effs) if effs else np.nan
    n = np.linalg.norm(a, axis=1) + 1e-9
    cosser = (a[1:] * a[:-1]).sum(1) / (n[1:] * n[:-1])
    out["flip"] = float((cosser < 0).mean())
    out["ac1"] = float(np.corrcoef(np.concatenate([a[1:].ravel()[None], a[:-1].ravel()[None]]))[0, 1])
    out["mag"] = float(np.median(n))
    return out

def agg(tag, segs):
    ks = ["eff", "flip", "ac1", "mag"]
    vals = {k: np.array([s[k] for s in segs if not np.isnan(s[k])]) for k in ks}
    print(f"SERVO {tag} (n={len(segs)}): " + " ".join(f"{k}={np.median(vals[k]):.3f}" for k in ks), flush=True)

# GT segments
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
gt_align1, gt_align2, gt_carry = [], [], []
for k in keys[:60]:
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    p = np.asarray(h[f"data/{k}/obs/robot0_eef_pos"])
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if r1 and r1 - 30 > c1: gt_align1.append(seg_metrics(p[r1-30:r1], a[r1-30:r1]))
    gt_align2.append(seg_metrics(p[T-40:], a[T-40:]))
    if r1 and r1 - 60 > c1 + 20: gt_carry.append(seg_metrics(p[c1+10:c1+60], a[c1+10:c1+60]))
h.close()
agg("GT align1", gt_align1); agg("GT align2", gt_align2); agg("GT carry1", gt_carry)

for name in ["hMSE_s5", "hMIP_s5"]:
    f = f"analysis/traj_vis/human_{name}.npz"
    if not os.path.exists(f): continue
    z = np.load(f)
    fail_segs, succ_segs = [], []
    i = 0
    while f"ep{i}_obs" in z.files:
        meta = z[f"ep{i}_meta"]; pos = z[f"ep{i}_pos"]; act = z[f"ep{i}_act"]
        L = min(len(pos) - 1, len(act))
        pos = pos[:L]; act = act[:L]
        if int(meta[0]) == 0 and L > 130:
            fail_segs.append(seg_metrics(pos[-120:], act[-120:]))
        elif int(meta[0]) == 1 and L > 70:
            succ_segs.append(seg_metrics(pos[-60:], act[-60:]))
        i += 1
    agg(f"{name} FAIL-dither", fail_segs)
    agg(f"{name} SUCC-final", succ_segs)
print("SERVO-DONE")
