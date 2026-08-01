"""Condition-1 check on 'common' human teleop data: do quiet windows carry
label VARIATION (tremor-protected) or exact/near-zero constants (deadband ->
collapse-prone)? Reports: fraction of exact-zero pos-action steps, fraction
of quiet steps (|a_pos|<q40) with near-zero variation in their own +-2 window,
and local label std at quiet vs moving states. Envs: DSP, TAG."""
import os
import h5py
import numpy as np

DSP = os.environ["DSP"]; TAG = os.environ["TAG"]
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:150]
Z = MW = QN = 0; NT = 0
qvars, mvars = [], []
for k in keys:
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    pos = np.concatenate([a[:, :3], a[:, 7:10]], 1) if a.shape[1] >= 14 else a[:, :3]
    m = np.linalg.norm(pos, axis=1)
    NT += len(m)
    Z += int((m < 1e-8).sum())
    thr = np.quantile(m, 0.4)
    for t in range(2, len(m) - 2):
        w = pos[t - 2:t + 3]
        v = float(np.std(w))
        if m[t] < thr:
            qvars.append(v)
            if v < 1e-6: QN += 1
        else:
            mvars.append(v)
h.close()
qvars = np.array(qvars); mvars = np.array(mvars)
print(f"PAUSEVAR {TAG}: exact-zero steps {100*Z/NT:.1f}% | quiet steps with ~zero local variation "
      f"{100*QN/max(1,len(qvars)):.1f}% | local std quiet p10/50={np.percentile(qvars,10):.4f}/{np.median(qvars):.4f} "
      f"vs moving p50={np.median(mvars):.4f}", flush=True)
