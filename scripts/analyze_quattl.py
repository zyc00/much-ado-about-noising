"""Orientation-deviation timeline from swdump trajectories.
For each policy and episode: geodesic angle (deg) between the rollout eef quat and the
demo mean eef quat at the same closure-relative step, for rel in [-48, +16].
Reference: full2ins demos aligned at c1 (n=300). Prints per-policy median angle by
rel bins, split by outcome."""
import glob, os
import numpy as np, h5py

import os as _o
QDIMS = [int(x) for x in _o.environ.get('QDIMS', '47,48,49,50').split(',')]
def qfix(Q):  # sign-align to first
    Q = Q.copy()
    for i in range(1, len(Q)):
        if np.dot(Q[i], Q[i-1]) < 0: Q[i] = -Q[i]
    return Q
def ang(q1, q2):
    d = abs(float(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2) + 1e-12))
    return np.degrees(2 * np.arccos(min(1.0, d)))

RELS = np.arange(-48, 17)
h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:300]
acc = {r: [] for r in RELS}
for k in keys:
    d = h[f"data/{k}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]; Q = qfix(np.asarray(np.asarray(d["obs/object"])[:, [c-0 for c in QDIMS]] if QDIMS[0] < 44 else d["obs/robot0_eef_quat"]))
    for r in RELS:
        t = c1 + r
        if 0 <= t < len(Q): acc[r].append(Q[t])
h.close()
ref = {}
for r in RELS:
    Q = np.stack(acc[r])
    Q[np.sum(Q * Q[0], axis=1) < 0] *= -1
    m = Q.mean(0); ref[r] = m / np.linalg.norm(m)

BINS = [(-48, -25), (-24, -9), (-8, -1), (0, 7), (8, 16)]
for pol in sorted(os.listdir("logs/swdump")):
    rows = {1: {b: [] for b in BINS}, 0: {b: [] for b in BINS}}
    for f in sorted(glob.glob(f"logs/swdump/{pol}/sw_*.npz")):
        z = np.load(f)
        ca = int(z["closed_at"])
        if ca < 0: continue
        traj = z["obs_traj"]; asm = int(z["asm"])
        for r in RELS:
            t = ca + r + 1  # obs index after step ca+r
            if 0 <= t < len(traj):
                a_ = ang(traj[t, QDIMS], ref[r])
                for b in BINS:
                    if b[0] <= r <= b[1]: rows[asm][b].append(a_)
    line = f"QTL {pol}:"
    for b in BINS:
        s = np.median(rows[1][b]) if rows[1][b] else float("nan")
        fl = np.median(rows[0][b]) if rows[0][b] else float("nan")
        line += f" rel[{b[0]:+d},{b[1]:+d}] {s:.1f}/{fl:.1f}"
    print(line + "   (deg, succ/fail)", flush=True)
print("QTL-DONE")
