"""Transport-mh failure anatomy from TRAJDUMP npz files.
Stages from executed gripper channels (delta 14-dim: g0=act[:,6], g1=act[:,13]):
  c0 = arm0 first sustained close (hammer/lid grasp)
  c1 = arm1 first sustained close after c0 (receive at handover)
  o0 = arm0 open after c1 (handover complete)
  o1 = arm1 open after o0 (place)
Failure stage = last event reached. Also pacing: median |pos-action| while moving,
policy vs demos."""
import glob
import os

import numpy as np

PREFIX = os.environ.get("PREFIX", "analysis/traj_vis/trmh_ht")
def sustained(sig, thr, k=5):
    on = sig >= thr
    for t in range(len(on) - k):
        if on[t:t + k].all():
            return t
    return None
rows = {"none": 0, "grasp0": 0, "receive1": 0, "handover": 0, "place": 0, "success": 0}
mags_active = []
for f in sorted(glob.glob(f"{PREFIX}_ep*.npz")):
    z = np.load(f)
    O, A, S = z["obs"], z["act"], z["succ"]
    for e in range(len(S)):
        a = A[e]
        g0, g1 = a[:, 6], a[:, 13]
        c0 = sustained(g0, 0.0)
        c1 = sustained(g1, 0.0) if c0 is not None else None
        if c1 is not None and c1 <= c0:
            c1 = sustained(g1[c0:], 0.0)
            c1 = c1 + c0 if c1 is not None else None
        o0 = None
        if c1 is not None:
            oo = sustained(-g0[c1:], 0.0)
            o0 = oo + c1 if oo is not None else None
        o1 = None
        if o0 is not None:
            oo = sustained(-g1[o0:], 0.0)
            o1 = oo + o0 if oo is not None else None
        if S[e] > 0:
            rows["success"] += 1
        elif o0 is not None:
            rows["place"] += 1
        elif c1 is not None:
            rows["handover"] += 1
        elif c0 is not None:
            rows["receive1"] += 1
        else:
            rows["none"] += 1
        p0 = np.linalg.norm(a[:, 0:3], axis=1)
        p1 = np.linalg.norm(a[:, 7:10], axis=1)
        m = np.maximum(p0, p1)
        mags_active.append(np.median(m[m > np.quantile(m, 0.3)]))
n = sum(rows.values())
print(f"STAGE {os.path.basename(PREFIX)} n={n}:", {k: v for k, v in rows.items()})
print(f"  interpretation: none=never grasped | receive1=grasped, no handover-receive |")
print(f"  handover=received, arm0 never released | place=released, place failed")
print(f"PACING {os.path.basename(PREFIX)}: active |pos-cmd| median over episodes = {np.median(mags_active):.4f}")
# demo reference pacing
import h5py
DSP = os.environ.get("TDS", "/mnt/pfs/yuchen/code/much-ado-about-noising/data/hfdl")
cands = glob.glob("/root/.cache/huggingface/**/transport/mh/low_dim.hdf5", recursive=True) + \
        glob.glob(os.path.expanduser("~/.cache/huggingface/**/transport/mh/low_dim.hdf5"), recursive=True)
if cands:
    h = h5py.File(cands[0], "r")
    dm = []
    for k in list(h["data"].keys())[:100]:
        a = np.asarray(h[f"data/{k}/actions"]).astype(np.float32)
        p0 = np.linalg.norm(a[:, 0:3], axis=1)
        p1 = np.linalg.norm(a[:, 7:10], axis=1)
        m = np.maximum(p0, p1)
        dm.append(np.median(m[m > np.quantile(m, 0.3)]))
    print(f"PACING demos: active |pos-cmd| median = {np.median(dm):.4f}")
print("ANALYZE-DONE")
