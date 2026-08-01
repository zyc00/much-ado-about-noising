"""Collect paired episodes (same seeds) with time-indexed vs state-indexed
MP tracking and dump eef trajectories + actions to npz for 3D comparison.

Canonical MP recipe env is set here (MJTRAJ/MJ_FF/MJ_SLERP/MJ_KFB=5/
MJ_ALIGN=0); the fixed mode adds MP_STATEIDX=1. Env toggled between calls
(run_episode reads os.environ per call). Prints VIZ lines.
"""
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ.update({"MJTRAJ": "1", "MJ_FF": "1", "MJ_SLERP": "1",
                   "MJ_KFB": "5", "MJ_ALIGN": "0"})
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np

from scripted_tool_hang_v2 import run_episode

SEEDS = [int(s) for s in os.environ.get("VZ_SEEDS", "300001,300002,300003").split(",")]
out = {}
for mode, six in (("time", "0"), ("state", "1")):
    os.environ["MP_STATEIDX"] = six
    os.environ["MP_DPCAP"] = "0"   # governor retired: wrong abstraction
    for sd in SEEDS:
        ok, traj, _i, _m = run_episode(sd, record=True)
        eef = np.asarray(traj["obs"]["robot0_eef_pos"])
        act = np.asarray(traj["actions"])
        out[f"{mode}_{sd}_eef"] = eef
        out[f"{mode}_{sd}_act"] = act
        ph = traj.get("phase")
        if ph is not None:
            labs = sorted(set(ph))
            out[f"{mode}_{sd}_phase"] = np.array([labs.index(p) for p in ph])
            out[f"{mode}_{sd}_phaselabs"] = np.array(labs)
        dstep = np.linalg.norm(np.diff(eef, axis=0), axis=1)
        cmd = np.abs(act[:, 0:3]).max(1) * 0.05
        stall = float(((dstep < 0.001) & (cmd[:-1] > 0.005)).mean())
        print(f"VIZ {mode} seed {sd} ok={ok} T={len(eef)} "
              f"speed_p50={np.median(dstep)*1000:.2f}mm p95={np.percentile(dstep,95)*1000:.2f}mm "
              f"stall_frac={stall:.3f}", flush=True)
np.savez_compressed("analysis/manifold/viz_stateidx_trajs.npz", **out)
print("VIZ done", flush=True)
