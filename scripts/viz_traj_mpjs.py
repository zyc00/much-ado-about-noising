"""Pilot: canonical MP recipe vs Plan-3 (MP_JS=1, joint-space-planned
transits with kinematically verified references). 10 paired seeds; per-
episode success, spacing stats, stall fraction; trajectories saved to npz
for the 3D comparison. Prints VJS lines."""
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ.update({"MJTRAJ": "1", "MJ_FF": "1", "MJ_SLERP": "1",
                   "MJ_KFB": "5", "MJ_ALIGN": "0"})
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np

from scripted_tool_hang_v2 import run_episode

SEEDS = [300001 + i for i in range(10)]
out = {}
for mode, js in (("base", "0"), ("mpjs", "1")):
    os.environ["MP_JS"] = js
    for sd in SEEDS:
        ok, traj, _i, _m = run_episode(sd, record=True)
        eef = np.asarray(traj["obs"]["robot0_eef_pos"])
        act = np.asarray(traj["actions"])
        if sd <= 300003:
            out[f"{mode}_{sd}_eef"] = eef
            out[f"{mode}_{sd}_act"] = act
        dstep = np.linalg.norm(np.diff(eef, axis=0), axis=1)
        cmd = np.abs(act[:, 0:3]).max(1) * 0.05
        stall = float(((dstep < 0.001) & (cmd[:-1] > 0.005)).mean())
        print(f"VJS {mode} seed {sd} ok={ok} T={len(eef)} "
              f"p50={np.median(dstep)*1000:.2f}mm "
              f"p95={np.percentile(dstep,95)*1000:.2f}mm "
              f"stall={stall:.3f}", flush=True)
np.savez_compressed("analysis/manifold/viz_mpjs_trajs.npz", **out)
print("VJS done", flush=True)
