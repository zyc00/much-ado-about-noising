"""Collect held-out EVAL demos using the SAME scripted policy as the training
data (collect_tool_hang_demos.run_episode), so the eval reference matches the
training distribution. Saves warmstart format: state0 + full actions + phase
frames (c1/align_done/o1). Replaces warmstart_demos for cluster-consistent eval.

Usage:
  MUJOCO_GL=egl python scripts/collect_eval_clusterstyle.py --start_seed 21000 --n 200 \
     --output data/eval_demos_c.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import h5py
import robosuite
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS, run_episode


def frames(traj):
    a = np.array(traj["actions"]); ez = np.array(traj["obs"]["robot0_eef_pos"])[:, 2]; g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl:
        return None
    c1 = cl[0]
    op = [t for t in range(1, len(g)) if g[t-1] >= 0 and g[t] < 0 and t > c1]
    if not op:
        return None
    o1 = op[0]; align_done = c1 + int(np.argmax(ez[c1:o1]))
    return c1, align_done, o1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_seed", type=int, default=21000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--output", default="data/eval_demos_c.hdf5")
    args = ap.parse_args()

    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
    out = h5py.File(args.output, "w")
    g = out.create_group("demos")
    seed = args.start_seed; saved = 0
    from tqdm import tqdm
    pbar = tqdm(total=args.n, desc="eval-demos-c")
    while saved < args.n:
        success, traj, init_state, model_xml = run_episode(env, seed)
        if success:
            fr = frames(traj)
            if fr is not None:
                c1, align_done, o1 = fr
                d = g.create_group(f"seed_{seed}")
                d.attrs["seed"] = seed
                d.attrs["c1"] = c1; d.attrs["align_done"] = align_done; d.attrs["o1"] = o1
                d.create_dataset("state0", data=np.array(traj["states"][0], dtype=np.float64))
                d.create_dataset("actions", data=np.array(traj["actions"], dtype=np.float64))
                saved += 1; pbar.update(1)
        seed += 1
    pbar.close(); out.close()
    print(f"EVAL_DEMOS_C saved {saved} demos to {args.output} (seeds {args.start_seed}..{seed-1})")


if __name__ == "__main__":
    main()
