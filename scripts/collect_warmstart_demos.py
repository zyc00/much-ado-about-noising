"""Collect scripted demos for the held-out eval seeds, to drive WARM-START eval.
For each seed: save the post-settle initial state (frame0), the full clean action
sequence, and phase frames (c1=grasp, align_done, o1=release). Warm-start eval
reaches depth K by: set_state(frame0) -> reset_goal -> replay actions[:K] -> policy.
This reproduces the natural OSC controller reference at K (no mid-traj set_state).

Usage:
  MUJOCO_GL=egl python scripts/collect_warmstart_demos.py \
     --seeds_file data/full_eval_seeds.npy --output data/warmstart_demos.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import h5py
import sys
sys.path.insert(0, "scripts")
from scripted_tool_hang_v2 import run_episode


def phase_frames(traj):
    a = np.array(traj["actions"]); ez = np.array(traj["obs"]["robot0_eef_pos"])[:, 2]; g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, len(g)) if g[t-1] >= 0 and g[t] < 0]
    if not cl or not op:
        return None
    c1, o1 = cl[0], op[0]
    align_done = c1 + int(np.argmax(ez[c1:o1]))
    return c1, align_done, o1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_file", default="data/full_eval_seeds.npy")
    ap.add_argument("--output", default="data/warmstart_demos.hdf5")
    args = ap.parse_args()

    seeds = np.load(args.seeds_file)
    from tqdm import tqdm
    out = h5py.File(args.output, "w")
    g = out.create_group("demos")
    n = 0
    for sd in tqdm(seeds, desc="warmstart-demos"):
        ok, traj, init_state, model_xml = run_episode(int(sd), record=True)
        if not ok:
            continue
        fr = phase_frames(traj)
        if fr is None:
            continue
        c1, align_done, o1 = fr
        d = g.create_group(f"seed_{int(sd)}")
        d.attrs["seed"] = int(sd)
        d.attrs["c1"] = c1; d.attrs["align_done"] = align_done; d.attrs["o1"] = o1
        d.create_dataset("state0", data=np.array(traj["states"][0], dtype=np.float64))
        d.create_dataset("actions", data=np.array(traj["actions"], dtype=np.float64))
        n += 1
    out.attrs["n"] = n
    out.close()
    print(f"Saved {args.output}: {n} warm-start demos")


if __name__ == "__main__":
    main()
