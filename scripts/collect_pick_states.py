"""Collect held-out pick-done (grasp-close, c1) states for front-half eval.
Same scripted controller on held-out seeds (21000+); save flattened state at c1.

Usage:
  MUJOCO_GL=egl python scripts/collect_pick_states.py \
     --start_seed 21000 --n 200 --output data/pick_eval_states.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import h5py
import sys
sys.path.insert(0, "scripts")
from scripted_tool_hang_v2 import run_episode


def pick_frame(traj):
    a = np.array(traj["actions"]); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    return cl[0] if cl else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_seed", type=int, default=21000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--output", default="data/pick_eval_states.hdf5")
    args = ap.parse_args()

    states = []; seeds = []; xmls = []
    seed = args.start_seed
    from tqdm import tqdm
    pbar = tqdm(total=args.n, desc="pick-states")
    while len(states) < args.n:
        ok, traj, init_state, model_xml = run_episode(seed, record=True)
        if ok and len(traj["actions"]) > 0:
            c1 = pick_frame(traj)
            if c1 is not None:
                states.append(np.array(traj["states"])[c1])
                seeds.append(seed); xmls.append(model_xml)
                pbar.update(1)
        seed += 1
    pbar.close()
    print(f"Collected {len(states)} pick states from seeds {args.start_seed}..{seed-1}")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("states", data=np.array(states))
        f.create_dataset("seeds", data=np.array(seeds))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("model_xmls", data=np.array(xmls, dtype=object), dtype=dt)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
