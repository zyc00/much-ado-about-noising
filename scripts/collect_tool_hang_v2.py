"""Collect scripted tool_hang demos (Markovian v2 controller) in robomimic HDF5.

Thin wrapper: calls scripted_tool_hang_v2.run_episode(record=True), which is the
SAME code path as the 100%-SR controller (single source of truth -- no divergent
duplicate). The leading settle steps are dropped from the recording, and all
phase transitions are obs-triggered (Markovian).

Usage:
    MUJOCO_GL=egl python scripts/collect_tool_hang_v2.py \
        --n_demos 200 --output data/tool_hang_markovian_200.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, "scripts")
from scripted_tool_hang_v2 import run_episode, ENV_ARGS, OBS_KEYS_TO_RECORD


def save_demos(demos, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(ENV_ARGS)
        data.attrs["total"] = sum(len(d["actions"]) for d in demos)
        for i, demo in enumerate(demos):
            grp = data.create_group(f"demo_{i}")
            grp.attrs["num_samples"] = len(demo["actions"])
            grp.attrs["model_file"] = demo["model_file"]
            grp.attrs["seed"] = demo["seed"]
            grp.create_dataset("actions", data=np.array(demo["actions"], dtype=np.float64))
            grp.create_dataset("states", data=np.array(demo["states"], dtype=np.float64))
            grp.create_dataset("rewards", data=np.array(demo["rewards"], dtype=np.float64))
            grp.create_dataset("dones", data=np.array(demo["dones"], dtype=np.int64))
            obs_grp = grp.create_group("obs")
            for k in OBS_KEYS_TO_RECORD:
                obs_grp.create_dataset(k, data=np.array(demo["obs"][k]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_demos", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/tool_hang_markovian_200.hdf5")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--fixed_placement", type=int, default=-1,
                        help="if >=0, fix placement seed to this; vary only DART noise seed (single-tube recovery coverage)")
    args = parser.parse_args()

    demos = []
    seed = args.start_seed
    attempts = 0
    pbar = tqdm(total=args.n_demos, desc="Collecting")
    while len(demos) < args.n_demos:
        if args.fixed_placement >= 0:
            success, traj, init_state, model_xml = run_episode(
                args.fixed_placement, record=True, noise_seed=seed)
        else:
            success, traj, init_state, model_xml = run_episode(seed, record=True)
        attempts += 1
        if success and len(traj["actions"]) > 0:
            traj["model_file"] = model_xml
            traj["seed"] = seed
            demos.append(traj)
            pbar.update(1)
        seed += 1
    pbar.close()
    print(f"Collected {len(demos)} successful demos from {attempts} attempts "
          f"({100 * len(demos) / attempts:.0f}% success)")

    save_demos(demos, args.output)
    print(f"Saved to {args.output}")
    n_samples = sum(len(d["actions"]) for d in demos)
    print(f"Total samples: {n_samples}, avg per demo: {n_samples / len(demos):.0f}")


if __name__ == "__main__":
    main()
