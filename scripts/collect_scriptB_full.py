"""Collect full robomimic demos using scripted_tool_hang_v2 (script B) — the SAME
scripted policy as the project's original training data + warmstart eval. Saves
in the standard robomimic format (via collect_tool_hang_demos.save_demos), so the
cluster-generated data matches the project distribution exactly.

Usage:
  MUJOCO_GL=egl python scripts/collect_scriptB_full.py --n_demos 1000 --start_seed 0 \
     --output data/clean_shardB_0.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from scripted_tool_hang_v2 import run_episode
from collect_tool_hang_demos import save_demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_demos", type=int, default=1000)
    ap.add_argument("--start_seed", type=int, default=0)
    ap.add_argument("--output", default="data/clean_shardB.hdf5")
    args = ap.parse_args()

    from tqdm import tqdm
    demos = []; seed = args.start_seed; attempts = 0
    pbar = tqdm(total=args.n_demos, desc="scriptB")
    while len(demos) < args.n_demos:
        ok, traj, init_state, model_xml = run_episode(seed, record=True)
        attempts += 1
        if ok:
            demo = dict(traj); demo["model_file"] = model_xml; demo["seed"] = seed
            demos.append(demo); pbar.update(1)
        seed += 1
    pbar.close()
    print(f"scriptB collected {len(demos)} from {attempts} attempts "
          f"({100*len(demos)/attempts:.0f}%), seeds {args.start_seed}..{seed-1}")
    save_demos(demos, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
