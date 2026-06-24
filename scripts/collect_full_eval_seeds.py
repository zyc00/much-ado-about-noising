"""Collect held-out, SOLVABLE eval seeds for the init->insertion segment.
Run the scripted policy on fresh seeds (>=21000, not in any training set); keep
seeds where it succeeds (guarantees the placement is solvable). Save the seed
list so eval uses natural env.reset(seed) -> same placement, no set_state.

Usage:
  MUJOCO_GL=egl python scripts/collect_full_eval_seeds.py \
     --start_seed 21000 --n 200 --output data/full_eval_seeds.npy
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import sys
sys.path.insert(0, "scripts")
from scripted_tool_hang_v2 import run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_seed", type=int, default=21000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--output", default="data/full_eval_seeds.npy")
    args = ap.parse_args()

    seeds = []
    seed = args.start_seed
    from tqdm import tqdm
    pbar = tqdm(total=args.n, desc="eval-seeds")
    while len(seeds) < args.n:
        ok, *_ = run_episode(seed, record=True)
        if ok:
            seeds.append(seed); pbar.update(1)
        seed += 1
    pbar.close()
    seeds = np.array(seeds, dtype=np.int64)
    np.save(args.output, seeds)
    print(f"Saved {args.output}: {len(seeds)} solvable held-out seeds "
          f"({args.start_seed}..{seed-1}, yield {100*len(seeds)/(seed-args.start_seed):.0f}%)")


if __name__ == "__main__":
    main()
