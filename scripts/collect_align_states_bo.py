"""Collect held-out align-done states for insertion-only eval.

Run the scripted controller on fresh seeds (held-out: not in any training set)
up to the align-done frame (eef_z peak while holding frame), and save the
flattened mujoco state + seed + model_xml. These states serve as independent
reset points for isolated-insertion evaluation, valid for ALL training scales
(including 20000).

Usage:
  MUJOCO_GL=egl python scripts/collect_align_states.py \
     --start_seed 21000 --n 200 --output data/insertion_eval_states.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import h5py
import sys
sys.path.insert(0, "scripts")
from scripted_tool_hang_v2 import run_episode


def align_done_frame_from_traj(traj):
    a = np.array(traj["actions"]); ez = np.array(traj["obs"]["robot0_eef_pos"])[:, 2]; g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, len(g)) if g[t-1] >= 0 and g[t] < 0]
    if not cl or not op:
        return None
    c1, o1 = cl[0], op[0]
    return c1 + int(np.argmax(ez[c1:o1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_seed", type=int, default=21000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--output", default="data/insertion_eval_states.hdf5")
    ap.add_argument("--backoff_max", type=int, default=0)
    ap.add_argument("--backoff_fixed", type=int, default=0)
    args = ap.parse_args()

    states = []; seeds = []; xmls = []
    seed = args.start_seed
    from tqdm import tqdm
    pbar = tqdm(total=args.n, desc="align-states")
    while len(states) < args.n:
        ok, traj, init_state, model_xml = run_episode(seed, record=True)
        if ok and len(traj["actions"]) > 0:
            af = align_done_frame_from_traj(traj)
            if af is not None:
                if args.backoff_max>0 or args.backoff_fixed>0:
                    a=np.array(traj["actions"]);g=a[:,6]
                    c1=next((t for t in range(1,len(g)) if g[t-1]<0 and g[t]>=0),0)
                    bo=args.backoff_fixed if args.backoff_fixed>0 else int(np.random.RandomState(seed).randint(0,args.backoff_max+1))
                    af=max(c1+2, af-bo)
                states.append(np.array(traj["states"])[af])
                seeds.append(seed); xmls.append(model_xml)
                pbar.update(1)
        seed += 1
    pbar.close()
    print(f"Collected {len(states)} align states from seeds {args.start_seed}..{seed-1}")

    with h5py.File(args.output, "w") as f:
        f.create_dataset("states", data=np.array(states))
        f.create_dataset("seeds", data=np.array(seeds))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("model_xmls", data=np.array(xmls, dtype=object), dtype=dt)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
