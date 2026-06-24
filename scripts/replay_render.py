"""Replay recorded mujoco states from a demo hdf5 and render to mp4.

Usage:
  MUJOCO_GL=egl python scripts/replay_render.py \
     --src data/tool_hang_aligninsert_20000.hdf5 \
     --demos demo_0 demo_3 demo_7 --cam sideview --out logs/aligninsert
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import h5py
import imageio
import robosuite
import sys
sys.path.insert(0, "scripts")
from collect_tool_hang_demos import ENV_KWARGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tool_hang_aligninsert_20000.hdf5")
    ap.add_argument("--demos", nargs="+", default=["demo_0"])
    ap.add_argument("--cam", default="sideview")
    ap.add_argument("--out", default="logs/aligninsert")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    kw = dict(ENV_KWARGS)
    kw["has_offscreen_renderer"] = True
    kw["use_camera_obs"] = False
    kw["camera_names"] = args.cam
    kw["camera_heights"] = args.res
    kw["camera_widths"] = args.res
    env = robosuite.make("ToolHang", horizon=4000, **kw)

    sf = h5py.File(args.src, "r")
    for dk in args.demos:
        d = sf["data/" + dk]
        states = d["states"][:]
        sd = int(d.attrs.get("seed", 0))
        np.random.seed(sd)
        env.reset()
        frames = []
        for st in states:
            env.sim.set_state_from_flattened(st)
            env.sim.forward()
            img = env.sim.render(camera_name=args.cam, width=args.res, height=args.res)[::-1]
            frames.append(img)
        path = f"{args.out}_{dk}.mp4"
        imageio.mimsave(path, frames, fps=args.fps, macro_block_size=1)
        print(f"SAVED {path}  ({len(frames)} frames, seed {sd})")
    sf.close()


if __name__ == "__main__":
    main()
