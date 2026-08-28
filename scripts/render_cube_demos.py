"""Render demo videos from the cube NFL training datasets by state playback.

For each requested dataset, replays the stored sim states of a few demos with
an offscreen camera and writes one mp4 per (setting, demo).

Usage (cluster GPU node):
  MUJOCO_GL=egl python scripts/render_cube_demos.py \
      --sets clean:data/th_cbclean_200.hdf5 sk5:data/th_cbsk5_200.hdf5 \
      --idx 0 1 --out /mnt/pfs/yuchen/cubevids
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import sys

import h5py
import imageio
import numpy as np
import robosuite

sys.path.insert(0, "scripts")
from collect_cube_script import ENV_KWARGS, BigCubeLift  # noqa: F401,E402  (registers class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="tag:path pairs")
    ap.add_argument("--idx", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--out", default="cubevids")
    ap.add_argument("--cam", default="frontview")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    kw = dict(ENV_KWARGS)
    kw["has_offscreen_renderer"] = True
    kw["use_camera_obs"] = False
    kw["camera_names"] = args.cam
    kw["camera_heights"] = args.res
    kw["camera_widths"] = args.res
    env = robosuite.make("BigCubeLift", horizon=4000, **kw)

    for pair in args.sets:
        tag, path = pair.split(":", 1)
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for i in args.idx:
                g = f["data"][demos[i]]
                states = g["states"][:]
                xml = g.attrs["model_file"]
                env.reset()
                env.reset_from_xml_string(xml)
                frames = []
                for s in states:
                    env.sim.set_state_from_flattened(s)
                    env.sim.forward()
                    fr = env.sim.render(
                        camera_name=args.cam, width=args.res, height=args.res
                    )[::-1]
                    frames.append(fr)
                outp = os.path.join(args.out, f"cube_{tag}_demo{i}.mp4")
                imageio.mimwrite(outp, frames, fps=args.fps, quality=8)
                print(f"WROTE {outp} ({len(frames)} frames)", flush=True)
    print("RENDER_DONE", flush=True)


if __name__ == "__main__":
    main()
