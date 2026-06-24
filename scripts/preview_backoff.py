"""Preview the align+insertion segment at a chosen backoff: replay recorded
states from (align_done - backoff) through the end of insertion, render to mp4.
Pulls from the clean source so we can see how misaligned a larger backoff start is.
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


def align_done(d):
    a = d["actions"][:]; ez = d["obs/robot0_eef_pos"][:, 2]; g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, len(g)) if g[t-1] >= 0 and g[t] < 0]
    if not cl or not op:
        return None, None, None
    c1, o1 = cl[0], op[0]
    return c1 + int(np.argmax(ez[c1:o1])), c1, o1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tool_hang_clean_20000.hdf5")
    ap.add_argument("--demos", nargs="+", default=["demo_0", "demo_5"])
    ap.add_argument("--backoff", type=int, default=60)
    ap.add_argument("--cam", default="sideview")
    ap.add_argument("--out", default="logs/backoff")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    kw = dict(ENV_KWARGS)
    kw["has_offscreen_renderer"] = True
    kw["use_camera_obs"] = False
    kw["camera_names"] = args.cam
    kw["camera_heights"] = args.res; kw["camera_widths"] = args.res
    env = robosuite.make("ToolHang", horizon=4000, **kw)

    sf = h5py.File(args.src, "r")
    for dk in args.demos:
        d = sf["data/" + dk]
        af, c1, o1 = align_done(d)
        if af is None:
            print(f"skip {dk}: no grasp"); continue
        start = max(c1 + 2, af - args.backoff)
        states = d["states"][:]
        sd = int(d.attrs.get("seed", 0)); np.random.seed(sd)
        env.reset()
        frames = []
        # replay from backoff start through release (o1+15)
        for st in states[start:min(o1 + 15, len(states))]:
            env.sim.set_state_from_flattened(st); env.sim.forward()
            frames.append(env.sim.render(camera_name=args.cam, width=args.res, height=args.res)[::-1])
        path = f"{args.out}{args.backoff}_{dk}.mp4"
        imageio.mimsave(path, frames, fps=20, macro_block_size=1)
        print(f"SAVED {path}  start={start} align_done={af} (backoff {af-start}), {len(frames)} frames")
    sf.close()


if __name__ == "__main__":
    main()
