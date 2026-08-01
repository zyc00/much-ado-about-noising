"""Render one original MP-200 demo and one rotlim demo from stored sim
states, same placement-agnostic framing, to mp4s."""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import imageio
import numpy as np
import robosuite

from scripted_tool_hang_v2 import ENV_KWARGS

kw = dict(ENV_KWARGS)
kw["has_offscreen_renderer"] = True
kw["use_camera_obs"] = False
env = robosuite.make("ToolHang", horizon=4000, **kw)
env.reset()
for path, dn, out, stride in [
        ("data/tool_hang_full2ins_mp_200.hdf5", "demo_0",
         "analysis/paper/vid_orig_demo.mp4", 1),
        ("data/tool_hang_mp200_rotlim.hdf5", "demo_0",
         "analysis/paper/vid_rotlim_demo.mp4", 2)]:
    h = h5py.File(path, "r")
    S = np.asarray(h[f"data/{dn}/states"])
    h.close()
    W = imageio.get_writer(out, fps=20)
    for i in range(0, len(S), stride):
        env.sim.set_state_from_flattened(S[i].astype(np.float64))
        env.sim.forward()
        W.append_data(env.sim.render(camera_name="frontview", width=512,
                                     height=512)[::-1])
    W.close()
    print(f"VV saved {out} ({len(S)} states)", flush=True)
