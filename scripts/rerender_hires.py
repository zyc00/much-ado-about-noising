"""Re-render a robomimic image dataset at higher camera resolution by
replaying recorded sim states. Source env_meta/states/actions from the
existing image_abs file (robosuite-1.5-format env_args).

Usage: python scripts/rerender_hires.py SRC DST HEIGHT [d_lo d_hi]"""
import collections
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np

SRC, DST, RES = sys.argv[1], sys.argv[2], int(sys.argv[3])
D_LO = int(sys.argv[4]) if len(sys.argv) > 4 else 0
D_HI = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 9

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

fs = h5py.File(SRC, "r")
env_meta = json.loads(fs["data"].attrs["env_args"])
env_meta["env_kwargs"]["camera_heights"] = RES
env_meta["env_kwargs"]["camera_widths"] = RES
d0 = sorted(fs["data"].keys(), key=lambda k: int(k.split("_")[1]))[0]
obs_keys = list(fs[f"data/{d0}/obs"].keys())
img_keys = [k for k in obs_keys if k.endswith("_image")]
low_keys = [k for k in obs_keys if not k.endswith("_image")]

mod = collections.defaultdict(list)
for k in img_keys:
    mod["rgb"].append(k)
for k in low_keys:
    mod["low_dim"].append(k)
ObsUtils.initialize_obs_modality_mapping_from_dict(mod)
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta, render=False, render_offscreen=True,
    use_image_obs=True)
print("env ready", flush=True)


def to_uint8(x):
    # processed rgb float CHW [0,1] -> uint8 HWC
    if x.dtype == np.uint8:
        return x
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8).transpose(
        1, 2, 0)


demos = sorted(fs["data"].keys(), key=lambda k: int(k.split("_")[1]))
demos = demos[D_LO:D_HI]
fd = h5py.File(DST, "w")
g = fd.create_group("data")
g.attrs["env_args"] = json.dumps(env_meta)
total = 0
for di, dk in enumerate(demos):
    src = fs[f"data/{dk}"]
    states = np.asarray(src["states"])
    T = len(states)
    buf = {k: [] for k in obs_keys}
    env.reset()
    for t in range(T):
        ob = env.reset_to({"states": states[t]})
        for k in img_keys:
            buf[k].append(to_uint8(ob[k]))
        for k in low_keys:
            buf[k].append(np.asarray(ob[k], dtype=np.float64))
    gd = g.create_group(dk)
    for name in ["actions", "rewards", "dones", "states"]:
        gd.create_dataset(name, data=np.asarray(src[name]))
    go = gd.create_group("obs")
    for k in img_keys:
        go.create_dataset(k, data=np.stack(buf[k]), compression="gzip",
                          compression_opts=1, chunks=(1, RES, RES, 3))
    for k in low_keys:
        go.create_dataset(k, data=np.stack(buf[k]))
    gd.attrs["num_samples"] = T
    total += T
    print(f"[{D_LO + di + 1}] {dk} T={T}", flush=True)
g.attrs["total"] = total
if "mask" in fs and D_LO == 0:
    fs.copy("mask", fd)
fd.close()
fs.close()
print("DONE", flush=True)
