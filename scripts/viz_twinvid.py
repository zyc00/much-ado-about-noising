"""Side-by-side sim replay of a near-twin conflict: two demos whose states
nearly coincide at a sync point, then diverge. Renders mp4."""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import imageio
import numpy as np
import robosuite
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.datasets.robomimic_dataset import make_dataset
from scripted_tool_hang_v2 import ENV_KWARGS

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
RES = 512
CAM = "frontview"

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
no = ds.normalizer["obs"]["state"]

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PD, PI = [], [], []
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    for i in range(1, len(S) - 16):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        PD.append(di)
        PI.append(i)
PW = np.stack(PW)
PD, PI = np.asarray(PD), np.asarray(PI)
tP = torch.tensor(PW)
PA = []
for di, dn in enumerate(names):
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    pass

rng = np.random.default_rng(0)
qidx = rng.choice(np.where(PD < 8)[0], 500, replace=False)
best, bestconf, bestnb = None, -1, None
AC = {}
for qi in qidx:
    d = torch.norm(tP - torch.tensor(PW[qi]), dim=1).numpy()
    d[PD == PD[qi]] = 1e9
    nb = np.where(d < 0.5)[0]
    if len(nb) < 4:
        continue
    T = []
    for j in nb:
        dn = names[PD[j]]
        if dn not in AC:
            AC[dn] = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
        T.append(AC[dn][PI[j]:PI[j] + 8, 0:3].reshape(-1))
    T = np.stack(T)
    conf = float(np.sqrt(((T - T.mean(0)) ** 2).mean()))
    if conf > bestconf:
        bestconf, best, bestnb = conf, qi, nb
dA, iA = PD[best], PI[best]
j = bestnb[int(np.argmin([abs(PI[j] - iA) for j in bestnb]))]
dB, iB = PD[j], PI[j]
print(f"TV twin: demo{dA}@{iA} vs demo{dB}@{iB} conf {bestconf:.3f}",
      flush=True)
SA = np.asarray(h[f"data/{names[dA]}/states"])
SB = np.asarray(h[f"data/{names[dB]}/states"])
h.close()

kw = dict(ENV_KWARGS)
kw["has_offscreen_renderer"] = True
kw["use_camera_obs"] = False
env = robosuite.make("ToolHang", horizon=4000, **kw)
env.reset()


def frames_for(S, i0, lo, hi):
    out = []
    for i in range(max(0, i0 + lo), min(len(S), i0 + hi)):
        env.sim.set_state_from_flattened(S[i].astype(np.float64))
        env.sim.forward()
        f = env.sim.render(camera_name=CAM, width=RES, height=RES)[::-1]
        out.append((i - i0, f))
    return out


LO, HI = -60, 45
FA = frames_for(SA, iA, LO, HI)
FB = frames_for(SB, iB, LO, HI)
import numpy as np
n = min(len(FA), len(FB))
W = imageio.get_writer("analysis/paper/twin_conflict.mp4", fps=10)
for k in range(n):
    ta, fa = FA[k]
    tb, fb = FB[k]
    pane = np.concatenate([fa, fb], axis=1).copy()
    if -2 <= ta <= 2:
        pane[:12, :, 0] = 255
        pane[:12, :, 1:] = 0
    W.append_data(pane)
    if ta == 0:
        for _ in range(14):
            W.append_data(pane)
W.close()
print("TV saved analysis/paper/twin_conflict.mp4", flush=True)
