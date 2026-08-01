"""Render failure videos + trajectories for a policy on the f2i segment
(twofactor protocol, MP-200 dataset anchors). Saves per-episode:
  analysis/failvids/<tag>_sd<seed>_<outcome>.mp4   (sideview)
  analysis/failvids/<tag>_trajs.npz                (eef pos + tube distance)
Env: FV_CKPT, FV_LOSS, FV_TAG, FV_N (episodes), FV_AS.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

import robosuite
from collect_tool_hang_demos import ENV_KWARGS
from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler
import h5py

CKPT = os.environ["FV_CKPT"]
LOSS = os.environ["FV_LOSS"]
TAG = os.environ.get("FV_TAG", "l2mp200")
NEP_ = int(os.environ.get("FV_N", "16"))
AS = int(os.environ.get("FV_AS", "8"))
DSET = os.environ.get("FV_DATASET", "data/tool_hang_full2ins_mp_200.hdf5")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(DSET),
        "network=chiunet", f"optimization.loss_type={LOSS}",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
H = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
cfg.task.horizon = H
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
ag.load(CKPT, load_optimizer=False)
ag.eval()
sampler = get_sampler(LOSS)
no = ds.normalizer["obs"]["state"]
na = ds.normalizer["action"]
dev = cfg.optimization.device
AD = 10
start = cfg.task.obs_steps - 1

h = h5py.File(DSET, "r")
anc = []
for i in range(40):
    o = h[f"data/demo_{i}/obs"]
    anc.append(np.concatenate([np.asarray(o[k]) for k in OK], axis=1)
               .astype(np.float32))
h.close()
cl = np.concatenate(anc, 0)
mu, sig = cl.mean(0), cl.std(0) + 1e-6
tree = cKDTree((cl - mu) / sig)


def ov(o):
    return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)


def chunk(hist):
    w = np.stack(hist[-2:])[None]
    ot = {"state": torch.tensor(no.normalize(w), device=dev,
                                dtype=torch.float32)}
    with torch.no_grad():
        a0 = torch.randn((1, H, AD), device=dev)
        an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema, a0, ot)
    return ds.undo_transform_action(
        na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]


env_kw = dict(ENV_KWARGS)
env_kw.update(use_camera_obs=True, has_offscreen_renderer=True,
              camera_names=["sideview"], camera_heights=320,
              camera_widths=320)
env = robosuite.make("ToolHang", horizon=4000, **env_kw)
os.makedirs("analysis/failvids", exist_ok=True)
trajs = {}
for k, sd in enumerate(range(21000, 21000 + NEP_)):
    np.random.seed(sd)
    env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]
    arm.update()
    arm.reset_goal()
    for _ in range(10):
        env.step(np.zeros(7))
    o = env._get_observations(force_update=True)
    hist = [ov(o), ov(o)]
    frames, P, D, OBS = [], [], [], []
    steps, asm = 0, False
    while steps < 700 and not asm:
        for a in chunk(hist):
            o, _, _, _ = env.step(a)
            steps += 1
            hist.append(ov(o))
            P.append(np.asarray(o["robot0_eef_pos"]).copy())
            OBS.append(np.stack(hist[-2:]).copy())
            d, _ = tree.query((hist[-1] - mu) / sig)
            D.append(float(d))
            frames.append(o["sideview_image"][::-1])
            if env._check_frame_assembled():
                asm = True
                break
            if steps >= 700:
                break
    out = "success" if asm else "fail"
    trajs[f"P{sd}"] = np.array(P)
    trajs[f"D{sd}"] = np.array(D)
    trajs[f"O{sd}"] = np.array([1 if asm else 0])
    trajs[f"W{sd}"] = np.array(OBS, dtype=np.float32)
    if (not asm or k < 2) and os.environ.get("FV_VIDEO", "1") == "1":
        vp = f"analysis/failvids/{TAG}_sd{sd}_{out}.mp4"
        imageio.mimsave(vp, frames[::2], fps=20)
        print(f"VID {vp} steps={steps} out={out} maxd={max(D):.1f}",
              flush=True)
    else:
        print(f"EP sd{sd} out={out} steps={steps} maxd={max(D):.1f}",
              flush=True)
np.savez_compressed(f"analysis/failvids/{TAG}_trajs.npz", **trajs)
print("FAILVIZ done", flush=True)
