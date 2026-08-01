#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
# build PD assets for the dart02 dataset normalizer
python - <<'PYEOF'
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)
DS = "data/tool_hang_full2ins_wp3dart02_2000.hdf5"
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(DS), "network=chiunet",
        "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
h = h5py.File(DS, "r")
cl = np.concatenate([np.concatenate([np.asarray(h[f"data/demo_{i}/obs"][k]) for k in OK], axis=1) for i in range(40)], 0)
h.close()
sig_raw = cl.std(0) + 1e-9
base = cl[:1]
slope_obs = (no.normalize(base + 1.0) - no.normalize(base))[0]
obs_scale = (sig_raw * slope_obs)[POS].astype(np.float32)
a0 = na.unnormalize(np.zeros((1, 16, 10))); a1 = na.unnormalize(np.ones((1, 16, 10)))
act_scale = (1.0 / (a1 - a0)[0, 0, :3]).astype(np.float32)
np.savez("logs/pd_assets_dart02.npz", obs_scale=obs_scale, act_scale=act_scale)
print("obs_scale", obs_scale, "act_scale", act_scale)
PYEOF
export PD_NPZ=logs/pd_assets_dart02.npz
export KZ=$1 LAMPD=100 PDH=16
rm -rf logs/pds_dart02_k$1
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_full2ins_wp3dart02_2000.hdf5 \
  task.num_envs=1 network=chiunet optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=300001 log.save_freq=5000 \
  optimization.loss_type=regression_pdprior log.exp_name=pds_dart02_k$1 log.log_dir=logs/pds_dart02_k$1
