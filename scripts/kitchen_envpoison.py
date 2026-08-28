"""Test: does creating the kitchen env break subsequent GPU math?"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, ".")
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=kitchen_state", "network=chiunet",
        "+task.dataset_path=/mnt/pfs/yuchen/data/kitchen_raw/kitchen",
        "~task.dataset_repo", "optimization.loss_type=regression",
        "log.wandb_mode=disabled", "optimization.auto_resume=false"])
OmegaConf.set_struct(cfg, False)
from mip.datasets.kitchen_dataset import make_dataset
from mip.agent import TrainingAgent

ds = make_dataset(cfg.task)
dl = torch.utils.data.DataLoader(ds, batch_size=cfg.optimization.batch_size,
                                 shuffle=True)
b = next(iter(dl))
dev = cfg.optimization.device
obs = b["obs"]["state"].to(dev)[:, :cfg.task.obs_steps, :]
act = b["action"].to(dev)[:, :cfg.task.horizon, :]
dt = torch.zeros(len(act), device=dev)
ag = TrainingAgent(cfg)

l1, _ = ag.loss_fn(cfg.optimization, ag.flow_map, ag.encoder, ag.interpolant,
                   act, obs, dt)
print(f"BEFORE_ENV loss {float(l1)} (bs={len(act)})")

from examples.train_kitchen import make_vec_env
envs = make_vec_env(cfg.task, seed=1)
print("env created")

l2, _ = ag.loss_fn(cfg.optimization, ag.flow_map, ag.encoder, ag.interpolant,
                   act, obs, dt)
print(f"AFTER_ENV loss {float(l2)}")
info = ag.update(act, obs, dt)
print(f"AFTER_ENV update {float(info['loss'])}")
