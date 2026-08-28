"""Replicate train_kitchen's DataLoader config exactly; find NaN batch."""
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
from mip.utils import loop_dataloader

ds = make_dataset(cfg.task)
dl = torch.utils.data.DataLoader(
    ds, batch_size=cfg.optimization.batch_size, num_workers=4, shuffle=True,
    pin_memory=True, persistent_workers=True)
loop = loop_dataloader(dl)
dev = cfg.optimization.device
ag = TrainingAgent(cfg)
for i in range(12):
    b = next(loop)
    obs = b["obs"]["state"].to(dev)[:, :cfg.task.obs_steps, :]
    act = b["action"].to(dev)[:, :cfg.task.horizon, :]
    dt = torch.zeros(len(act), device=dev)
    info = ag.update(act, obs, dt)
    l = float(info["loss"])
    print(f"step {i} loss {l:.5f} obsnan {int(torch.isnan(obs).sum())} "
          f"actnan {int(torch.isnan(act).sum())}")
    if l != l:
        print("NaN AT STEP", i)
        break
