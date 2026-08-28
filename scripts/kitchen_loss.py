"""Direct loss computation with the trainer's exact signature."""
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
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
b = next(iter(dl))
dev = cfg.optimization.device
obs = b["obs"]["state"].to(dev)[:, :cfg.task.obs_steps, :]   # BARE tensor
act = b["action"].to(dev)[:, :cfg.task.horizon, :]
ag = TrainingAgent(cfg)
dt = torch.zeros(len(act), device=dev)

emb = ag.encoder(obs, None)          # bare tensor, as trainer does
print(f"emb {tuple(emb.shape)} nan {int(torch.isnan(emb).sum())}")
loss, _ = ag.loss_fn(cfg.optimization, ag.flow_map, ag.encoder,
                     ag.interpolant, act, obs, dt)
print("DIRECT_LOSS", float(loss))
info = ag.update(act, obs, dt)       # exact trainer call
print("UPDATE_LOSS", float(info["loss"]), "gn", float(info["grad_norm"]))
print("act_dim cfg", cfg.task.act_dim, "obs_dim", cfg.task.obs_dim,
      "horizon", cfg.task.horizon, "obs_steps", cfg.task.obs_steps,
      "act shape", tuple(act.shape))
