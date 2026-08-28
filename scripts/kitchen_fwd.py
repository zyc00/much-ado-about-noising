"""Exact-path kitchen forward: replicate train_kitchen batch handling."""
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
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
b = next(iter(dl))
dev = cfg.optimization.device
obs = b["obs"]["state"].to(dev)[:, :cfg.task.obs_steps, :]
act = b["action"].to(dev)[:, :cfg.task.horizon, :]
print(f"obs {tuple(obs.shape)} nan {int(torch.isnan(obs).sum())} "
      f"absmax {float(obs.abs().max()):.3f}")
print(f"act {tuple(act.shape)} nan {int(torch.isnan(act).sum())} "
      f"absmax {float(act.abs().max()):.3f}")
ag = TrainingAgent(cfg)
emb = ag.encoder({"state": obs}, None)
print(f"emb {tuple(emb.shape)} nan {int(torch.isnan(emb).sum())} "
      f"absmax {float(emb.abs().max()):.3f}")
z = torch.zeros_like(act)
t0 = torch.zeros(len(act), device=dev)
out, s_raw = ag.flow_map.net(z, t0, t0, emb)
print(f"net_out {tuple(out.shape)} nan {int(torch.isnan(out).sum())} "
      f"absmax {float(out.abs().max()):.3f}")
print(f"s_raw nan {int(torch.isnan(s_raw).sum())}")
# param check
nn_bad = [(n, int(torch.isnan(p).sum())) for n, p in ag.flow_map.named_parameters() if torch.isnan(p).any()]
print("params with NaN:", nn_bad[:5], "count", len(nn_bad))
en_bad = [(n, int(torch.isnan(p).sum())) for n, p in ag.encoder.named_parameters() if torch.isnan(p).any()]
print("encoder params with NaN:", en_bad[:5], "count", len(en_bad))
