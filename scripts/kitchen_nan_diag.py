"""Trace kitchen NaN: batch -> encoder -> loss."""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, ".")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=kitchen_state", "network=chiunet",
        "+task.dataset_path=/mnt/pfs/yuchen/data/kitchen_raw/kitchen",
        "~task.dataset_repo", "optimization.loss_type=regression_hetero_t",
        "log.wandb_mode=disabled", "optimization.auto_resume=false"])
OmegaConf.set_struct(cfg, False)
from mip.datasets.kitchen_dataset import make_dataset
from mip.agent import TrainingAgent

ds = make_dataset(cfg.task)
dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
b = next(iter(dl))


def rep(name, t):
    t = t.float()
    print(f"  {name}: shape {tuple(t.shape)} nan {int(torch.isnan(t).sum())} "
          f"inf {int(torch.isinf(t).sum())} "
          f"min {float(t.min()):.3f} max {float(t.max()):.3f}")


print("BATCH keys:", list(b.keys()) if hasattr(b, "keys") else type(b))
for k in (b.keys() if hasattr(b, "keys") else []):
    v = b[k]
    if torch.is_tensor(v):
        rep(k, v)
    elif hasattr(v, "keys"):
        for k2 in v.keys():
            rep(f"{k}.{k2}", v[k2])

dev = cfg.optimization.device
ag = TrainingAgent(cfg)
obs = {k: v.to(dev).float() for k, v in b["obs"].items()} \
    if hasattr(b["obs"], "keys") else {"state": b["obs"].to(dev).float()}
act = b["action"].to(dev).float()
dt = torch.zeros(len(act), device=dev)
emb = ag.encoder(obs, None)
rep("ENCODER_OUT", emb)
loss, _ = ag.loss_fn(cfg.optimization, ag.flow_map, ag.encoder,
                     ag.interpolant, act, obs, dt)
print(f"  LOSS {float(loss)}")
for n, p in list(ag.flow_map.named_parameters())[:3]:
    rep(f"param.{n}", p.data)
