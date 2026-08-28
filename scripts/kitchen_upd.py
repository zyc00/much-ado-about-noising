"""Call agent.update exactly as train_kitchen does; find the NaN."""
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
ag = TrainingAgent(cfg)

# 1) direct loss_fn call (no update machinery)
loss, _ = ag.loss_fn(cfg.optimization, ag.flow_map, ag.encoder,
                     ag.interpolant, act, obs, torch.zeros(len(act), device=dev))
print("direct loss_fn:", float(loss))

# 2) exact update path, several delta_t values
for dtv in [0.0, 1.0]:
    dt = torch.full((len(act),), dtv, device=dev)
    info = ag.update(act, {"state": obs} if not isinstance(obs, dict) else obs, dt)
    v = {k: (float(x) if torch.is_tensor(x) else x) for k, x in
         (info.items() if hasattr(info, "items") else [])}
    print(f"update(delta_t={dtv}):", {k: v[k] for k in list(v)[:4]})

print("loss_scale", cfg.optimization.loss_scale,
      "norm_type", getattr(cfg.optimization, "norm_type", None),
      "grad_clip", cfg.optimization.grad_clip_norm)
