"""Scan the real training DataLoader for NaN batches."""
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
        "~task.dataset_repo", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
from mip.datasets.kitchen_dataset import make_dataset

ds = make_dataset(cfg.task)
print("len", len(ds), "batch_size", cfg.optimization.batch_size)
dl = torch.utils.data.DataLoader(ds, batch_size=cfg.optimization.batch_size,
                                 shuffle=True, num_workers=0)
bad_o = bad_a = 0
for i, b in enumerate(dl):
    o, a = b["obs"]["state"], b["action"]
    no, na = int(torch.isnan(o).sum()), int(torch.isnan(a).sum())
    bad_o += no
    bad_a += na
    if (no or na) and bad_o + bad_a == no + na:
        print(f"FIRST NaN batch {i}: obs {no}/{o.numel()} act {na}/{a.numel()}")
    if i >= 40:
        break
print(f"scanned 41 batches: obs NaN {bad_o}, act NaN {bad_a}")
# full-dataset element scan (fast, numpy)
rb = ds.replay_buffer
ends = np.asarray(rb.episode_ends[:])
print("episodes", len(ends), "min len",
      int(np.min(np.diff(np.concatenate([[0], ends])))),
      "seq_len needed", cfg.task.horizon)
