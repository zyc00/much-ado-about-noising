"""Scan kitchen dataset arrays for NaN/Inf."""
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
rb = getattr(ds, "replay_buffer", None)
if rb is not None:
    for k in rb.keys():
        a = np.asarray(rb[k])
        print(f"RAW {k}: shape {a.shape} nan {int(np.isnan(a).sum())} "
              f"inf {int(np.isinf(a).sum())} min {np.nanmin(a):.3f} "
              f"max {np.nanmax(a):.3f}")
print("--- normalized samples ---")
bad = 0
for i in [0, 1, 100, 5000, 50000, len(ds) - 1]:
    s = ds[i]
    for k in (s.keys() if hasattr(s, "keys") else []):
        v = s[k]
        if hasattr(v, "keys"):
            for k2 in v.keys():
                t = torch.as_tensor(v[k2]).float()
                n = int(torch.isnan(t).sum())
                bad += n
                if n:
                    print(f"  sample{i} {k}.{k2} NaN {n}/{t.numel()}")
        else:
            t = torch.as_tensor(v).float()
            n = int(torch.isnan(t).sum())
            bad += n
            if n:
                print(f"  sample{i} {k} NaN {n}/{t.numel()}")
print("total NaN in sampled entries:", bad)
