"""Verify MIP_FAST_IMG batch path == per-sample path numerically.

Run with MIP_FAST_IMG=1. Compares _get_batch output against the
per-sample __getitem__ + default_collate on identical indices,
including episode-boundary (padded) windows.
"""
import sys

import numpy as np
import torch
from hydra import compose, initialize
from torch.utils.data import default_collate

from mip.datasets.robomimic_dataset import make_dataset

DPATH, TASK = sys.argv[1], sys.argv[2]

with initialize(version_base=None, config_path="../examples/configs"):
    cfg = compose(
        config_name="main",
        overrides=[f"task={TASK}", f"+task.dataset_path={DPATH}"],
    )

import os

ds = make_dataset(cfg.task, mode="train")
assert ds._fast, "MIP_FAST_IMG=1 required"
os.environ["MIP_FAST_IMG"] = "0"
ds_slow = make_dataset(cfg.task, mode="train")  # independent slow instance

inds = np.asarray(ds.sampler.indices)
n = len(ds)
# padded windows: sample_start>0 or sample_end<horizon
padded = np.where((inds[:, 2] > 0) | (inds[:, 3] < ds.horizon))[0]
rng = np.random.default_rng(0)
idx = np.concatenate([
    rng.integers(0, n, 40),
    rng.choice(padded, 24, replace=False),
]).tolist()

fast = ds._get_batch(idx)
slow = default_collate([ds_slow[int(i)] for i in idx])

worst = 0.0
for k in ds.rgb_keys:
    f = fast["obs"][k].float().mul(2.0 / 255.0).sub(1.0)
    s = slow["obs"][k]
    d = (f - s).abs().max().item()
    worst = max(worst, d)
    print(f"rgb {k}: shape {tuple(f.shape)} maxdiff {d:.2e}")
for k in ds.lowdim_keys:
    d = (fast["obs"][k] - slow["obs"][k]).abs().max().item()
    worst = max(worst, d)
    print(f"low {k}: maxdiff {d:.2e}")
d = (fast["action"] - slow["action"]).abs().max().item()
worst = max(worst, d)
print(f"action: maxdiff {d:.2e}")
print(f"EQUIVALENCE {'PASS' if worst < 1e-5 else 'FAIL'} worst {worst:.2e}")
