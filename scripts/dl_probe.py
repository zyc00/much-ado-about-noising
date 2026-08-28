"""Profile the image dataloading path stage by stage on one pod.

Usage: python scripts/dl_probe.py <staged_hdf5_path> <task_yaml_name>
"""
import cProfile
import io
import pstats
import sys
import time

import numpy as np
import torch
import zarr
from hydra import compose, initialize

from mip.datasets.robomimic_dataset import make_dataset

DPATH, TASK = sys.argv[1], sys.argv[2]

with initialize(version_base=None, config_path="../examples/configs"):
    cfg = compose(
        config_name="main",
        overrides=[f"task={TASK}", f"+task.dataset_path={DPATH}"],
    )

print("zarr", zarr.__version__, "| torch", torch.__version__, flush=True)

t0 = time.time()
ds = make_dataset(cfg.task, mode="train")
print(f"A. dataset init: {time.time() - t0:.1f}s, len={len(ds)}", flush=True)

rb = ds.replay_buffer
ik = ds.rgb_keys[0]
lk = ds.lowdim_keys[0]

# B. zarr micro-latency per access
za = rb[ik]
zl = rb[lk]
t0 = time.time()
for i in range(300):
    _ = za[i % 1000]
img_us = (time.time() - t0) / 300 * 1e6
t0 = time.time()
for i in range(300):
    _ = zl[i % 1000 : i % 1000 + 16]
low_us = (time.time() - t0) / 300 * 1e6
print(f"B. zarr get: image chunk {img_us:.0f}us | lowdim 16-slice {low_us:.0f}us",
      flush=True)

# C. single-process __getitem__
for _ in range(5):
    _ = ds[0]
t0 = time.time()
N = 100
for i in range(N):
    _ = ds[i * 37 % len(ds)]
per = (time.time() - t0) / N * 1e3
print(f"C. __getitem__: {per:.2f} ms/sample -> {per * 1024:.0f} ms/1024-batch "
      f"single-thread", flush=True)

pr = cProfile.Profile()
pr.enable()
for i in range(50):
    _ = ds[i * 53 % len(ds)]
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
print("C2. profile top:", flush=True)
for line in s.getvalue().splitlines():
    if any(k in line for k in ("robomimic_dataset", "sampler", "zarr",
                                "normalize", "tensor", "{built-in")):
        print("   ", line.strip()[:150], flush=True)

# D. DataLoader steady-state, current path
def rate(dataset, n_batches=12, bs=1024, nw=12):
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=bs, num_workers=nw, shuffle=True,
        pin_memory=True, persistent_workers=True, drop_last=True,
        prefetch_factor=2,
    )
    it = iter(dl)
    next(it)  # warmup incl worker spawn
    next(it)
    t0 = time.time()
    for _ in range(n_batches):
        next(it)
    dt = (time.time() - t0) / n_batches
    del it, dl
    return dt

dt = rate(ds)
print(f"D. DataLoader(current): {dt * 1e3:.0f} ms/batch = {1 / dt:.2f} steps/s",
      flush=True)

# E. uint8-until-GPU variant: skip float cast + normalize in worker
class U8(torch.utils.data.Dataset):
    def __init__(self, base):
        self.b = base

    def __len__(self):
        return len(self.b)

    def __getitem__(self, idx):
        sample = self.b.sampler.sample_sequence(idx)
        T = slice(self.b.n_obs_steps)
        out = {}
        for key in self.b.rgb_keys:
            out[key] = torch.from_numpy(
                np.ascontiguousarray(np.moveaxis(sample[key][T], -1, 1))
            )  # uint8 T,C,H,W
        for key in self.b.lowdim_keys:
            out[key] = torch.from_numpy(
                self.b.normalizer["obs"][key]
                .normalize(sample[key][T].astype(np.float32))
            )
        out["action"] = torch.from_numpy(
            self.b.normalizer["action"].normalize(
                sample["action"].astype(np.float32)
            )
        )
        return out

dt8 = rate(U8(ds))
print(f"E. DataLoader(uint8-to-GPU): {dt8 * 1e3:.0f} ms/batch = "
      f"{1 / dt8:.2f} steps/s", flush=True)
print(f"SPEEDUP {dt / dt8:.2f}x", flush=True)
