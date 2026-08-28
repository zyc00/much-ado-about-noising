"""Clean dataloader variant benchmark: steady-state wall-clock windows,
hard worker teardown between variants.

Usage: python scripts/dl_probe2.py <staged_hdf5> <task_yaml>
"""
import gc
import multiprocessing as mp
import sys
import time

import numpy as np
import torch
from hydra import compose, initialize

from mip.datasets.robomimic_dataset import make_dataset

DPATH, TASK = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)

with initialize(version_base=None, config_path="../examples/configs"):
    cfg = compose(
        config_name="main",
        overrides=[f"task={TASK}", f"+task.dataset_path={DPATH}"],
    )

t0 = time.time()
ds = make_dataset(cfg.task, mode="train")
print(f"init {time.time() - t0:.0f}s len={len(ds)}", flush=True)

rb = ds.replay_buffer


def teardown(dl, it):
    del it
    del dl
    gc.collect()
    for _ in range(40):
        if not mp.active_children():
            break
        time.sleep(0.5)


def bench(dataset, name, bs=1024, nw=12, batch_none=False):
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=None if batch_none else bs,
        num_workers=nw,
        shuffle=not batch_none,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False if batch_none else True,
        prefetch_factor=2,
    )
    it = iter(dl)
    t_end = time.time() + 40
    n_all = 0
    n_warm = 0
    t_warm = None
    while time.time() < t_end:
        try:
            _ = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        n_all += 1
        if time.time() > t_end - 25:
            if t_warm is None:
                t_warm = time.time()
                n_warm = 0
            n_warm += 1
    dt = (time.time() - t_warm) / max(n_warm, 1)
    print(f"{name}: {dt * 1e3:.0f} ms/batch = {1 / dt:.2f} steps/s "
          f"({n_warm} timed)", flush=True)
    teardown(dl, it)
    return dt


# V2/V3 batch-assembly dataset: one __getitem__ builds a whole batch
class BatchDS(torch.utils.data.Dataset):
    def __init__(self, base, bs, uint8, n_batches=400):
        self.b = base
        self.bs = bs
        self.uint8 = uint8
        self.n = n_batches
        s = base.sampler
        self.idx_pool = len(s)

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        rng = np.random.default_rng()
        idxs = rng.integers(0, self.idx_pool, size=self.bs)
        b = self.b
        s = b.sampler
        T = b.n_obs_steps
        out_obs = {}
        first = [s.sample_sequence(int(i)) for i in idxs]  # per-sample gather
        for key in b.rgb_keys:
            arr = np.stack([f[key][:T] for f in first])  # B,T,H,W,C uint8
            arr = np.moveaxis(arr, -1, 2)  # B,T,C,H,W
            if self.uint8:
                out_obs[key] = torch.from_numpy(np.ascontiguousarray(arr))
            else:
                x = np.ascontiguousarray(arr).astype(np.float32) / 255.0
                x = b.normalizer["obs"][key].normalize(x)
                out_obs[key] = torch.from_numpy(x)
        for key in b.lowdim_keys:
            x = np.stack([f[key][:T] for f in first]).astype(np.float32)
            out_obs[key] = torch.from_numpy(
                b.normalizer["obs"][key].normalize(x))
        act = np.stack([f["action"] for f in first]).astype(np.float32)
        action = torch.from_numpy(b.normalizer["action"].normalize(act))
        return {"obs": out_obs, "action": action}


# V1 per-sample uint8
class U8(torch.utils.data.Dataset):
    def __init__(self, base):
        self.b = base

    def __len__(self):
        return len(self.b)

    def __getitem__(self, idx):
        b = self.b
        sample = b.sampler.sample_sequence(idx)
        T = slice(b.n_obs_steps)
        out = {}
        for key in b.rgb_keys:
            out[key] = torch.from_numpy(
                np.ascontiguousarray(np.moveaxis(sample[key][T], -1, 1)))
        for key in b.lowdim_keys:
            out[key] = torch.from_numpy(
                b.normalizer["obs"][key].normalize(
                    sample[key][T].astype(np.float32)))
        out["action"] = torch.from_numpy(
            b.normalizer["action"].normalize(
                sample["action"].astype(np.float32)))
        return out




# V4: numpy-materialized batch assembly (no zarr in the hot path)
class NpBatchDS(torch.utils.data.Dataset):
    def __init__(self, base, bs, uint8, n_batches=4000):
        b = base
        self.bs = bs
        self.uint8 = uint8
        self.n = n_batches
        self.T = b.n_obs_steps
        self.H = b.horizon
        self.rgb = {k: b.replay_buffer[k][:] for k in b.rgb_keys}  # N,H,W,C u8
        self.low = {k: b.replay_buffer[k][:].astype(np.float32)
                    for k in b.lowdim_keys}
        self.act = b.replay_buffer["action"][:].astype(np.float32)
        self.norm = b.normalizer
        ee = b.replay_buffer.episode_ends[:]
        starts = np.concatenate([[0], ee[:-1]])
        ok = []
        for s0, e0 in zip(starts, ee):
            if e0 - s0 >= self.H:
                ok.append(np.arange(s0, e0 - self.H + 1))
        self.valid = np.concatenate(ok)

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        rng = np.random.default_rng()
        base_idx = self.valid[rng.integers(0, len(self.valid), size=self.bs)]
        obs_idx = base_idx[:, None] + np.arange(self.T)[None]   # B,T
        act_idx = base_idx[:, None] + np.arange(self.H)[None]   # B,Hz
        out_obs = {}
        for k, arr in self.rgb.items():
            g = arr[obs_idx]                       # B,T,H,W,C uint8 fancy gather
            g = np.moveaxis(g, -1, 2)              # B,T,C,H,W
            if self.uint8:
                out_obs[k] = torch.from_numpy(np.ascontiguousarray(g))
            else:
                x = np.ascontiguousarray(g).astype(np.float32) / 255.0
                out_obs[k] = torch.from_numpy(self.norm["obs"][k].normalize(x))
        for k, arr in self.low.items():
            out_obs[k] = torch.from_numpy(
                self.norm["obs"][k].normalize(arr[obs_idx]))
        action = torch.from_numpy(self.norm["action"].normalize(self.act[act_idx]))
        return {"obs": out_obs, "action": action}

# zarr fancy-gather microbench (sizing for a vectorized sampler)
ik = ds.rgb_keys[0]
za = rb[ik]
rng = np.random.default_rng(0)
sel = np.sort(rng.integers(0, za.shape[0], size=2048))
t0 = time.time()
_ = za[sel]
print(f"zarr fancy gather 2048 frames: {(time.time() - t0) * 1e3:.0f} ms",
      flush=True)

d0 = bench(ds, "V0 current")
d1 = bench(U8(ds), "V1 uint8-per-sample")
d2 = bench(BatchDS(ds, 1024, uint8=False), "V2 zarr-batch-f32",
           batch_none=True)
d4 = bench(NpBatchDS(ds, 1024, uint8=False), "V4 np-batch-f32",
           batch_none=True)
d5 = bench(NpBatchDS(ds, 1024, uint8=True), "V5 np-batch-uint8",
           batch_none=True)
print(f"SPEEDUPS vs V0: V1 {d0/d1:.2f}x V2 {d0/d2:.2f}x "
      f"V4 {d0/d4:.2f}x V5 {d0/d5:.2f}x", flush=True)
