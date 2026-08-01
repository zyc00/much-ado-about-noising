"""Within-chunk crowding probe (H=16): inside junction-spanning chunks, do the
large-amplitude (stroke) steps dominate the chunk's L2 loss so the quiet steps
are second-class within the target vector? Teacher-forced per-step residuals on
the trained model, comparing QUIET steps at matched r that appear (a) inside
mixed junction chunks vs (b) inside pure-quiet chunks. Chunks/targets come from
the dataset pipeline (correct rot6d transform + normalization); raw |a_pos| per
step from the hdf5 for masks. Chunk classes by (start r, content): PUREQ =
start r in [-12,-8), no step |a_pos|>0.5; MIXED = start r in [-4,6), >=3 steps
|a_pos|>0.5; STROKE = start r in [14,30). Envs: CKPT, LOSS, TAG."""
import os

os.environ["MUJOCO_GL"] = "egl"
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import h5py
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=data/tool_hang_full2ins_2000.hdf5", "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
HZ = int(os.environ.get("HZ", "16"))
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = HZ
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device

h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
NDEMO = 200
c1s, amag = [], []
for k in keys[:NDEMO]:
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    g = a[:, 6]
    cl = [t for t in range(1, len(a)) if g[t - 1] < 0 and g[t] >= 0]
    c1s.append(cl[0] if cl else -10**6)
    amag.append(np.linalg.norm(a[:, :3], axis=1))
h.close()

ee = ds.sampler.replay_buffer.episode_ends[:]
starts = np.concatenate([[0], ee[:-1]])
idxs = ds.sampler.indices
CH = {"PUREQ": [], "MIXED": [], "STROKE": [], "ANY": []}
for i in range(len(idxs)):
    b0, b1, s0, s1 = idxs[i]
    d = int(np.searchsorted(ee, b0, side="right"))
    if d >= NDEMO: break
    t = int(b0 - starts[d] + (1 - s0))
    r = t - c1s[d]
    if not (-12 <= r < 30): continue
    am = amag[d][t:t + HZ]
    if len(am) < HZ:
        am = np.pad(am, (0, HZ - len(am)), "edge")
    nbig = int((am > 0.5).sum())
    CH["ANY"].append((i, r, am))
    if -12 <= r < -8 and nbig == 0:
        CH["PUREQ"].append((i, r, am))
    elif -4 <= r < 6 and nbig >= 3:
        CH["MIXED"].append((i, r, am))
    elif 14 <= r < 30:
        CH["STROKE"].append((i, r, am))
rng = np.random.RandomState(0)
for c in CH:
    if len(CH[c]) > 600:
        CH[c] = [CH[c][j] for j in rng.choice(len(CH[c]), 600, replace=False)]

ag.load(os.environ["CKPT"], load_optimizer=False)
ag.encoder.eval(); ag.flow_map.eval()
TAG = os.environ.get("TAG", "ckpt")

def perstep_resid(entries):
    R, QM = [], []
    with torch.no_grad():
        for b0_ in range(0, len(entries), 256):
            bb = entries[b0_:b0_ + 256]
            xs, ys = [], []
            for i, r, am in bb:
                b = ds[int(i)]
                o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
                xs.append(o[:2] if o.shape[0] > 2 else o)
                ys.append(b["action"])
            wb = torch.stack(xs).to(dev); yb = torch.stack(ys).to(dev)
            emb = ag.encoder(wb, None)
            t0 = torch.zeros(len(wb), device=dev)
            pred = ag.flow_map.get_velocity(t0, torch.zeros_like(yb), emb)
            R.append(((pred - yb) ** 2).sum(dim=2).cpu().numpy())
            QM.append(np.stack([e[2] for e in bb]))
    return np.concatenate(R), np.concatenate(QM)

RES = {}
for c in ["PUREQ", "MIXED", "STROKE", "ANY"]:
    if not CH[c]:
        print(f"CHUNKGRAD {TAG} {c}: empty"); continue
    R, QM = perstep_resid(CH[c])
    RES[c] = R
    quiet = QM < 0.05; big = QM > 0.5
    tot = R.sum()
    print(f"CHUNKGRAD {TAG} {c} n={len(R)} | chunk-loss p50={np.median(R.sum(1)):.5f} | "
          f"quiet-step resid p50={np.median(R[quiet]) if quiet.any() else float('nan'):.5f} "
          f"big-step resid p50={np.median(R[big]) if big.any() else float('nan'):.5f} | "
          f"share of chunk loss on big steps={R[big].sum()/tot if big.any() else 0:.0%} "
          f"(big steps are {big.mean():.0%} of steps)", flush=True)

def matched_quiet(entries, R):
    vals = []
    for i_, (i, r0, am) in enumerate(entries):
        for j in range(HZ):
            if -4 <= r0 + j < 2 and am[j] < 0.05:
                vals.append(R[i_, j])
    return np.array(vals)

vp = matched_quiet(CH["PUREQ"], RES["PUREQ"]); vm = matched_quiet(CH["MIXED"], RES["MIXED"])
print(f"CHUNKGRAD {TAG} MATCHED quiet steps r in [-4,2): resid p50 pure-chunk={np.median(vp):.5f} "
      f"(n={len(vp)}) vs mixed-chunk={np.median(vm):.5f} (n={len(vm)}) ratio={np.median(vm)/max(np.median(vp),1e-9):.2f}", flush=True)
print("CHUNKGRAD-DONE")
