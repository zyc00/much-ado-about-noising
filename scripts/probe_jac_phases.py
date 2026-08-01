"""Phase-resolved encoder-Jacobian spectra: states sampled across normalized
episode time (deciles = phase proxy for monotone f2i episodes) + pooled.
Per decile: PR(s^2), k90, smax/smed, ||J||_F (means/medians over 12 states).
Env: JP_DATASET, JP_ARMS ("name:loss:logdir,..." — model_latest used).
Prints: JACPHASE <arm> <decile|all> <PR> <k90> <ratio> <fro>
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

DATASET = os.environ["JP_DATASET"]
ARMS = [tuple(a.split(":")) for a in os.environ["JP_ARMS"].split(",")]


def load(loss):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(DATASET),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)


cfg0, ds0, _ = load("regression")
rb = ds0.replay_buffer
ends = rb.episode_ends[:]
obs_entry = rb["obs"]
try:
    S_all = obs_entry["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_entry[:]
no = ds0.normalizer["obs"]["state"]
dev = cfg0.optimization.device

# 12 states per decile: episode e's step at frac f
rng = np.random.RandomState(0)
starts = np.concatenate([[0], ends[:-1]])
DEC = {d: [] for d in range(10)}
for d in range(10):
    while len(DEC[d]) < 12:
        e = rng.randint(0, min(150, len(ends)))
        s0, e0 = int(starts[e]), int(ends[e])
        T = e0 - s0
        if T < 40:
            continue
        i = s0 + 1 + int((T - 3) * (d + rng.rand()) / 10)
        DEC[d].append(np.stack([S_all[i - 1], S_all[i]]))
print("states: 12 x 10 deciles", flush=True)


def spec(ag, w):
    x = torch.tensor(no.normalize(w), device=dev,
                     dtype=torch.float32).reshape(-1)

    def f(inp):
        return ag.encoder_ema({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)

    J = torch.autograd.functional.jacobian(f, x, vectorize=True)
    J = J.reshape(J.shape[0], -1)
    s = torch.linalg.svdvals(J)
    s2 = s ** 2
    pr = float(s2.sum() ** 2 / (s2 ** 2).sum())
    c = torch.cumsum(s2, 0) / s2.sum()
    k90 = int((c < 0.90).sum()) + 1
    rat = float(s[0] / (s[len(s) // 2] + 1e-12))
    return pr, k90, rat, float(s2.sum().sqrt())


for name, loss, d in ARMS:
    cfg, ds, ag = load(loss)
    ag.load(f"{d}/models/model_latest.pt", load_optimizer=False)
    ag.eval()
    pooled = []
    for dec in range(10):
        vals = np.array([spec(ag, w) for w in DEC[dec]])
        pooled.append(vals)
        print(f"JACPHASE {name} {dec} {vals[:,0].mean():.2f} "
              f"{vals[:,1].mean():.1f} {np.median(vals[:,2]):.1f} "
              f"{vals[:,3].mean():.2f}", flush=True)
    allv = np.concatenate(pooled)
    print(f"JACPHASE {name} all {allv[:,0].mean():.2f} {allv[:,1].mean():.1f} "
          f"{np.median(allv[:,2]):.1f} {allv[:,3].mean():.2f}", flush=True)
print("JACPHASE done", flush=True)
