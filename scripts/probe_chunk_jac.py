"""Per-chunk-position Jacobian spectra, human-data policies.
For each arm and state group (settle-core / stroke), compute the full policy
Jacobian J = d(chunk)/d(obs window) at t=0 (16x10 outputs x 106 inputs), slice
per chunk position k (10x106), report PR(s^2), condition s1/s10, Frobenius.
MIP uses its t=0 net map (step-1 anchor map) — caveat: not the 2-step sampler.
Prints: CHUNKJAC <arm> <group> <k> <PR> <cond> <fro>
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


def load(loss):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_human_lowdim_up.hdf5"),
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
A_all = rb["action"][:]
H = int(cfg0.task.horizon)
AD = A_all.shape[1]
no = ds0.normalizer["obs"]["state"]
dev = cfg0.optimization.device

sets, strs = [], []
start = 0
for e in range(min(200, len(ends))):
    end = int(ends[e])
    S, A = S_all[start:end], A_all[start:end]
    T = len(S)
    start = end
    if T < 60:
        continue
    an = np.linalg.norm(A, axis=1)
    lo, hi = T // 3, 2 * T // 3
    settle = lo + int(np.argmin(an[lo:hi]))
    if settle >= 1 and settle + H < T and len(sets) < 12:
        sets.append(np.stack([S[settle - 1], S[settle]]))
    seg = an[settle:min(settle + 40, T - H - 1)]
    if len(seg) >= 6:
        stroke = settle + int(np.argmax(seg))
        if stroke - settle >= 5 and stroke + H < T and len(strs) < 12:
            strs.append(np.stack([S[stroke - 1], S[stroke]]))
print(f"states: settle={len(sets)} stroke={len(strs)}", flush=True)

GROUPS = {"settle": sets, "stroke": strs}
ARMS = [("L2", "regression", "logs/ath_muL2_s1000"),
        ("HG", "regression_hetero_gauss", "logs/ath_hgauss_s1000"),
        ("HT", "regression_hetero_t", "logs/ath_muHT_s1000"),
        ("MIP", "mip", "logs/ath_muMIP_s1000")]

for name, loss, d in ARMS:
    cfg, ds, ag = load(loss)
    ag.load(f"{d}/models/model_latest.pt", load_optimizer=False)
    ag.eval()

    def f(inp):
        emb = ag.encoder_ema({"state": inp.reshape(1, 2, 53)}, None)
        t = torch.zeros(1, device=dev)
        act0 = torch.zeros(1, H, AD, device=dev)
        pred, _ = ag.flow_map_ema.net(act0, t, t, emb)
        return pred.reshape(-1)  # (H*AD,)

    for gname, wins in GROUPS.items():
        stats = np.zeros((H, 3, len(wins)))
        for i, w in enumerate(wins):
            x = torch.tensor(no.normalize(w), device=dev,
                             dtype=torch.float32).reshape(-1)
            J = torch.autograd.functional.jacobian(f, x, vectorize=True)
            J = J.reshape(H, AD, -1)  # (H, 10, 106)
            for k in range(H):
                s = torch.linalg.svdvals(J[k])
                s2 = s ** 2
                pr = float(s2.sum() ** 2 / (s2 ** 2).sum())
                cond = float(s[0] / (s[-1] + 1e-12))
                fro = float(s2.sum().sqrt())
                stats[k, :, i] = (pr, cond, fro)
        for k in range(H):
            pr, cond, fro = stats[k, 0].mean(), np.median(stats[k, 1]), stats[k, 2].mean()
            print(f"CHUNKJAC {name} {gname} {k} {pr:.2f} {cond:.0f} {fro:.3f}",
                  flush=True)
print("CHUNKJAC done", flush=True)
