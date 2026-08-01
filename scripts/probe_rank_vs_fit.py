"""Is representation rank simply downstream of fit precision?

MIP fits its training set 5x tighter than L2 and carries 2x the embedding
rank. If rank were a function of fit quality alone, all arms would fall on
ONE curve in (train_err, emb_PR). Measured across training snapshots, on a
common state batch with a common normalizer.
Env: RF_ARMS, RF_SNAPS, RF_N. Prints RVF lines.
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

DSET = os.environ.get("RF_DATASET", "data/tool_hang_full2ins_mp_200.hdf5")
NST = int(os.environ.get("RF_N", "512"))
SNAPS = [int(x) for x in os.environ.get(
    "RF_SNAPS", "20000,60000,120000,180000,300000").split(",")]
ARMS = [tuple(a.split(":")) for a in os.environ["RF_ARMS"].split(",")]


def load(loss):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(DSET),
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
obs_e = rb["obs"]
try:
    S_all = obs_e["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_e[:]
A_all = rb["action"][:]
H = int(cfg0.task.horizon)
no, na = ds0.normalizer["obs"]["state"], ds0.normalizer["action"]
dev = cfg0.optimization.device
starts = np.concatenate([[0], ends[:-1]])
Wl, Yl = [], []
for e in range(len(ends)):
    s0, e0 = int(starts[e]), int(ends[e])
    if e0 - s0 < H + 3:
        continue
    S, A = S_all[s0:e0], A_all[s0:e0]
    for i in range(1, e0 - s0 - H, 5):
        Wl.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        Yl.append(na.normalize(A[i:i + H]))
Wl, Yl = np.stack(Wl), np.stack(Yl)
sel = np.random.RandomState(0).choice(len(Wl), min(NST, len(Wl)), replace=False)
X = torch.tensor(Wl[sel], device=dev, dtype=torch.float32)
Y = torch.tensor(Yl[sel], device=dev, dtype=torch.float32)
print(f"states {len(X)}", flush=True)


def pr(ev):
    ev = np.clip(np.asarray(ev, dtype=np.float64), 0, None)
    return float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30))


for name, loss, base in ARMS:
    cfg, ds, ag = load(loss)
    for snap in SNAPS:
        ck = f"{base}/models/snap_{snap}.pt"
        if not os.path.exists(ck):
            print(f"RVF {name} snap {snap} MISSING", flush=True)
            continue
        ag.load(ck, load_optimizer=False)
        ag.eval()
        with torch.no_grad():
            E = ag.encoder_ema({"state": X.reshape(-1, 2, 53)}, None)
            t0 = torch.zeros(len(X), device=dev)
            a0 = torch.zeros(len(X), H, 10, device=dev)
            pa, _ = ag.flow_map_ema.net(a0, t0, t0, E)
            err = float((pa - Y).abs().mean())
        Ef = E.reshape(len(X), -1)
        Ec = (Ef - Ef.mean(0)).double()
        ev = torch.linalg.svdvals(Ec).cpu().numpy() ** 2
        sr = []
        for n_, p_ in ag.encoder_ema.named_parameters():
            if p_.dim() == 2 and min(p_.shape) > 4:
                s_ = torch.linalg.svdvals(p_.double())
                sr.append(round(float((s_ ** 2).sum() / (s_[0] ** 2 + 1e-30)), 1))
        print(f"RVF {name} snap {snap} emb_PR {pr(ev):.2f} "
              f"train_err {err:.5f} W_srank {sr}", flush=True)
print("RVF done", flush=True)
