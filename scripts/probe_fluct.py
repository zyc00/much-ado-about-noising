"""Valid T5 (OU fluctuation-dissipation) probe: resume a converged checkpoint
and train at CONSTANT lr with the arm's own loss; record LIVE-net outputs at
probe states every FL_DUMP steps; jitter^2(x) over dumps vs data-stiffness
kappa(x) -> predicted log-log slope -1 (L2); HG additionally vs kappa/sigma^2.
Env: TC_DATASET, FL_ARM name:loss:ckpt, FL_STEPS(20000), FL_LR(1e-4).
Prints: FLUCT lines.
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
from mip.losses import get_loss_fn

DATASET = os.environ["TC_DATASET"]
NAME, LOSS, CKPT = os.environ["FL_ARM"].split(":")
STEPS = int(os.environ.get("FL_STEPS", "20000"))
LR = float(os.environ.get("FL_LR", "1e-4"))
DUMP = int(os.environ.get("FL_DUMP", "2000"))
NST = 60
K = 32


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


cfg, ds, ag = load(LOSS)
rb = ds.replay_buffer
ends = rb.episode_ends[:]
obs_entry = rb["obs"]
try:
    S_all = obs_entry["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_entry[:]
A_all = rb["action"][:]
H = int(cfg.task.horizon)
AD = A_all.shape[1]
no = ds.normalizer["obs"]["state"]
na = ds.normalizer["action"]
dev = cfg.optimization.device

starts = np.concatenate([[0], ends[:-1]])
W_list, Y_list = [], []
for e in range(min(300, len(ends))):
    s0, e0 = int(starts[e]), int(ends[e])
    T = e0 - s0
    if T < H + 3:
        continue
    S, A = S_all[s0:e0], A_all[s0:e0]
    for i in range(1, T - H, 2):
        W_list.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        Y_list.append(na.normalize(A[i:i + H]).reshape(-1))
WN = np.stack(W_list).astype(np.float64)
YN = np.stack(Y_list).astype(np.float64)
M = len(WN)

from scipy.spatial import cKDTree
tree = cKDTree(WN)
rng = np.random.RandomState(0)
an = np.linalg.norm(YN[:, :AD], axis=1)
quiet = np.argsort(an)[:M // 10]
IDX = np.concatenate([rng.choice(M, NST // 2, replace=False),
                      rng.choice(quiet, NST - NST // 2, replace=False)])
kap = np.zeros(NST)
for j, i in enumerate(IDX):
    _, nb = tree.query(WN[i], k=K + 1)
    nb = nb[1:]
    dX = WN[nb] - WN[i]
    dY = YN[nb] - YN[i]
    G = dX.T @ dX + 1e-6 * np.eye(dX.shape[1])
    Wl = np.linalg.solve(G, dX.T @ dY)
    kap[j] = float(((dX @ Wl) ** 2).mean())
XT = torch.tensor(WN[IDX], device=dev, dtype=torch.float32)

# full training tensors for batch sampling
WT = torch.tensor(WN, device=dev, dtype=torch.float32)
YT = torch.tensor(YN, device=dev, dtype=torch.float32).reshape(M, H, AD)

ag.load(CKPT, load_optimizer=False)
loss_fn = get_loss_fn(LOSS)
params = list(ag.flow_map.parameters()) + list(ag.encoder.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-5)
brng = np.random.RandomState(7)


def probe_out():
    t = torch.zeros(len(XT), device=dev)
    act0 = torch.zeros(len(XT), H, AD, device=dev)
    with torch.no_grad():
        emb = ag.encoder({"state": XT.reshape(-1, 2, 53)}, None)
        pred, sraw = ag.flow_map.net(act0, t, t, emb)
        sig = torch.nn.functional.softplus(sraw).reshape(len(XT), -1) \
            .mean(1).cpu().numpy() + 1e-3
    return pred.reshape(len(XT), -1).cpu().numpy(), sig


dumps = []
sig = None
for it in range(STEPS + 1):
    idx = torch.tensor(brng.randint(0, M, 256), device=dev)
    obs_b = {"state": WT[idx].reshape(-1, 2, 53)}
    dt = torch.zeros(len(idx), device=dev)
    loss, _ = loss_fn(cfg.optimization, ag.flow_map, ag.encoder,
                      ag.interpolant, YT[idx], obs_b, dt)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 10.0)
    opt.step()
    if it % DUMP == 0 and it > 0:
        o, sig = probe_out()
        dumps.append(o)
        print(f"FLUCT {NAME} dump {it} loss {float(loss):.3e}", flush=True)

O = np.stack(dumps[2:])          # discard warm-up dumps
mu = O.mean(0)
jit = ((O - mu) ** 2).mean(axis=(0, 2))
m = (kap > 1e-9) & (jit > 1e-16)


def fit(lx, ly):
    A = np.stack([lx, np.ones_like(lx)], 1)
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return sol[0], np.corrcoef(lx, ly)[0, 1]


s5, r5 = fit(np.log(kap[m]), np.log(jit[m]))
print(f"FLUCT {NAME} T5_OU slope {s5:+.2f} r {r5:+.2f} (pred -1)", flush=True)
if LOSS.startswith("regression_hetero"):
    s5b, r5b = fit(np.log(kap[m] / sig[m] ** 2), np.log(jit[m]))
    print(f"FLUCT {NAME} T5_repriced slope {s5b:+.2f} r {r5b:+.2f}",
          flush=True)
print("FLUCT done", flush=True)
