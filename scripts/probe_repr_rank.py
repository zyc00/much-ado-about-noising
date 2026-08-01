"""WHERE does the L2-vs-MIP rank difference live? Global representation
probes that do not assume per-state locality (the shared network serves all
phases, so per-phase data statistics cannot explain a per-state Jacobian).

Measured over a large common batch of states (same states, same normalizer
for every arm):
  emb_PR   : participation ratio of the EMBEDDING covariance spectrum
             (how many representation directions are actually populated)
  emb_k90  : dims for 90% of embedding variance
  W_srank  : stable rank ||W||_F^2/||W||_2^2 of each encoder weight matrix
             (a layer-wise, data-free measure of learned rank)
  act_dead : fraction of hidden units with near-zero activation variance
  Jmean_PR : PR of the AVERAGE Jacobian (mean over states of J^T J) —
             the global input-sensitivity rank, as opposed to per-state
Also reports the DEMANDED rank from the data (local linear map) for scale.
Env: RR_ARMS, RR_DATASET, RR_N. Prints RREP lines.
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

DSET = os.environ.get("RR_DATASET", "data/tool_hang_full2ins_mp_200.hdf5")
NST = int(os.environ.get("RR_N", "1024"))
NJ = int(os.environ.get("RR_NJ", "48"))
ARMS = [tuple(a.split(":")) for a in os.environ["RR_ARMS"].split(",")]


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
obs_entry = rb["obs"]
try:
    S_all = obs_entry["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_entry[:]
H = int(cfg0.task.horizon)
no = ds0.normalizer["obs"]["state"]
dev = cfg0.optimization.device
starts = np.concatenate([[0], ends[:-1]])
Wl = []
for e in range(len(ends)):
    s0, e0 = int(starts[e]), int(ends[e])
    if e0 - s0 < H + 3:
        continue
    S = S_all[s0:e0]
    for i in range(1, e0 - s0 - H, 5):
        Wl.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
WN = np.stack(Wl)
sel = np.random.RandomState(0).choice(len(WN), min(NST, len(WN)), replace=False)
X = torch.tensor(WN[sel], device=dev, dtype=torch.float32)
print(f"states {len(X)} of {len(WN)}", flush=True)


def pr(ev):
    ev = np.asarray(ev, dtype=np.float64)
    return float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30))


for name, loss, ck in ARMS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    with torch.no_grad():
        E = ag.encoder_ema({"state": X.reshape(-1, 2, 53)}, None)
    E = E.reshape(len(X), -1)
    Ec = (E - E.mean(0)).double()
    ev = torch.linalg.svdvals(Ec).cpu().numpy() ** 2
    cum = np.cumsum(ev) / ev.sum()
    k90 = int((cum < 0.90).sum()) + 1
    dead = float((E.std(0) < 1e-3 * E.std()).float().mean())
    # layer-wise stable rank of encoder weights
    sr = []
    for n_, p_ in ag.encoder_ema.named_parameters():
        if p_.dim() == 2 and min(p_.shape) > 4:
            s = torch.linalg.svdvals(p_.double())
            sr.append(float((s ** 2).sum() / (s[0] ** 2 + 1e-30)))
    # global (state-averaged) Jacobian rank
    G = torch.zeros(X.shape[1], X.shape[1], device=dev, dtype=torch.float64)
    for i in range(NJ):
        xi = X[i]

        def f(inp):
            return ag.encoder_ema({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)

        J = torch.autograd.functional.jacobian(f, xi, vectorize=True)
        J = J.reshape(-1, X.shape[1]).double()
        G += J.T @ J
    evg = torch.linalg.svdvals(G).cpu().numpy()
    print(f"RREP {name} emb_dim {E.shape[1]} emb_PR {pr(ev):.2f} emb_k90 {k90} "
          f"dead {dead:.3f} W_srank {[round(s,1) for s in sr]} "
          f"Jmean_PR {pr(evg):.2f}", flush=True)
print("RREP done", flush=True)
