"""Is the representation-rank split explained by the NUMBER OF READOUTS the
shared embedding has to serve?

L2 trains one readout of the embedding (the action head at s=t=0).
HG/HT additionally train a sigma head; MIP additionally trains the same
action head at a SECOND conditioning (t=tau, act_t = a + (1-tau)eps).
Hypothesis: the populated embedding subspace is the UNION of the subspaces
its trained readouts actually read, so a second readout roughly doubles it
(measured: L2 emb_PR 3.3 vs HG/HT/MIP 6.8).

Measured per arm, over a common state batch:
  emb_PR      : PR of the embedding covariance (populated directions)
  rd_act      : PR of the accumulated action-head read subspace
                (sum_x  Ja^T Ja  with Ja = d action / d emb at s=t=0)
  rd_2nd      : same for the SECOND readout — sigma head (HG/HT/L2 head is
                untrained but present) or the action head at (tau, act_t)
                for MIP
  ang         : energy of the 2nd read subspace OUTSIDE the top-k action
                subspace (k = ceil(rd_act)) — how much NEW rank it demands
  cov_act     : fraction of embedding variance inside the action subspace
Env: RS_ARMS, RS_N. Prints RSUB lines.
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

DSET = os.environ.get("RS_DATASET", "data/tool_hang_full2ins_mp_200.hdf5")
NST = int(os.environ.get("RS_N", "512"))
NJ = int(os.environ.get("RS_NJ", "48"))
ARMS = [tuple(a.split(":")) for a in os.environ["RS_ARMS"].split(",")]


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
sel = np.random.RandomState(0).choice(len(Wl), NST, replace=False)
X = torch.tensor(Wl[sel], device=dev, dtype=torch.float32)
Y = torch.tensor(Yl[sel], device=dev, dtype=torch.float32)
print(f"states {len(X)}", flush=True)


def pr(ev):
    ev = np.clip(np.asarray(ev, dtype=np.float64), 0, None)
    return float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30))


for name, loss, ck in ARMS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    tau = float(cfg.optimization.t_two_step)
    with torch.no_grad():
        E = ag.encoder_ema({"state": X.reshape(-1, 2, 53)}, None)
    Eflat = E.reshape(len(X), -1)
    Ec = (Eflat - Eflat.mean(0)).double()
    ev_emb = torch.linalg.svdvals(Ec).cpu().numpy() ** 2
    D = Eflat.shape[1]

    def read_gram(second):
        G = torch.zeros(D, D, device=dev, dtype=torch.float64)
        for i in range(NJ):
            e_i = E[i:i + 1].clone().requires_grad_(True)
            if not second:
                t_ = torch.zeros(1, device=dev)
                a0 = torch.zeros(1, H, 10, device=dev)
                out, sc = ag.flow_map_ema.net(a0, t_, t_, e_i)
                tgt = out.reshape(-1)
            elif loss == "mip":
                t_ = torch.full((1,), tau, device=dev)
                s_ = torch.zeros(1, device=dev)
                at = (Y[i:i + 1] + (1 - tau) * torch.randn_like(Y[i:i + 1]))
                out, sc = ag.flow_map_ema.net(at, s_, t_, e_i)
                tgt = out.reshape(-1)
            else:
                t_ = torch.zeros(1, device=dev)
                a0 = torch.zeros(1, H, 10, device=dev)
                out, sc = ag.flow_map_ema.net(a0, t_, t_, e_i)
                tgt = sc.reshape(-1)
            J = torch.autograd.functional.jacobian(
                lambda z: (ag.flow_map_ema.net(
                    (torch.zeros(1, H, 10, device=dev) if (not second or loss != "mip")
                     else (Y[i:i + 1] + (1 - tau) * torch.zeros_like(Y[i:i + 1]))),
                    torch.zeros(1, device=dev),
                    torch.full((1,), tau if (second and loss == "mip") else 0.0,
                               device=dev),
                    z.reshape(e_i.shape))[0 if (not second or loss == "mip") else 1]
                ).reshape(-1), e_i.reshape(-1), vectorize=True)
            J = J.reshape(-1, D).double()
            G += J.T @ J
        return G

    Ga = read_gram(False)
    Gb = read_gram(True)
    ra, rb_ = pr(torch.linalg.svdvals(Ga).cpu().numpy()), pr(
        torch.linalg.svdvals(Gb).cpu().numpy())
    # energy of 2nd readout outside the action subspace
    k = max(int(np.ceil(ra)), 1)
    Ua = torch.linalg.svd(Ga)[0][:, :k]
    res = Gb - Ua @ (Ua.T @ Gb @ Ua) @ Ua.T
    frac_out = float(torch.diagonal(res).sum() / (torch.diagonal(Gb).sum() + 1e-30))
    # embedding variance inside the action subspace
    covE = (Ec.T @ Ec)
    cov_act = float((Ua.T @ covE @ Ua).diagonal().sum() /
                    (covE.diagonal().sum() + 1e-30))
    print(f"RSUB {name} emb_PR {pr(ev_emb):.2f} rd_act {ra:.2f} rd_2nd {rb_:.2f} "
          f"2nd_outside_act {frac_out:.3f} embvar_in_act {cov_act:.3f}",
          flush=True)
print("RSUB done", flush=True)
