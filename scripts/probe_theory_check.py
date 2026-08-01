"""Quantitative theory checks on scripted data (jacobian_supervision_note):
T2 sigma-stationarity: log sigma^2(x) vs log measured residual -> slope 1.
T5 OU relation: log snapshot-jitter^2(x) vs log data-stiffness kappa(x)
   -> slope -1 for L2; HG collapse vs kappa/sigma^2.
T1 stiffness->survival: corr(kappa(x), retained encoder-Jacobian energy).
T4 MIP endpoint identity: ||d f/d act_t||_F ~ Cov[y|x,y_t]/(1-tau)^2:
   near-zero on MP data, structured on pointing data.
Env: TC_DATASET, TC_ARMS "name:loss:dir,...". Prints THEORY lines.
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

DATASET = os.environ["TC_DATASET"]
ARMS = [tuple(a.split(":")) for a in os.environ["TC_ARMS"].split(",")]
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
na = ds0.normalizer["action"]
dev = cfg0.optimization.device

# ---- build window/label arrays (normalized spaces) ------------------------
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
WN = np.stack(W_list).astype(np.float64)   # (M, 106)
YN = np.stack(Y_list).astype(np.float64)   # (M, H*AD)
M = len(WN)
print(f"pairs {M}", flush=True)

from scipy.spatial import cKDTree
tree = cKDTree(WN)
rng = np.random.RandomState(0)
an = np.linalg.norm(YN[:, :AD], axis=1)
quiet = np.argsort(an)[:M // 10]
IDX = np.concatenate([rng.choice(M, NST // 2, replace=False),
                      rng.choice(quiet, NST - NST // 2, replace=False)])

# ---- data-side quantities per state: kappa (label-variation energy),
#      cond-noise n(x), local input spread ------------------------------------
kap, cnoise = np.zeros(NST), np.zeros(NST)
for j, i in enumerate(IDX):
    _, nb = tree.query(WN[i], k=K + 1)
    nb = nb[1:]
    dX = WN[nb] - WN[i]
    dY = YN[nb] - YN[i]
    # ridge local linear fit W: dY ~ dX @ Wl
    G = dX.T @ dX + 1e-6 * np.eye(dX.shape[1])
    Wl = np.linalg.solve(G, dX.T @ dY)          # (106, H*AD)
    pred = dX @ Wl
    kap[j] = float((pred ** 2).mean())          # label-variation energy captured
    cnoise[j] = float(((dY - pred) ** 2).mean())  # irreducible local residual
print("kappa p10/50/90:", np.percentile(kap, [10, 50, 90]).round(6),
      "| cnoise p10/50/90:", np.percentile(cnoise, [10, 50, 90]).round(6),
      flush=True)

XT = torch.tensor(WN[IDX], device=dev, dtype=torch.float32)


def outputs(ag, loss):
    t = torch.zeros(len(XT), device=dev)
    act0 = torch.zeros(len(XT), H, AD, device=dev)
    emb = ag.encoder_ema({"state": XT.reshape(-1, 2, 53)}, None)
    with torch.no_grad():
        pred, sraw = ag.flow_map_ema.net(act0, t, t, emb)
    return pred.reshape(len(XT), -1), sraw


def fit_slope(lx, ly):
    A = np.stack([lx, np.ones_like(lx)], 1)
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    r = np.corrcoef(lx, ly)[0, 1]
    return sol[0], r


for name, loss, d in ARMS:
    cfg, ds, ag = load(loss)
    snaps = [f"{d}/models/snap_{s}.pt" for s in (260000, 280000, 300000)]
    if not any(os.path.exists(c) for c in snaps):
        snaps = [f"{d}/models/model_latest.pt"]
    outs, sig = [], None
    for ck in snaps:
        if not os.path.exists(ck):
            continue
        ag.load(ck, load_optimizer=False)
        ag.eval()
        o, sraw = outputs(ag, loss)
        outs.append(o.cpu().numpy())
        if loss.startswith("regression_hetero"):
            sig = torch.nn.functional.softplus(sraw).reshape(len(XT), -1) \
                .mean(1).cpu().numpy() + 1e-3
    outs = np.stack(outs)                     # (S, NST, m)
    mu = outs.mean(0)
    m = kap > 1e-9
    if len(outs) >= 2:
        jit = ((outs - mu) ** 2).mean(axis=(0, 2))
        m = m & (jit > 1e-14)
        s5, r5 = fit_slope(np.log(kap[m]), np.log(jit[m]))
        print(f"THEORY {name} T5_OU slope {s5:+.2f} r {r5:+.2f} (pred -1)",
              flush=True)

    # T2: sigma stationarity (hetero arms)
    if sig is not None:
        m2 = (cnoise > 1e-12)
        s2_, r2_ = fit_slope(np.log(cnoise[m2]), np.log(sig[m2] ** 2))
        print(f"THEORY {name} T2_stationarity slope {s2_:+.2f} r {r2_:+.2f} "
              f"(pred +1; sig2 vs local condvar)", flush=True)
        # HG collapse: jitter vs kappa/sigma^2
        s5b, r5b = fit_slope(np.log(kap[m] / sig[m] ** 2), np.log(jit[m]))
        print(f"THEORY {name} T5_repriced slope {s5b:+.2f} r {r5b:+.2f}",
              flush=True)

    # T1: stiffness->survival: corr(log kappa, log ||J_enc||_F^2)
    jn = []
    for i in range(len(XT)):
        x = XT[i:i + 1].clone().requires_grad_(True)

        def f(inp):
            return ag.encoder_ema({"state": inp.reshape(1, 2, 53)}, None) \
                .reshape(-1)

        J = torch.autograd.functional.jacobian(f, x, vectorize=True)
        jn.append(float((J ** 2).sum()))
    jn = np.array(jn)
    s1, r1 = fit_slope(np.log(kap[m]), np.log(jn[m]))
    print(f"THEORY {name} T1_survival corr {r1:+.2f} slope {s1:+.2f}",
          flush=True)

    # T4: MIP endpoint identity ||df/dact_t||
    if loss == "mip":
        tau = float(cfg.optimization.t_two_step)
        t2 = torch.full((1,), tau, device=dev)
        jaf = []
        for i in range(min(20, len(XT))):
            emb = ag.encoder_ema({"state": XT[i:i + 1].reshape(1, 2, 53)}, None)
            y = torch.tensor(YN[IDX[i]], device=dev,
                             dtype=torch.float32).reshape(1, H, AD)
            at = (y + (1 - tau) * torch.randn_like(y)).requires_grad_(True)

            def g(a):
                return ag.flow_map_ema.get_velocity(t2, a.reshape(1, H, AD),
                                                    emb).reshape(-1)

            J = torch.autograd.functional.jacobian(g, at.reshape(-1),
                                                   vectorize=True)
            jaf.append(float((J ** 2).mean()))
        print(f"THEORY {name} T4_actjac meanF2 {np.mean(jaf):.4e} "
              f"p90 {np.percentile(jaf, 90):.4e} (pred: MP << pointing; "
              f"scale ~ condvar/(1-tau)^2, (1-tau)^2={ (1-tau)**2:.4f})",
              flush=True)
print("THEORY done", flush=True)
