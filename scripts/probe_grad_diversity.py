"""Per-sample GRADIENT DIVERSITY at convergence — the candidate mechanism
for the global representation-rank split (L2 emb_PR 3.3-4.8 vs everyone
else 6.8-7.4).

Hypothesis: weight rank accumulates from the span of the gradients the
objective supplies. If per-sample gradients are nearly parallel (as they
become for MSE once residuals are tiny and targets are deterministic), the
accumulated update is low-rank. Objectives that keep gradients diverse —
per-state reweighting (HG/HT), fresh input noise + two conditionings (MIP) —
accumulate rank.

For each arm at its own converged checkpoint, on its own training data:
  grad_PR   : participation ratio of the per-sample gradient Gram spectrum
              (effective number of independent gradient directions)
  cos_mean  : mean pairwise cosine of per-sample gradients (alignment)
  cos_seq   : mean cosine between successive MINIBATCH gradients (drift
              direction diversity across steps, incl. any input resampling)
  gnorm     : mean per-sample gradient norm
Env: GD_ARMS, GD_N, GD_DATASET. Prints GDIV lines.
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

DSET = os.environ.get("GD_DATASET", "data/tool_hang_full2ins_mp_200.hdf5")
NS = int(os.environ.get("GD_N", "128"))
NB = int(os.environ.get("GD_NB", "24"))
ARMS = [tuple(a.split(":")) for a in os.environ["GD_ARMS"].split(",")]


def load(loss, dset):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)


for name, loss, ck, dset in ARMS:
    cfg, ds, ag = load(loss, dset)
    ag.load(ck, load_optimizer=False)
    ag.train()
    fn = get_loss_fn(loss)
    dev = cfg.optimization.device
    params = [p for p in list(ag.flow_map.parameters()) +
              list(ag.encoder.parameters()) if p.requires_grad]
    rb = ds.replay_buffer
    ends = rb.episode_ends[:]
    obs_e = rb["obs"]
    try:
        S_all = obs_e["state"][:]
    except (TypeError, IndexError, KeyError):
        S_all = obs_e[:]
    A_all = rb["action"][:]
    H = int(cfg.task.horizon)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    starts = np.concatenate([[0], ends[:-1]])
    Wl, Yl = [], []
    for e in range(len(ends)):
        s0, e0 = int(starts[e]), int(ends[e])
        if e0 - s0 < H + 3:
            continue
        S, A = S_all[s0:e0], A_all[s0:e0]
        for i in range(1, e0 - s0 - H, 7):
            Wl.append(no.normalize(np.stack([S[i - 1], S[i]])))
            Yl.append(na.normalize(A[i:i + H]))
    Wl, Yl = np.stack(Wl), np.stack(Yl)
    rng = np.random.RandomState(0)
    sel = rng.choice(len(Wl), NS, replace=False)

    def grad_of(idx):
        xb = torch.tensor(Wl[idx], device=dev, dtype=torch.float32)
        yb = torch.tensor(Yl[idx], device=dev, dtype=torch.float32)
        dt = torch.zeros(len(idx), device=dev)
        for p in params:
            p.grad = None
        l, _ = fn(cfg.optimization, ag.flow_map, ag.encoder, ag.interpolant,
                  yb, {"state": xb}, dt)
        l.backward()
        return torch.cat([(p.grad if p.grad is not None
                           else torch.zeros_like(p)).reshape(-1)
                          for p in params]).detach()

    G = torch.stack([grad_of([i]) for i in sel])          # (NS, P)
    Gr = (G @ G.T).double().cpu().numpy()
    ev = np.linalg.eigvalsh(Gr)
    ev = np.clip(ev, 0, None)
    pr = float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30))
    nrm = np.sqrt(np.diag(Gr)) + 1e-30
    C = Gr / np.outer(nrm, nrm)
    cos_mean = float((C.sum() - np.trace(C)) / (len(C) * (len(C) - 1)))
    # successive-minibatch gradient alignment (includes input resampling)
    bg = []
    for b in range(NB):
        idx = rng.choice(len(Wl), 256, replace=False)
        bg.append(grad_of(idx))
    B = torch.stack(bg)
    Bn = B / (B.norm(dim=1, keepdim=True) + 1e-30)
    Cb = (Bn @ Bn.T).double().cpu().numpy()
    cos_seq = float((Cb.sum() - np.trace(Cb)) / (NB * (NB - 1)))
    print(f"GDIV {name} grad_PR {pr:.2f} cos_mean {cos_mean:+.4f} "
          f"cos_batch {cos_seq:+.4f} gnorm {float(G.norm(dim=1).mean()):.3e}",
          flush=True)
print("GDIV done", flush=True)
