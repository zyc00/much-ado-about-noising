"""Direct Jacobian axis decomposition on existing MP-200 checkpoints.

Question (no training needed): the conditioning/rank benefit on script data
— is it dimension-wise (flat gain spectrum across obs dims within a frame)
or observation-history-wise (balanced reliance across the To frames)?

For each checkpoint, compute J = d action / d obs (autograd, EMA nets, the
regression functional form f(o) = net(act0, t=0, emb(o))) at N normalized
data windows, and decompose:
  PR_full    participation ratio of singular values of J (out x To*D)
  PR_dim_f   PR of the per-frame block J_f (out x D), f = prev / last
  gain_f     ||J_f||_F   -> hist_ratio = gain_prev / gain_last
  PR_frames  PR of eigenvalues of the To x To frame Gram <J_f, J_f'>_F
             (2 = frames carry equal independent signal, 1 = one frame
             dominates or frames redundant)
Run ON A POD:
  MODELS="l2:logs/mp200_l2_s1000/models/model_latest.pt:regression,..." \
  DS=data/tool_hang_full2ins_mp200_relabel.hdf5 python scripts/probe_jac_axis.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

DS = os.environ.get("DS", "data/tool_hang_full2ins_mp200_relabel.hdf5")
N_SAMPLES = int(os.environ.get("SAMPLES", "48"))
cfgdir = os.path.abspath("examples/configs")


def pr(sq):  # participation ratio of nonneg spectrum (eigs or s^2)
    sq = np.asarray(sq, dtype=np.float64)
    s = sq.sum()
    return float((s * s) / ((sq * sq).sum() + 1e-30))


results = {}
ds = None
for spec in os.environ["MODELS"].split(","):
    tag, ckpt, loss = spec.split(":")[:3]
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(DS),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    if ds is None:
        ds = make_dataset(cfg.task)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(ds), N_SAMPLES)
        obs_windows = torch.stack(
            [torch.as_tensor(ds[int(i)]["obs"]["state"])[: cfg.task.obs_steps]
             for i in idx])
    dev = cfg.optimization.device
    ag = TrainingAgent(cfg)
    ag.load(ckpt, load_optimizer=False)
    ag.eval()
    net = ag.flow_map_ema
    enc = ag.encoder_ema
    To, D = obs_windows.shape[1], obs_windows.shape[2]

    def f(o_flat, To=To, D=D, net=net, enc=enc, dev=dev, cfg=cfg):
        o = o_flat.reshape(1, To, D)
        emb = enc(o, None)
        act0 = torch.zeros(1, 16, cfg.task.act_dim, device=dev)  # chiunet 2^n
        t = torch.zeros(1, device=dev)
        pred, _ = net.net(act0, t, t, emb)
        return pred.reshape(-1)

    # ANNULUS_MM: perturb eef pos dims (44:47) of both frames in RAW space
    # by a random direction of this magnitude (mm), then renormalize -> probes
    # J at off-manifold states in the failure annulus.
    ann_mm = float(os.environ.get("ANNULUS_MM", "0"))
    probe_windows = obs_windows
    if ann_mm > 0:
        no = ds.normalizer["obs"]["state"]
        raw = no.unnormalize(obs_windows.numpy())
        rng2 = np.random.default_rng(1)
        for i in range(raw.shape[0]):
            d = rng2.normal(size=3)
            d = d / (np.linalg.norm(d) + 1e-12) * (ann_mm / 1000.0)
            raw[i, :, 44:47] += d
        probe_windows = torch.as_tensor(
            no.normalize(raw), dtype=obs_windows.dtype)

    rows = []
    for i in range(N_SAMPLES):
        o = probe_windows[i].reshape(-1).to(dev).requires_grad_(True)
        J = torch.autograd.functional.jacobian(f, o, vectorize=True)
        J = J.detach().cpu().numpy().reshape(-1, To, D)  # (out, To, D)
        Jflat = J.reshape(J.shape[0], To * D)
        s2 = np.linalg.svd(Jflat, compute_uv=False) ** 2
        blocks = [J[:, fidx, :] for fidx in range(To)]
        s2_dim = [np.linalg.svd(b, compute_uv=False) ** 2 for b in blocks]
        gains = np.array([np.linalg.norm(b) for b in blocks])
        G = np.zeros((To, To))
        for a in range(To):
            for bb in range(To):
                G[a, bb] = (blocks[a] * blocks[bb]).sum()
        eig = np.clip(np.linalg.eigvalsh(G), 0, None)
        rows.append(dict(
            pr_full=pr(s2),
            pr_dim_prev=pr(s2_dim[0]), pr_dim_last=pr(s2_dim[-1]),
            gain=float(np.sqrt((gains ** 2).sum())),
            hist_ratio=float(gains[0] / (gains[-1] + 1e-12)),
            pr_frames=pr(eig),
        ))
    agg = {k: (float(np.mean([r[k] for r in rows])),
               float(np.std([r[k] for r in rows]))) for k in rows[0]}
    results[tag] = agg
    print(f"MODEL {tag:10s} " + "  ".join(
        f"{k} {m:.3f}±{s:.3f}" for k, (m, s) in agg.items()), flush=True)

print("\nPROBE-DONE")
