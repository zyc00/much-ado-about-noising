"""Dinh et al. (1703.04933) test applied to our rank measurements.

The encoder is an MLP with LeakyReLU (positively homogeneous) and the
embedding is consumed by a Linear layer (global_cond_encoder), so there is an
EXACT function-preserving reparameterization: scale the encoder's output
units by alpha (final Linear rows + bias) and divide the consuming Linear's
corresponding input columns by alpha. The function — hence every prediction,
every rollout, every success rate — is unchanged bit-for-bit up to float
error.

We then re-measure every rank statistic we have been quoting and report which
ones MOVE (parameter-space quantities, meaningless per Dinh) and which ones
DO NOT (function-space quantities).

Modes for alpha: 'equalize' (alpha_i = 1/std_i of the embedding coordinate —
maximally raises the embedding PR), 'random' (lognormal).
Env: RP_ARM (name:loss:ckpt), RP_DATASET, RP_MODE, RP_N. Prints RPAR lines.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

DSET = os.environ.get("RP_DATASET", "data/tool_hang_full2ins_mp_200.hdf5")
NAME, LOSS, CK = os.environ["RP_ARM"].split(":")
MODE = os.environ.get("RP_MODE", "equalize")
NST = int(os.environ.get("RP_N", "512"))
NJ = int(os.environ.get("RP_NJ", "24"))

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(DSET),
        "network=chiunet", f"optimization.loss_type={LOSS}",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
H = int(cfg.task.horizon)
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
ag.load(CK, load_optimizer=False)
ag.eval()
sampler = get_sampler(LOSS)
no = ds.normalizer["obs"]["state"]
dev = cfg.optimization.device

rb = ds.replay_buffer
ends = rb.episode_ends[:]
obs_e = rb["obs"]
try:
    S_all = obs_e["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_e[:]
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


def pr(ev):
    ev = np.clip(np.asarray(ev, dtype=np.float64), 0, None)
    return float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30))


def measure(tag):
    with torch.no_grad():
        E = ag.encoder_ema({"state": X.reshape(-1, 2, 53)}, None).reshape(len(X), -1)
        a0 = torch.zeros((len(X), H, 10), device=dev)
        out = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema, a0,
                      {"state": X.reshape(-1, 2, 53)})
    Ec = (E - E.mean(0)).double()
    ev = torch.linalg.svdvals(Ec).cpu().numpy() ** 2
    # encoder-Jacobian PR (parameter/representation space) and deployed-map
    # Jacobian norms + PR (function space)
    ej, dj, dn2, dnf = [], [], [], []
    for i in range(NJ):
        x = X[i].clone()

        def fe(inp):
            return ag.encoder_ema({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)

        Je = torch.autograd.functional.jacobian(fe, x, vectorize=True)
        ej.append(pr((torch.linalg.svdvals(Je.reshape(-1, x.numel()).double())
                      ** 2).cpu().numpy()))

        def fd(inp):
            aa = torch.zeros((1, H, 10), device=dev)
            return sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                           aa, {"state": inp.reshape(1, 2, 53)}).reshape(-1)

        Jd = torch.autograd.functional.jacobian(fd, x, vectorize=True)
        s = torch.linalg.svdvals(Jd.reshape(-1, x.numel()).double())
        dj.append(pr((s ** 2).cpu().numpy()))
        dn2.append(float(s[0]))
        dnf.append(float((s ** 2).sum().sqrt()))
    sr = []
    for n_, p_ in ag.encoder_ema.named_parameters():
        if p_.dim() == 2 and min(p_.shape) > 4:
            s_ = torch.linalg.svdvals(p_.double())
            sr.append(round(float((s_ ** 2).sum() / (s_[0] ** 2 + 1e-30)), 1))
    print(f"RPAR {NAME} {tag} emb_PR {pr(ev):.3f} encJac_PR {np.mean(ej):.3f} "
          f"W_srank {sr} || deployedJac_PR {np.mean(dj):.4f} "
          f"||J||_2 {np.mean(dn2):.4f} ||J||_F {np.mean(dnf):.4f}", flush=True)
    return out.detach().clone()


out_before = measure("BEFORE")

# ---- find the exact symmetry
enc_lin = [m for m in ag.encoder_ema.modules() if isinstance(m, nn.Linear)]
if MODE == "internal":
    # EXACT symmetry: LeakyReLU is positively homogeneous, so scaling the
    # units of a HIDDEN layer and compensating in the next layer leaves the
    # encoder output (hence the whole function) bit-identical.
    L1, L2_ = enc_lin[0], enc_lin[1]
    g = torch.Generator(device=dev); g.manual_seed(0)
    al = torch.exp(0.8 * torch.randn(L1.out_features, device=dev, generator=g))
    with torch.no_grad():
        L1.weight.mul_(al.reshape(-1, 1))
        if L1.bias is not None:
            L1.bias.mul_(al)
        L2_.weight.div_(al.reshape(1, -1))
    out_after = measure("AFTER(internal-exact)")
    d = (out_after - out_before).abs()
    print(f"RPAR {NAME} FUNCTION-CHANGE max {float(d.max()):.3e} "
          f"mean {float(d.mean()):.3e} (|a| {float(out_before.abs().mean()):.3e})",
          flush=True)
    print("RPAR done", flush=True)
    sys.exit(0)
enc_last = enc_lin[-1]
cons = [m for m in ag.flow_map_ema.modules()
        if isinstance(m, nn.Linear) and m.in_features == enc_last.out_features]
print(f"RPAR-INFO enc_last out={enc_last.out_features} consumers={len(cons)} "
      f"{[c.in_features for c in cons]}", flush=True)
if not cons:
    print("RPAR-INFO no linear consumer found; aborting", flush=True)
    sys.exit(0)

with torch.no_grad():
    E0 = ag.encoder_ema({"state": X.reshape(-1, 2, 53)}, None).reshape(len(X), -1)
    sd = E0.std(0) + 1e-6
    if MODE == "equalize":
        alpha = (sd.mean() / sd).clamp(0.05, 20.0)
    else:
        g = torch.Generator(device=dev); g.manual_seed(0)
        alpha = torch.exp(0.8 * torch.randn(enc_last.out_features, device=dev,
                                            generator=g))
    # encoder output units scaled by alpha ...
    enc_last.weight.mul_(alpha.reshape(-1, 1))
    if enc_last.bias is not None:
        enc_last.bias.mul_(alpha)
    # ... and every consumer's input columns divided by alpha
    for c in cons:
        c.weight.div_(alpha.reshape(1, -1))

out_after = measure(f"AFTER({MODE})")
d = (out_after - out_before).abs()
print(f"RPAR {NAME} FUNCTION-CHANGE max {float(d.max()):.3e} "
      f"mean {float(d.mean()):.3e} (relative to |a| "
      f"{float(out_before.abs().mean()):.3e})", flush=True)
print("RPAR done", flush=True)
