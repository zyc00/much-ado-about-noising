"""Rahaman et al. (1806.08734), 'On the Spectral Bias of Neural Networks':
ReLU nets learn LOW frequencies first and express high frequencies only with
finely-tuned parameters (so high-frequency content is fragile).

Test on MP data. A demonstration trajectory is a dense 1-D path through
observation space with GROUND-TRUTH actions at every point, so we can compare
the FREQUENCY CONTENT of each policy's output along that path against the
target's own spectrum.

Measured, per arm, over K demo trajectories (stride 1):
  cap_lo / cap_mid / cap_hi : captured power ratio (pred/GT) per frequency
                              band -- spectral bias predicts cap_hi << cap_lo
  err_lo / err_mid / err_hi : error power per band (fraction of total error)
  frag_hi                   : after a small PARAMETER perturbation
                              (eps*||theta||), the fraction of high-band power
                              lost -- their 'finely tuned parameters' claim
Optionally across snapshots (SB_SNAPS) to see the learning ORDER.
Env: SB_ARMS "name:loss:logdir:dataset", SB_SNAPS, SB_K, SB_EPS.
Prints SBIAS lines.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

ARMS = [tuple(a.split(":")) for a in os.environ["SB_ARMS"].split(",")]
SNAPS = [int(x) for x in os.environ.get("SB_SNAPS", "300000").split(",")]
K = int(os.environ.get("SB_K", "12"))
EPSP = float(os.environ.get("SB_EPS", "2e-3"))
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def traj_set(path, lo, hi, k):
    h = h5py.File(path, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[lo:hi]
    rng = np.random.RandomState(0)
    out = []
    for dn in rng.choice(names, min(k, len(names)), replace=False):
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[kk]) for kk in OK], 1).astype(np.float32)
        A = np.asarray(h[f"data/{dn}/actions"]).astype(np.float32)
        if len(S) < 96:
            continue
        W = np.stack([np.stack([S[i - 1], S[i]]) for i in range(1, len(S) - 9)])
        Y = np.stack([A[i] for i in range(1, len(S) - 9)])   # executed action
        out.append((W, Y))
    h.close()
    return out


def bands(n):
    f = np.fft.rfftfreq(n)
    return f < 0.05, (f >= 0.05) & (f < 0.2), f >= 0.2


def spec(x):
    """power spectrum along time, summed over action dims (x: T x d)"""
    X = np.fft.rfft(x - x.mean(0), axis=0)
    return (np.abs(X) ** 2).sum(1)


for name, loss, d, dset in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false",
            "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    H = int(cfg.task.horizon)
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device
    ndem = 200 if "_200" in dset else 2000
    TR = traj_set(dset, 0, ndem, K)

    def predict(W):
        out = []
        for i in range(0, len(W), 256):
            xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 256]]),
                              device=dev, dtype=torch.float32)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
            pe = np.asarray(ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, 1:2]))[:, 0]
            out.append(pe)
        return np.concatenate(out)

    for snap in SNAPS:
        ck = f"{d}/models/snap_{snap}.pt"
        if not os.path.exists(ck):
            continue
        ag.load(ck, load_optimizer=False)
        ag.eval()
        cap, errp, tot = np.zeros(3), np.zeros(3), np.zeros(3)
        gtp = np.zeros(3)
        base_hi = []
        for W, Y in TR:
            P = predict(W)
            n = min(len(P), len(Y))
            sp, sg = spec(P[:n]), spec(Y[:n])
            se = spec(P[:n] - Y[:n])
            lo, mid, hi = bands(n)
            for j, m in enumerate((lo, mid, hi)):
                cap[j] += sp[m].sum()
                gtp[j] += sg[m].sum()
                errp[j] += se[m].sum()
            base_hi.append(sp[hi].sum())
        capr = cap / (gtp + 1e-30)
        errf = errp / (errp.sum() + 1e-30)
        # fragility: small parameter perturbation, how much high-band power is lost
        ps = list(ag.flow_map_ema.parameters()) + list(ag.encoder_ema.parameters())
        orig = [p.detach().clone() for p in ps]
        nrm = torch.sqrt(sum((p ** 2).sum() for p in ps))
        g = torch.Generator(device=dev); g.manual_seed(3)
        dirs = [torch.randn(p.shape, device=dev, generator=g) for p in ps]
        dn = torch.sqrt(sum((x ** 2).sum() for x in dirs))
        with torch.no_grad():
            for p, dd in zip(ps, dirs):
                p.add_(EPSP * nrm / dn * dd)
        hi_after = []
        for (W, Y), b0 in zip(TR, base_hi):
            P = predict(W)
            n = min(len(P), len(Y))
            _, _, hi = bands(n)
            hi_after.append(spec(P[:n])[hi].sum())
        with torch.no_grad():
            for p, o in zip(ps, orig):
                p.copy_(o)
        frag = 1.0 - float(np.sum(hi_after) / (np.sum(base_hi) + 1e-30))
        print(f"SBIAS {name} snap {snap} cap_lo {capr[0]:.3f} cap_mid "
              f"{capr[1]:.3f} cap_hi {capr[2]:.3f} | errfrac_lo {errf[0]:.3f} "
              f"mid {errf[1]:.3f} hi {errf[2]:.3f} | GTpow_hi/lo "
              f"{gtp[2]/(gtp[0]+1e-30):.4f} | frag_hi {frag:+.3f}", flush=True)
print("SBIAS done", flush=True)
