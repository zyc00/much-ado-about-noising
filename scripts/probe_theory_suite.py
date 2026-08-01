"""Rigorous test suite for the two theories on MP (scripted) data.

SOKOLIC (1605.08254): generalization is controlled by the Jacobian SPECTRAL
norm near the training samples.
  S2  within-arm trajectory: does the generalization gap track ||J||_2 across
      training snapshots of the SAME arm (no cross-arm confound)?
  S5  data-scale: gap vs ||J||_2 / sqrt(N).

DINH (1703.04933): parameter-space geometry (flatness/sharpness) is not
reparameterization invariant, so it cannot explain generalization.
  D2  epsilon-sharpness  S(theta) = max_{||d||<=eps||theta||} (L(theta+d) -
      L(theta)) / (1 + L(theta)) — Dinh's own quantity — measured per arm,
      then RE-measured after an EXACT function-preserving reparameterization
      (LeakyReLU homogeneity inside the encoder MLP).
  D3  invariance audit of every statistic we quote, under (a) hidden-unit
      rescaling and (b) hidden-unit PERMUTATION (both exact).

Prints TSUITE lines. Env: TS_ARMS "name:loss:logdir:dataset", TS_SNAPS.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.losses import get_loss_fn
from mip.samplers import get_sampler

ARMS = [tuple(a.split(":")) for a in os.environ["TS_ARMS"].split(",")]
SNAPS = [int(x) for x in os.environ.get(
    "TS_SNAPS", "20000,60000,120000,180000,300000").split(",")]
NJ = int(os.environ.get("TS_NJ", "32"))
NSH = int(os.environ.get("TS_NSH", "16"))
EPS = float(os.environ.get("TS_EPS", "5e-4"))
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def h5_states(path, lo, hi, n=192, seed=0):
    h = h5py.File(path, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[lo:hi]
    rng = np.random.RandomState(seed)
    W, Y = [], []
    for dn in names:
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
        A = np.asarray(h[f"data/{dn}/actions"]).astype(np.float32)
        if len(S) < 12:
            continue
        for i in rng.choice(np.arange(1, len(S) - 9), 2, replace=False):
            W.append(np.stack([S[i - 1], S[i]]))
            Y.append(A[i:i + 8])
    h.close()
    W, Y = np.stack(W), np.stack(Y)
    s_ = rng.choice(len(W), min(n, len(W)), replace=False)
    return W[s_], Y[s_]


def held_states(n=192):
    h = h5py.File(HELD, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2300]
    rng = np.random.RandomState(0)
    W, Y = [], []
    for dn in names:
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
        A = np.asarray(h[f"data/{dn}/actions"]).astype(np.float32)
        if len(S) < 12:
            continue
        for i in rng.choice(np.arange(1, len(S) - 9), 2, replace=False):
            W.append(np.stack([S[i - 1], S[i]]))
            Y.append(A[i:i + 8])
    h.close()
    W, Y = np.stack(W), np.stack(Y)
    s = rng.choice(len(W), min(n, len(W)), replace=False)
    return W[s], Y[s]


HW, HY = held_states()


def load(loss, dset):
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
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)


def pr(ev):
    ev = np.clip(np.asarray(ev, dtype=np.float64), 0, None)
    return float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30))


for name, loss, d, dset in ARMS:
    cfg, ds, ag = load(loss, dset)
    dev = cfg.optimization.device
    H = int(cfg.task.horizon)
    sampler = get_sampler(loss)
    lossfn = get_loss_fn(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    rb = ds.replay_buffer
    ends = rb.episode_ends[:]
    oe = rb["obs"]
    try:
        S_all = oe["state"][:]
    except (TypeError, IndexError, KeyError):
        S_all = oe[:]
    A_all = rb["action"][:]
    starts = np.concatenate([[0], ends[:-1]])
    TW, TY = [], []
    for e in range(len(ends)):
        s0, e0 = int(starts[e]), int(ends[e])
        if e0 - s0 < H + 3:
            continue
        S, A = S_all[s0:e0], A_all[s0:e0]
        for i in range(1, e0 - s0 - H, 5):
            TW.append(no.normalize(np.stack([S[i - 1], S[i]])))
            TY.append(na.normalize(A[i:i + H]))
    TW, TY = np.stack(TW), np.stack(TY)
    nwin = len(TW)
    rng = np.random.RandomState(0)
    tsel = rng.choice(nwin, 192, replace=False)
    XT = torch.tensor(TW[tsel], device=dev, dtype=torch.float32)  # (N,2,53)
    YT_raw = np.stack([A_all[0][:0]] * 0) if False else None
    XH = torch.tensor(np.stack([no.normalize(w) for w in HW]), device=dev,
                      dtype=torch.float32)
    YH = torch.tensor(HY, device=dev, dtype=torch.float32)
    # raw train targets for the gap (env units)
    ndem = 200 if "_200" in dset else 2000
    TRW, TRY = h5_states(dset, 0, ndem, 192, seed=1)
    XT = torch.tensor(np.stack([no.normalize(w) for w in TRW]), device=dev,
                      dtype=torch.float32)
    YT = torch.tensor(TRY, device=dev, dtype=torch.float32)

    def err(Xb, Yb):
        out = []
        for i in range(0, len(Xb), 128):
            xb = Xb[i:i + 128]
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
            pe = np.asarray(ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, 1:9]))
            out.append(np.abs(pe - Yb[i:i + 128].cpu().numpy()))
        return float(np.concatenate(out).mean())

    def specnorm():
        s2, prs = [], []
        for i in range(NJ):
            x = XT[i].reshape(-1).clone()  # 106

            def f(inp):
                a0 = torch.zeros((1, H, 10), device=dev)
                return sampler(cfg.optimization, ag.flow_map_ema,
                               ag.encoder_ema, a0,
                               {"state": inp.reshape(1, 2, 53)}).reshape(-1)

            J = torch.autograd.functional.jacobian(f, x, vectorize=True)
            s = torch.linalg.svdvals(J.reshape(-1, x.numel()).double())
            s2.append(float(s[0]))
            prs.append(pr((s ** 2).cpu().numpy()))
        return float(np.mean(s2)), float(np.mean(prs))

    def base_loss():
        idx = rng.choice(nwin, 512, replace=False)
        xb = torch.tensor(TW[idx], device=dev, dtype=torch.float32)
        yb = torch.tensor(TY[idx], device=dev, dtype=torch.float32)
        dt = torch.zeros(len(idx), device=dev)
        with torch.no_grad():
            l, _ = lossfn(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                          ag.interpolant, yb, {"state": xb}, dt)
        return float(l)

    def sharpness():
        """Dinh eps-sharpness on the trained params (EMA copies)."""
        ps = [p for p in list(ag.flow_map_ema.parameters()) +
              list(ag.encoder_ema.parameters())]
        base = base_loss()
        orig = [p.detach().clone() for p in ps]
        nrm = torch.sqrt(sum((p ** 2).sum() for p in ps))
        worst = 0.0
        g = torch.Generator(device=dev)
        for k in range(NSH):
            g.manual_seed(1000 + k)
            dirs = [torch.randn(p.shape, device=dev, generator=g) for p in ps]
            dn = torch.sqrt(sum((d ** 2).sum() for d in dirs))
            sc = EPS * nrm / dn
            with torch.no_grad():
                for p, dd in zip(ps, dirs):
                    p.add_(sc * dd)
            worst = max(worst, base_loss() - base)
            with torch.no_grad():
                for p, o in zip(ps, orig):
                    p.copy_(o)
        return worst / (1 + abs(base)), base

    for snap in SNAPS:
        ck = f"{d}/models/snap_{snap}.pt"
        if not os.path.exists(ck):
            continue
        ag.load(ck, load_optimizer=False)
        ag.eval()
        tr, hd = err(XT, YT), err(XH, YH)
        sn, jpr = specnorm()
        sh, bl = sharpness()
        print(f"TSUITE {name} snap {snap} train {tr:.5f} held {hd:.5f} "
              f"gap {hd - tr:.5f} specnorm {sn:.4f} depPR {jpr:.4f} "
              f"sharp {sh:.3e} loss {bl:.3e} nwin {nwin}", flush=True)

    # ---- D3: invariance audit under two EXACT symmetries -------------------
    ag.load(f"{d}/models/snap_300000.pt", load_optimizer=False)
    ag.eval()
    before = (specnorm(), sharpness()[0])
    encl = [m for m in ag.encoder_ema.modules() if isinstance(m, nn.Linear)]
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    al = torch.exp(0.8 * torch.randn(encl[0].out_features, device=dev,
                                     generator=g))
    with torch.no_grad():
        encl[0].weight.mul_(al.reshape(-1, 1))
        encl[0].bias.mul_(al)
        encl[1].weight.div_(al.reshape(1, -1))
    a_rs = (specnorm(), sharpness()[0])
    with torch.no_grad():
        encl[0].weight.div_(al.reshape(-1, 1))
        encl[0].bias.div_(al)
        encl[1].weight.mul_(al.reshape(1, -1))
    perm = torch.randperm(encl[0].out_features, generator=g, device=dev)
    with torch.no_grad():
        encl[0].weight.copy_(encl[0].weight[perm])
        encl[0].bias.copy_(encl[0].bias[perm])
        encl[1].weight.copy_(encl[1].weight[:, perm])
    a_pm = (specnorm(), sharpness()[0])
    print(f"TSUITE-INV {name} specnorm {before[0][0]:.4f} -> rescale "
          f"{a_rs[0][0]:.4f} / permute {a_pm[0][0]:.4f} | "
          f"depPR {before[0][1]:.4f} -> {a_rs[0][1]:.4f} / {a_pm[0][1]:.4f} | "
          f"sharpness {before[1]:.3e} -> {a_rs[1]:.3e} / {a_pm[1]:.3e}",
          flush=True)
print("TSUITE done", flush=True)
