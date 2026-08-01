"""Tangent/normal-decomposed recovery gain: displace eef-pos obs along
(a) the local demo tangent (eef velocity direction) and (b) random
directions orthogonal to it; signed gain each. Env RV_ARMS, RV_DATA.
Prints R2 lines."""
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

D2 = os.environ.get("RV_DATA", "data/tool_hang_full2ins_mp_200.hdf5")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [tuple(a.split(":")) for a in os.environ["RV_ARMS"].split(",")]
DELTAS = [0.01, 0.02]
NDIR = 6

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
demos = []
for dn in names[:8]:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    demos.append(S)
h.close()

rng = np.random.default_rng(0)
for arm, loss, ck in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false",
            "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    Hn = int(cfg.task.horizon)
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device

    def pred(Wraw):
        xb = torch.tensor(np.stack([no.normalize(w) for w in Wraw]),
                          device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), Hn, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": xb})
        return np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, 1:9]))[:, :, 0:3]

    res = {("tanF", d): [] for d in DELTAS}
    res.update({("tanB", d): [] for d in DELTAS})
    res.update({("nrm", d): [] for d in DELTAS})
    for S in demos:
        L = len(S)
        idxs = list(range(3, L - Hn - 3, 6))
        W = np.stack([np.stack([S[i - 1], S[i]]) for i in idxs])
        tg = np.stack([S[i + 2, 44:47] - S[i - 2, 44:47] for i in idxs])
        tg = tg / (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9)
        base = pred(W)
        for d in DELTAS:
            for sgn, key in [(1.0, "tanF"), (-1.0, "tanB")]:
                Wp = W.copy()
                Wp[:, :, 44:47] += (sgn * d * tg)[:, None, :]
                dp = pred(Wp) - base
                s = (dp.mean(axis=1) * (sgn * tg)).sum(1) / d
                res[(key, d)].extend(s)
            for _ in range(NDIR):
                g = rng.standard_normal((len(idxs), 3))
                g = g - (g * tg).sum(1, keepdims=True) * tg
                g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
                Wp = W.copy()
                Wp[:, :, 44:47] += (d * g)[:, None, :]
                dp = pred(Wp) - base
                s = (dp.mean(axis=1) * g).sum(1) / d
                res[("nrm", d)].extend(s)
    for d in DELTAS:
        tf = np.asarray(res[("tanF", d)])
        tb = np.asarray(res[("tanB", d)])
        nr = np.asarray(res[("nrm", d)])
        print(f"R2 {arm} d={int(d * 1000)}mm tanFwd p50 {np.median(tf):+.4f} "
              f"tanBack p50 {np.median(tb):+.4f} normal p50 "
              f"{np.median(nr):+.4f} n_frac_amp {np.mean(nr > 0.05):.2f}",
              flush=True)
print("R2 done", flush=True)
