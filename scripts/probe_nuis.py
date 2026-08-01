"""Nuisance-coupling test: (1) per-arm FD gain of commanded action wrt
OBJECT-block vs EEF-block obs dims at descent-phase rollout states;
(2) across-seed regression of pick-z on object initial pose. Prints NC."""
import glob
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

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt",
     "logs/rd_ht"),
    ("HG", "regression_hetero_gauss",
     "logs/mp200_hg_s1000/models/snap_300000.pt", "logs/rd_hg"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt",
     "logs/rd_mip"),
]
EPS, K = 0.05, 8

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg0 = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg0, False)
cfg0.task.obs_dim = 53
cfg0.task.horizon = int(2 ** np.ceil(np.log2(cfg0.task.horizon)))
Hn = int(cfg0.task.horizon)
ds0 = make_dataset(cfg0.task)
no, na0 = ds0.normalizer["obs"]["state"], ds0.normalizer["action"]
AS = int(cfg0.task.act_steps)
start = int(cfg0.task.obs_steps) - 1
rng = np.random.default_rng(11)

for arm, loss, ck, dr in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    dev = cfg.optimization.device

    def cmd(xb):
        with torch.no_grad():
            a0 = torch.zeros((len(xb), Hn, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": xb})
        return an[:, start:start + AS].cpu().numpy()

    W, pickz, objxy = [], [], []
    for f in sorted(glob.glob(f"{dr}/ep_*.npz")):
        z = np.load(f)
        obs = z["obs"]
        g = obs[:, 51:53].sum(1)
        thr = 0.5 * (g.max() + g.min())
        closed = np.where(g < thr)[0]
        if not len(closed):
            continue
        tg = int(closed[0])
        zz = obs[:, 46]
        w0 = max(0, tg - 40)
        tpick = w0 + int(np.argmin(zz[w0:tg + 5]))
        pickz.append(zz[tpick])
        objxy.append(obs[0, 0:8])
        for i in range(max(1, tg - 30), tg, 6):
            W.append(np.stack([obs[i - 1], obs[i]]))
    W = np.stack(W)
    Xn = np.stack([no.normalize(w) for w in W])
    xb = torch.tensor(Xn, device=dev, dtype=torch.float32)
    base = cmd(xb)
    gains = {"obj": [], "eef": []}
    for grp, sl in [("obj", slice(0, 44)), ("eef", slice(44, 47))]:
        for _ in range(K):
            v = np.zeros_like(Xn)
            g_ = rng.standard_normal((len(Xn), 2, sl.stop - sl.start))
            g_ /= np.linalg.norm(g_.reshape(len(Xn), -1), axis=1
                                 )[:, None, None] + 1e-9
            v[:, :, sl] = g_
            pb = cmd(torch.tensor(Xn + EPS * v, device=dev,
                                  dtype=torch.float32))
            gains[grp].append(np.sqrt(((pb - base) ** 2).reshape(
                len(Xn), -1).mean(1)) / EPS)
    go = np.median(np.stack(gains["obj"]), 0)
    ge = np.median(np.stack(gains["eef"]), 0)
    pz, ox = np.asarray(pickz), np.stack(objxy)
    X = np.concatenate([ox, np.ones((len(ox), 1))], 1)
    beta, res, *_ = np.linalg.lstsq(X, pz, rcond=None)
    pred = X @ beta
    ss = 1 - ((pz - pred) ** 2).sum() / (((pz - pz.mean()) ** 2).sum()
                                         + 1e-12)
    print(f"NC {arm} descent gains: obj p50 {np.median(go):.4f} eef p50 "
          f"{np.median(ge):.4f} ratio obj/eef {np.median(go / (ge + 1e-9)):.2f}"
          f" | pick-z~obj R2 {ss:.2f} (n={len(pz)}, pickz std "
          f"{1000 * pz.std():.1f}mm)", flush=True)
print("NC done", flush=True)
