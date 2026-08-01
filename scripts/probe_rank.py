"""Unified deployed-Jacobian rank table: L2/HT/HG/NONOISE/ATK/MIP at
on-support / mid-path / annulus states. PR, rank@90% energy, svmax.
Prints RK lines."""
import os
import os as _os
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

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
D2 = _os.environ.get("RK_DATA", "data/tool_hang_full2ins_mp_200.hdf5")
ARMS = ([tuple(a.split(":")) for a in _os.environ["RK_ARMS"].split(",")]
        if _os.environ.get("RK_ARMS") else [
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt"),
    ("HG", "regression_hetero_gauss",
     "logs/mp200_hg_s1000/models/snap_300000.pt"),
    ("NONOISE", "mip_nonoise", "logs/mp200_nonoise_s1000/models/snap_300000.pt"),
    ("ATK", "mip_nonoise_atk", "logs/mp200_nonoise_atk/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
])
NQ = 20

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
no = ds0.normalizer["obs"]["state"]

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PD = [], []
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    for i in range(1, len(S) - Hn):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])))
        PD.append(di)
h.close()
PW, PD = np.stack(PW), np.asarray(PD)
PF = PW.reshape(len(PW), -1)
tPF = torch.tensor(PF)
rng = np.random.default_rng(9)
qs = rng.choice(np.where(PD < 8)[0], NQ, replace=False)
LOCS = {"on": [], "mid": [], "ann": []}
for qi in qs:
    d = torch.norm(tPF - torch.tensor(PF[qi]), dim=1).numpy()
    d[PD == PD[qi]] = 1e9
    j = int(d.argmin())
    x0 = torch.tensor(PW[qi])
    LOCS["on"].append(x0)
    LOCS["mid"].append(0.5 * (x0 + torch.tensor(PW[j])))
    g = torch.tensor(rng.standard_normal(x0.shape), dtype=x0.dtype)
    LOCS["ann"].append(x0 + 0.4 * g / g.norm())

for arm, loss, ck in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    fm, en = ag.flow_map_ema, ag.encoder_ema
    dev = cfg.optimization.device
    tts = float(cfg.optimization.t_two_step)
    two = loss.startswith("mip")

    def g_of(x):
        emb = en({"state": x[None]}, None)
        z = torch.zeros((1, Hn, 10), device=dev)
        s = torch.zeros((1,), device=dev)
        a1 = fm.get_velocity(s, z, emb)
        if two:
            t = torch.full((1,), tts, device=dev)
            a1 = fm.get_velocity(t, a1, emb)
        return a1[0, 1:9, 0:3].reshape(-1)

    for locname, xs in LOCS.items():
        rows = []
        for x in xs:
            J = torch.autograd.functional.jacobian(g_of, x.to(dev)
                                                   ).reshape(24, -1)
            S_ = np.linalg.svd(J.cpu().numpy(), compute_uv=False)
            e = S_ ** 2
            pr = float(e.sum() ** 2 / ((e ** 2).sum() + 1e-18))
            cum = np.cumsum(e) / e.sum()
            r90 = int(np.searchsorted(cum, 0.9) + 1)
            rows.append((pr, r90, float(S_[0])))
        R = np.array(rows)
        print(f"RK {arm} {locname} PR p50 {np.median(R[:, 0]):.2f} "
              f"rank90 p50 {np.median(R[:, 1]):.0f} svmax p50 "
              f"{np.median(R[:, 2]):.3f}", flush=True)
print("RK done", flush=True)
