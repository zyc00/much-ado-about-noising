"""Decompose the deployed 2-step Jacobian J_g = T1 + T2,
T1 = df2/dx (anchor frozen), T2 = df2/da . J1, at on-support / mid-path /
annulus points, for MIP / NONOISE / ATK. Reports |T1|, |T2|, |J1|, |J_g|,
and interference = (|T1+T2| - |T1|) / (|T2| + eps)  (negative => T2 cancels
T1). Prints JD lines."""
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

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
    ("NONOISE", "mip_nonoise", "logs/mp200_nonoise_s1000/models/snap_300000.pt"),
    ("ATK", "mip_nonoise_atk", "logs/mp200_nonoise_atk/models/snap_300000.pt"),
]
NQ = 24

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
rng = np.random.default_rng(3)
qs = rng.choice(np.where(PD < 8)[0], NQ, replace=False)

LOCS = {}
for qi in qs:
    d = torch.norm(tPF - torch.tensor(PF[qi]), dim=1).numpy()
    d[PD == PD[qi]] = 1e9
    j = int(d.argmin())
    x0 = torch.tensor(PW[qi])
    LOCS.setdefault("on", []).append(x0)
    LOCS.setdefault("mid", []).append(0.5 * (x0 + torch.tensor(PW[j])))
    g = torch.tensor(rng.standard_normal(x0.shape), dtype=x0.dtype)
    g = g / g.norm()
    LOCS.setdefault("ann", []).append(x0 + 0.4 * g)

for arm, loss, ck in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    fm, en = ag.flow_map_ema, ag.encoder_ema
    dev = cfg.optimization.device
    tts = float(cfg.optimization.t_two_step)

    def parts(x):
        def f1(z):
            emb = en({"state": z[None]}, None)
            s = torch.zeros((1,), device=dev)
            a1 = fm.get_velocity(s, torch.zeros((1, Hn, 10), device=dev), emb)
            return a1[0]

        def g_full(z):
            emb = en({"state": z[None]}, None)
            s = torch.zeros((1,), device=dev)
            a1 = fm.get_velocity(s, torch.zeros((1, Hn, 10), device=dev), emb)
            t = torch.full((1,), tts, device=dev)
            return fm.get_velocity(t, a1, emb)[0, 1:9, 0:3].reshape(-1)

        with torch.no_grad():
            a1c = f1(x).detach()

        def g_frozen(z):
            emb = en({"state": z[None]}, None)
            t = torch.full((1,), tts, device=dev)
            return fm.get_velocity(t, a1c[None], emb)[0, 1:9, 0:3].reshape(-1)

        Jf = torch.autograd.functional.jacobian(g_full, x).reshape(24, -1)
        T1 = torch.autograd.functional.jacobian(g_frozen, x).reshape(24, -1)
        J1 = torch.autograd.functional.jacobian(
            lambda z: f1(z)[1:9, 0:3].reshape(-1), x).reshape(24, -1)
        T2 = Jf - T1
        return (float(torch.linalg.matrix_norm(Jf, 2)),
                float(torch.linalg.matrix_norm(T1, 2)),
                float(torch.linalg.matrix_norm(T2, 2)),
                float(torch.linalg.matrix_norm(J1, 2)))

    for locname, xs in LOCS.items():
        R = np.array([parts(x.to(dev)) for x in xs])
        interf = (R[:, 0] - R[:, 1]) / (R[:, 2] + 1e-9)
        print(f"JD {arm} {locname} |Jg| p50 {np.median(R[:, 0]):.3f} |T1| "
              f"{np.median(R[:, 1]):.3f} |T2| {np.median(R[:, 2]):.3f} |J1| "
              f"{np.median(R[:, 3]):.3f} | interference p50 "
              f"{np.median(interf):+.2f}", flush=True)
print("JD done", flush=True)
