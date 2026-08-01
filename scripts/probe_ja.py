"""Anchor-block Jacobian spectra: J_a = df2/d(anchor) (24 pos-out x 160
anchor-in) at anchor=a1(x), plus view-2 state block T1 and step-1 J1
spectra, for NONOISE / ATK / MIP at on-support and annulus points.
Reports svmax, Frobenius, PR (effective rank). Prints JA lines."""
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
    ("NONOISE", "mip_nonoise", "logs/mp200_nonoise_s1000/models/snap_300000.pt"),
    ("ATK", "mip_nonoise_atk", "logs/mp200_nonoise_atk/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
]
NQ = 16

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
PW = []
for dn in names[:8]:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    for i in range(1, len(S) - Hn):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])))
h.close()
PW = np.stack(PW)
rng = np.random.default_rng(5)
qs = rng.choice(len(PW), NQ, replace=False)

for arm, loss, ck in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    fm, en = ag.flow_map_ema, ag.encoder_ema
    dev = cfg.optimization.device
    tts = float(cfg.optimization.t_two_step)
    res = {"on": [], "ann": []}
    for qi in qs:
        x0 = torch.tensor(PW[qi], device=dev)
        g = torch.randn_like(x0)
        g = g / g.norm()
        for locname, x in [("on", x0), ("ann", x0 + 0.4 * g)]:
            with torch.no_grad():
                emb = en({"state": x[None]}, None)
                s = torch.zeros((1,), device=dev)
                a1 = fm.get_velocity(s, torch.zeros((1, Hn, 10), device=dev),
                                     emb)[0].detach()

            def f2_of_a(a):
                embl = en({"state": x[None]}, None)
                t = torch.full((1,), tts, device=dev)
                return fm.get_velocity(t, a[None], embl)[0, 1:9, 0:3
                                                         ].reshape(-1)

            def f2_of_x(z):
                embl = en({"state": z[None]}, None)
                t = torch.full((1,), tts, device=dev)
                return fm.get_velocity(t, a1[None], embl)[0, 1:9, 0:3
                                                          ].reshape(-1)

            def f1_of_x(z):
                embl = en({"state": z[None]}, None)
                s0 = torch.zeros((1,), device=dev)
                return fm.get_velocity(s0, torch.zeros((1, Hn, 10),
                                                       device=dev),
                                       embl)[0, 1:9, 0:3].reshape(-1)

            Ja = torch.autograd.functional.jacobian(f2_of_a, a1
                                                    ).reshape(24, -1)
            T1 = torch.autograd.functional.jacobian(f2_of_x, x
                                                    ).reshape(24, -1)
            J1 = torch.autograd.functional.jacobian(f1_of_x, x
                                                    ).reshape(24, -1)
            row = []
            for M in (Ja, T1, J1):
                S_ = np.linalg.svd(M.cpu().numpy(), compute_uv=False)
                row += [float(S_[0]), float(np.sqrt((S_ ** 2).sum())),
                        float((S_ ** 2).sum() ** 2 / ((S_ ** 4).sum()
                                                      + 1e-18))]
            res[locname].append(row)
    for locname, rows in res.items():
        R = np.array(rows)
        print(f"JA {arm} {locname} Ja(sv/fro/PR) "
              f"{np.median(R[:, 0]):.3f}/{np.median(R[:, 1]):.3f}/"
              f"{np.median(R[:, 2]):.1f} | T1 {np.median(R[:, 3]):.3f}/"
              f"{np.median(R[:, 4]):.3f}/{np.median(R[:, 5]):.1f} | J1 "
              f"{np.median(R[:, 6]):.3f}/{np.median(R[:, 7]):.3f}/"
              f"{np.median(R[:, 8]):.1f}", flush=True)
print("JA done", flush=True)
