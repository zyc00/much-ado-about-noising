"""Feature-to-state-input Jacobian (FD): response of the chiunet trunk
feature (input to final_conv) to obs perturbations, per arm, on-support vs
annulus. K=12 directions, eps=0.05. For two-step arms also the view-2
evaluation point. Prints FJ lines."""
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
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt"),
    ("HG", "regression_hetero_gauss",
     "logs/mp200_hg_s1000/models/snap_300000.pt"),
    ("NONOISE", "mip_nonoise", "logs/mp200_nonoise_s1000/models/snap_300000.pt"),
    ("ATK", "mip_nonoise_atk", "logs/mp200_nonoise_atk/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
]
NQ, K, EPS = 24, 12, 0.05

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
rng = np.random.default_rng(7)
qs = rng.choice(len(PW), NQ, replace=False)
DIRS = rng.standard_normal((K, *PW[0].shape)).astype(np.float32)
DIRS /= np.linalg.norm(DIRS.reshape(K, -1), axis=1)[:, None, None]
ANN = rng.standard_normal((NQ, *PW[0].shape)).astype(np.float32)
ANN /= np.linalg.norm(ANN.reshape(NQ, -1), axis=1)[:, None, None]

for arm, loss, ck in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    fm, en = ag.flow_map_ema, ag.encoder_ema
    dev = cfg.optimization.device
    tts = float(cfg.optimization.t_two_step)
    cap = {}
    hk = fm.net.final_conv.register_forward_pre_hook(
        lambda m, inp: cap.__setitem__("f", inp[0].detach()))

    def feat(xb, view):
        emb = en({"state": xb}, None)
        b = xb.shape[0]
        z = torch.zeros((b, Hn, 10), device=dev)
        s = torch.zeros((b,), device=dev)
        with torch.no_grad():
            a1 = fm.get_velocity(s, z, emb)
            if view == "v2":
                t = torch.full((b,), tts, device=dev)
                fm.get_velocity(t, a1, emb)
        return cap["f"].reshape(b, -1).cpu().numpy()

    views = ["v1"] + (["v2"] if loss.startswith("mip") else [])
    for view in views:
        gains_on, gains_ann, cv2s = [], [], []
        for i, qi in enumerate(qs):
            for base, sink in [(PW[qi], gains_on),
                               (PW[qi] + 0.4 * ANN[i], gains_ann)]:
                xb = torch.tensor(
                    np.concatenate([base[None], base[None] + EPS * DIRS]),
                    device=dev, dtype=torch.float32)
                F = feat(xb, view)
                d = np.linalg.norm(F[1:] - F[0], axis=1) / EPS
                sink.append(np.median(d))
                if sink is gains_on:
                    cv2s.append(d.var() / (d.mean() ** 2 + 1e-12))
        print(f"FJ {arm} {view} featgain on p50 {np.median(gains_on):.2f} "
              f"ann p50 {np.median(gains_ann):.2f} ratio "
              f"{np.median(gains_ann) / np.median(gains_on):.2f} | CV2 "
              f"{np.median(cv2s):.2f}", flush=True)
    hk.remove()
print("FJ done", flush=True)
