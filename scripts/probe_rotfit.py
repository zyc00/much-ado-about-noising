"""Wrist-rotation-area anatomy: per-state GT rotation-chunk magnitude
strata; per-arm residuals SPLIT pos vs rot dims (raw units), rotation
attenuation |pred_rot|/|gt_rot|, and for HT/HG the learned sigma and
effective t-weight per stratum. Prints RF lines."""
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
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt"),
    ("HG", "regression_hetero_gauss",
     "logs/mp200_hg_s1000/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
]
AS_LO, AS_HI = 1, 9

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
    nu = float(getattr(cfg.optimization, "student_t_df", 2.0))

    def fwdrot(a):
        rot = ds.rotation_transformer.forward(a[..., 3:6])
        return np.concatenate([a[..., :3], rot, a[..., [6]]], -1)

    h = h5py.File(D2, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
    RP, RR, ATT, RM, SIG, WT, RN = [], [], [], [], [], [], []
    for dn in names[:8]:
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(
            np.float32)
        A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
        L = len(S)
        hi = L - Hn
        W = np.stack([np.stack([S[i - 1], S[i]]) for i in range(1, hi)])
        GT = np.stack([A[i - 1 + AS_LO:i - 1 + AS_HI] for i in range(1, hi)])
        Tn = np.stack([na.normalize(fwdrot(A[i - 1:i - 1 + Hn]))
                       for i in range(1, hi)])
        for i in range(0, len(W), 256):
            xb = torch.tensor(np.stack([no.normalize(w)
                                        for w in W[i:i + 256]]),
                              device=dev, dtype=torch.float32)
            tb = torch.tensor(Tn[i:i + 256], device=dev, dtype=torch.float32)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), Hn, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
                emb = ag.encoder_ema({"state": xb}, None)
                t0 = torch.zeros((len(xb),), device=dev)
                pr, sraw = ag.flow_map_ema.net(a0, t0, t0, emb)
                sig = torch.nn.functional.softplus(sraw).reshape(
                    len(xb), -1).mean(dim=1) + 1e-3
                r2 = ((pr - tb) ** 2).reshape(len(xb), -1)
                D = r2.shape[1]
                w = (nu + 1.0) / (nu + r2.sum(1) / (nu * sig ** 2 * D
                                                    ) * nu)
            pe = np.asarray(ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, AS_LO:AS_HI]))
            g = GT[i:i + 256]
            RP.append(np.abs(pe[:, :, 0:3] - g[:, :, 0:3]).mean((1, 2)))
            RR.append(np.abs(pe[:, :, 3:6] - g[:, :, 3:6]).mean((1, 2)))
            gm = np.abs(g[:, :, 3:6]).sum((1, 2))
            pm = np.abs(pe[:, :, 3:6]).sum((1, 2))
            ATT.append(pm / (gm + 1e-6))
            RM.append(gm)
            SIG.append(sig.cpu().numpy())
            WT.append(w.cpu().numpy())
            RN.append(r2.mean(1).cpu().numpy())
    h.close()
    RP, RR, ATT, RM, SIG, WT, RN = map(np.concatenate,
                                       (RP, RR, ATT, RM, SIG, WT, RN))
    t1, t2, t9 = np.percentile(RM, [33, 66, 90])
    STR = {"rot-lo": RM < t1, "rot-mid": (RM >= t1) & (RM < t2),
           "rot-hi": RM >= t2, "rot-top10%": RM >= t9}
    for k, m in STR.items():
        print(f"RF {arm} {k} posres {np.median(RP[m]):.5f} rotres "
              f"{np.median(RR[m]):.5f} atten {np.median(ATT[m]):.2f} | "
              f"normres {np.median(RN[m]):.2e} sigma "
              f"{np.median(SIG[m]):.4f} t-weight {np.median(WT[m]):.2f}",
              flush=True)
print("RF done", flush=True)
