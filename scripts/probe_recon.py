"""Reconcile the human fit-residual sign flip: prior note (first-slot,
normalized 10-d, 47,833 pairs) says hMSE 2x TIGHTER than hMIP on easy bulk;
HFIT (executed slots 1-9, raw pos, EMA, 6 demos) says hMSE 4-10x LOOSER.
Factorize: net {ema, raw} x quantity {slot0-norm10d, exec19-norm10d,
exec19-rawpos} on ALL 200 demos, per tremor tercile. Prints RC lines."""
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

HUM = "data/tool_hang_human_lowdim_up.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [("hMSE", "regression", "logs/hbase2/models/model_latest.pt"),
        ("hHT", "regression_hetero_t",
         "logs/hheterot_s1000/models/model_latest.pt"),
        ("hMIP", "mip", "logs/hmip0/models/model_latest.pt")]

h = h5py.File(HUM, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
demos = []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    demos.append((S, A))
h.close()
print(f"RC demos {len(demos)}", flush=True)

for arm, loss, ck in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(HUM), "network=chiunet",
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

    def fwd(a):
        rot = ds.rotation_transformer.forward(a[..., 3:6])
        return np.concatenate([a[..., :3], rot, a[..., [6]]], -1)

    for tag, fm, em in [("ema", ag.flow_map_ema, ag.encoder_ema),
                        ("raw", ag.flow_map, ag.encoder)]:
        S0, E19N, E19P, TR = [], [], [], []
        for S, A in demos:
            L = len(S)
            hi = L - Hn
            W = np.stack([np.stack([S[i - 1], S[i]]) for i in range(1, hi)])
            Tn = np.stack([na.normalize(fwd(A[i - 1:i - 1 + Hn]))
                           for i in range(1, hi)])
            GP = np.stack([A[i:i + 8, 0:3] for i in range(1, hi)])
            for i in range(0, len(W), 512):
                xb = torch.tensor(
                    np.stack([no.normalize(w) for w in W[i:i + 512]]),
                    device=dev, dtype=torch.float32)
                tb = torch.tensor(Tn[i:i + 512], device=dev,
                                  dtype=torch.float32)
                with torch.no_grad():
                    a0 = torch.zeros((len(xb), Hn, 10), device=dev)
                    an = sampler(cfg.optimization, fm, em, a0, {"state": xb})
                e = an - tb
                S0.append(torch.sqrt((e[:, 0] ** 2).mean(1)).cpu().numpy())
                E19N.append(torch.sqrt(
                    (e[:, 1:9] ** 2).mean((1, 2))).cpu().numpy())
                pe = np.asarray(ds.undo_transform_action(
                    na.unnormalize(an.cpu().numpy())[:, 1:9]))
                E19P.append(np.abs(pe[:, :, 0:3] - GP[i:i + 512]
                                   ).mean(axis=(1, 2)))
            TR.append(np.asarray([np.abs(np.diff(A[i:i + 8, 0:3], axis=0)
                                         ).sum() for i in range(1, hi)]))
        S0, E19N, E19P = map(np.concatenate, (S0, E19N, E19P))
        TR = np.concatenate(TR)
        t1, t2 = np.percentile(TR, [33, 66])
        m_lo, m_hi = TR < t1, TR >= t2
        print(f"RC {arm} {tag} n={len(S0)} | slot0-norm RMS "
              f"{np.sqrt((S0 ** 2).mean()):.3f} med {np.median(S0):.3f} "
              f"lo/hi {np.median(S0[m_lo]):.3f}/{np.median(S0[m_hi]):.3f} | "
              f"exec19-norm med {np.median(E19N):.3f} lo/hi "
              f"{np.median(E19N[m_lo]):.3f}/{np.median(E19N[m_hi]):.3f} | "
              f"exec19-pos med {np.median(E19P):.5f} lo/hi "
              f"{np.median(E19P[m_lo]):.5f}/{np.median(E19P[m_hi]):.5f}",
              flush=True)
print("RC done", flush=True)
