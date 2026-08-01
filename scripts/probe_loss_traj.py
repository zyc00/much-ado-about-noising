"""Per-step training residual along complete MP-200 demos for L2 / HT / MIP,
at snapshots 60k (mid-training) and 300k (converged) — the loss-side
companion of probe_weight_traj. Targets aligned per the harness convention
(chunk index 0 = first obs frame). Saves npz; prints LTRAJ lines.
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

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
NDEMO = int(os.environ.get("LT_NDEMO", "4"))
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [("L2", "regression", "logs/mp200_l2_s1000/models"),
        ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models"),
        ("MIP", "mip", "logs/mp200_mip_s1000/models")]
SNAPS = ["60000", "300000"]

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
demos = []
for dn in names[:NDEMO]:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    demos.append((dn, S, A))
h.close()
print(f"LTRAJ demos {NDEMO}", flush=True)

out = {}
for arm, loss, mdir in ARMS:
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
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device

    def fwd(a):
        rot = ds.rotation_transformer.forward(a[..., 3:6])
        return np.concatenate([a[..., :3], rot, a[..., [6]]], -1)

    for snap in SNAPS:
        ag = TrainingAgent(cfg)
        ag.load(f"{mdir}/snap_{snap}.pt", load_optimizer=False)
        ag.eval()
        for dn, S, A in demos:
            L = len(S)
            hi = L - Hn
            W = np.stack([np.stack([S[i - 1], S[i]]) for i in range(1, hi)])
            Tn = np.stack([na.normalize(fwd(A[i - 1:i - 1 + Hn]))
                           for i in range(1, hi)])
            R = []
            for i in range(0, len(W), 256):
                xb = torch.tensor(
                    np.stack([no.normalize(w) for w in W[i:i + 256]]),
                    device=dev, dtype=torch.float32)
                tb = torch.tensor(Tn[i:i + 256], device=dev,
                                  dtype=torch.float32)
                with torch.no_grad():
                    a0 = torch.zeros((len(xb), Hn, 10), device=dev)
                    an = sampler(cfg.optimization, ag.flow_map_ema,
                                 ag.encoder_ema, a0, {"state": xb})
                R.append(((an - tb) ** 2).mean(dim=(1, 2)).cpu().numpy())
            R = np.concatenate(R)
            out[f"{arm}{snap}_{dn}_r"] = R
            if f"eef_{dn}" not in out:
                out[f"eef_{dn}"] = S[1:hi, 44:47]
                out[f"grip_{dn}"] = A[1:hi, 6]
            print(f"LTRAJ {arm}-{snap} {dn} T={len(R)} p50 {np.median(R):.2e} "
                  f"p90 {np.percentile(R,90):.2e} max {R.max():.2e}",
                  flush=True)
np.savez_compressed("analysis/manifold/loss_traj.npz", **out)
print("LTRAJ done", flush=True)
