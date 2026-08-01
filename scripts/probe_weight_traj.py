"""Per-step learned loss weight w(x) = 1/sigma(x)^2 of HT/HG along complete
MP-200 demo trajectories, for visualization of which regions the hetero
objectives up/down-weight.

For each of WT_NDEMO demos: every step's obs window -> sigma from the
scalar head at snapshots 20k and 60k, both arms. Weights normalized by the
mean of 1/sigma^2 over a 4000-state random sample of the whole dataset
(so w>1 = upweighted relative to dataset average). Saves npz with per-demo
eef_pos, progress, and w arrays + prints WTRAJ summary lines.
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

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
NDEMO = int(os.environ.get("WT_NDEMO", "4"))
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models"),
        ("HG", "regression_hetero_gauss", "logs/mp200_hg_s1000/models")]
SNAPS = ["20000", "60000"]

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
demos = []
for dn in names[:NDEMO]:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    demos.append((dn, S, A))
# random reference sample over ALL demos for the normalizing mean
rng = np.random.RandomState(0)
REF = []
for dn in names:
    S = np.concatenate([np.asarray(h[f"data/{dn}/obs"][k]) for k in OK],
                       1).astype(np.float32)
    if len(S) < 12:
        continue
    for i in rng.choice(np.arange(1, len(S)), 20, replace=False):
        REF.append(np.stack([S[i - 1], S[i]]))
h.close()
REF = np.stack(REF)
print(f"WTRAJ demos {NDEMO} ref {len(REF)}", flush=True)

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
    no = ds.normalizer["obs"]["state"]
    dev = cfg.optimization.device

    for snap in SNAPS:
        ag = TrainingAgent(cfg)
        ag.load(f"{mdir}/snap_{snap}.pt", load_optimizer=False)
        ag.eval()

        def sig(WS):
            res = []
            for i in range(0, len(WS), 256):
                xb = torch.tensor(
                    np.stack([no.normalize(w) for w in WS[i:i + 256]]),
                    device=dev, dtype=torch.float32)
                with torch.no_grad():
                    a0 = torch.zeros((len(xb), Hn, 10), device=dev)
                    t0 = torch.zeros(len(xb), device=dev)
                    emb = ag.encoder_ema({"state": xb}, None)
                    _, s_raw = ag.flow_map_ema.net(a0, t0, t0, emb)
                    res.append((torch.nn.functional.softplus(s_raw)
                                .reshape(len(xb), -1).mean(1) + 1e-3)
                               .cpu().numpy())
            return np.concatenate(res)

        wmean = float((1.0 / sig(REF) ** 2).mean())
        for dn, S, A in demos:
            W = np.stack([np.stack([S[i - 1], S[i]])
                          for i in range(1, len(S))])
            w = (1.0 / sig(W) ** 2) / wmean
            key = f"{arm}{snap}_{dn}"
            out[f"{key}_w"] = w
            if f"eef_{dn}" not in out:
                out[f"eef_{dn}"] = S[1:, 44:47]
                out[f"grip_{dn}"] = A[1:, 6]
            print(f"WTRAJ {arm}-{snap} {dn} T={len(w)} "
                  f"w_p10 {np.percentile(w,10):.2f} p50 {np.median(w):.2f} "
                  f"p90 {np.percentile(w,90):.2f} frac_dn(<0.5) "
                  f"{(w<0.5).mean():.2f} frac_up(>1.5) {(w>1.5).mean():.2f}",
                  flush=True)
np.savez_compressed("analysis/manifold/weight_traj.npz", **out)
print("WTRAJ done", flush=True)
