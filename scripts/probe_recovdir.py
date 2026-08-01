"""Directional recovery probe: displace eef-pos obs dims (44:47) by raw
delta in {5,10,20,50} mm along random unit u; signed servo gain
s = <mean_t (f(x_p)-f(x))_pos, u> / delta  (s<0 restoring, ~0 replay,
s>0 amplifying). Per arm, overall + settle deciles {8,9}. Prints RD lines."""
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
ARMS = ([tuple(a.split(":")) for a in os.environ["RV_ARMS"].split(",")]
        if os.environ.get("RV_ARMS") else [
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt"),
    ("HG", "regression_hetero_gauss",
     "logs/mp200_hg_s1000/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
])
DELTAS = [0.005, 0.01, 0.02, 0.05]
NDIR = 8

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

    res = {d: [] for d in DELTAS}
    res_set = {d: [] for d in DELTAS}
    for S in demos:
        L = len(S)
        idxs = range(1, L - Hn, 4)
        W = np.stack([np.stack([S[i - 1], S[i]]) for i in idxs])
        dec = np.minimum((10 * np.asarray(list(idxs)) / L).astype(int), 9)
        base = pred(W)
        for d in DELTAS:
            for _ in range(NDIR):
                u = rng.standard_normal(3)
                u /= np.linalg.norm(u)
                Wp = W.copy()
                Wp[:, :, 44:47] += d * u
                dp = pred(Wp) - base
                s = (dp.mean(axis=1) @ u) / d
                res[d].extend(s)
                res_set[d].extend(s[np.isin(dec, [8, 9])])
    for d in DELTAS:
        a_, b_ = np.asarray(res[d]), np.asarray(res_set[d])
        print(f"RD {arm} d={int(d * 1000)}mm gain p50 {np.median(a_):+.4f} "
              f"p10 {np.percentile(a_, 10):+.4f} p90 "
              f"{np.percentile(a_, 90):+.4f} | settle p50 "
              f"{np.median(b_):+.4f} frac_amp {np.mean(b_ > 0.05):.2f}",
              flush=True)
print("RD done", flush=True)
