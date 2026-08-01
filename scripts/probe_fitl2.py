"""Both fit-anatomy tables in L2 metric: per-sample squared error in
NORMALIZED action space, mean over full horizon x 10 dims (the training-loss
quantity), EMA nets. Script: per-decile + pocket/nonpocket. Human: tremor
terciles + late-slow. Prints FL2 lines."""
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

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POCKET = {3, 4, 8, 9}
REGIMES = {
    "script": ("data/tool_hang_full2ins_mp_200.hdf5", 8, [
        ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
        ("HT", "regression_hetero_t",
         "logs/mp200_ht_s1000/models/snap_300000.pt"),
        ("HG", "regression_hetero_gauss",
         "logs/mp200_hg_s1000/models/snap_300000.pt"),
        ("NONOISE", "mip_nonoise",
         "logs/mp200_nonoise_s1000/models/snap_300000.pt"),
        ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
    ]),
    "human": ("data/tool_hang_human_lowdim_up.hdf5", 6, [
        ("hMSE", "regression", "logs/hbase2/models/model_latest.pt"),
        ("hHT", "regression_hetero_t",
         "logs/hheterot_s1000/models/model_latest.pt"),
        ("hMIP", "mip", "logs/hmip0/models/model_latest.pt"),
    ]),
}

for regime, (dpath, nd, arms) in REGIMES.items():
    h = h5py.File(dpath, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
    demos = []
    for dn in names[:nd]:
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
        A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
        demos.append((dn, S, A))
    h.close()
    for arm, loss, ck in arms:
        with initialize_config_dir(version_base=None,
                                   config_dir=os.path.abspath(
                                       "examples/configs")):
            cfg = compose(config_name="main", overrides=[
                "task=tool_hang_ph_state_delta_legacy",
                "+task.dataset_path=" + os.path.abspath(dpath),
                "network=chiunet", f"optimization.loss_type={loss}",
                "optimization.auto_resume=false", "log.wandb_mode=disabled"])
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

        R, DEC, TR, PH = [], [], [], []
        for dn, S, A in demos:
            L = len(S)
            hi = L - Hn
            W = np.stack([np.stack([S[i - 1], S[i]]) for i in range(1, hi)])
            Tn = np.stack([na.normalize(fwd(A[i - 1:i - 1 + Hn]))
                           for i in range(1, hi)])
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
            DEC.append(np.minimum((10 * np.arange(1, hi) / L).astype(int), 9))
            PH.append(np.arange(1, hi) / L)
            TR.append(np.asarray([np.abs(np.diff(
                A[i:i + 8, 0:3], axis=0)).sum() for i in range(1, hi)]))
        R = np.concatenate(R)
        DEC, TR, PH = map(np.concatenate, (DEC, TR, PH))
        if regime == "script":
            pk = np.isin(DEC, list(POCKET))
            dl = " ".join(f"d{d} {np.median(R[DEC == d]):.2e}"
                          for d in range(10))
            print(f"FL2 script {arm} perdecile {dl}", flush=True)
            print(f"FL2 script {arm} pocket p50 {np.median(R[pk]):.2e} "
                  f"nonpocket {np.median(R[~pk]):.2e} ratio "
                  f"{np.median(R[pk]) / np.median(R[~pk]):.2f} | mean "
                  f"{R.mean():.2e} p90 {np.percentile(R, 90):.2e}", flush=True)
        else:
            t1, t2 = np.percentile(TR, [33, 66])
            print(f"FL2 human {arm} tremor-lo p50 "
                  f"{np.median(R[TR < t1]):.2e} mid "
                  f"{np.median(R[(TR >= t1) & (TR < t2)]):.2e} hi "
                  f"{np.median(R[TR >= t2]):.2e} | late-slow "
                  f"{np.median(R[(PH > 0.5) & (TR < t1)]):.2e} | mean "
                  f"{R.mean():.2e} p90 {np.percentile(R, 90):.2e}", flush=True)
print("FL2 done", flush=True)
