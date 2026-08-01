"""Human-data fit-allocation anatomy: per-arm |pred - GT| along full human
demos, stratified by (a) progress decile, (b) local tremor tercile
(within-chunk GT action total variation = noise proxy). Tests the
allocation account: L2 relatively tighter on noisy strata / looser on
smooth strata vs HT/MIP. Training demos (fit) + held-out demos (gen).
Prints HFIT lines. Env: PM_ARMS name:loss:ckpt."""
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
ARMS = [tuple(a.split(":")) for a in os.environ["PM_ARMS"].split(",")]
AS_LO, AS_HI = 1, 9

h = h5py.File(HUM, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
SETS = {"train": names[0:6], "held": names[185:191]}
data = {}
for lab, nm in SETS.items():
    W, GA, PH, TR = [], [], [], []
    for dn in nm:
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
        A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
        L = len(S)
        for i in range(1, L - 10, 2):
            W.append(np.stack([S[i - 1], S[i]]))
            g = A[i - 1 + AS_LO:i - 1 + AS_HI, 0:3]
            GA.append(g)
            PH.append(i / L)
            TR.append(np.abs(np.diff(g, axis=0)).sum())   # tremor proxy
    data[lab] = (np.stack(W), np.stack(GA), np.asarray(PH), np.asarray(TR))
    print(f"HFIT {lab} states {len(W)}", flush=True)
h.close()

for name, loss, ck in ARMS:
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
    H = int(cfg.task.horizon)
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device

    for lab, (W, GA, PH, TR) in data.items():
        R = []
        for i in range(0, len(W), 256):
            xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 256]]),
                              device=dev, dtype=torch.float32)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
            pe = np.asarray(ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, AS_LO:AS_HI]))
            R.append(np.abs(pe[:, :, 0:3] - GA[i:i + 256]).mean(axis=(1, 2)))
        R = np.concatenate(R)
        t1, t2 = np.percentile(TR, [33, 66])
        lo, mid, hi = R[TR < t1], R[(TR >= t1) & (TR < t2)], R[TR >= t2]
        print(f"HFIT {name} {lab} tremor-lo p50 {np.median(lo):.5f} "
              f"mid {np.median(mid):.5f} hi {np.median(hi):.5f} | "
              f"late-slow p50 "
              f"{np.median(R[(PH > 0.5) & (TR < t1)]):.5f}", flush=True)
print("HFIT done", flush=True)
