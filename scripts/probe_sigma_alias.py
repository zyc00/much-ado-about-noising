"""VERIFICATION of the history x hetero-t synergy mechanism: does the learned
sigma at ALIASED states (pose-matched but motion-mixed) shrink when the policy
is trained with longer obs history? Compares trmha_ht (os2) vs trmha_os8 (os8)
on transport-mh. Aliased = pose-kNN neighbor set contains BOTH paused and
moving members (speed from obs eef finite differences, model-free labels).
Prediction: sigma(aliased)/sigma(pure) elevated under os2, flattened under
os8. Envs: none (paths fixed)."""
import os

os.environ["MUJOCO_GL"] = "egl"
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import h5py
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
      "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]

def build(os_steps):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=transport_mh_state_abs", "network=chiunet",
            "optimization.loss_type=regression_hetero_t", "optimization.auto_resume=false",
            "log.wandb_mode=disabled", f"task.obs_steps={os_steps}"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 59
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    return cfg, ds, ag

cfg2, ds2, ag2 = build(2)
DSP = ds2.dataset_path if hasattr(ds2, "dataset_path") else None
# resolve the h5 the config used
from huggingface_hub import hf_hub_download
DSP = hf_hub_download(repo_id="ChaoyiPan/mip-dataset", filename="robomimic/transport/mh/low_dim_abs.hdf5", repo_type="dataset")

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:150]
OBS = {}
for di, k in enumerate(keys):
    o = h[f"data/{k}/obs"]
    n = len(h[f"data/{k}/actions"])
    OBS[di] = np.concatenate([np.asarray(o[q]).reshape(n, -1) for q in OK], axis=1).astype(np.float32)
h.close()
# eef pos dims: object block first
obj_dim = OBS[0].shape[1] - (3 + 4 + 2) * 2
E0 = obj_dim; E1 = obj_dim + 3
F0 = obj_dim + 9; F1 = obj_dim + 12
def speed(di, t):
    d0 = OBS[di][t + 1, E0:E1] - OBS[di][t, E0:E1]
    d1 = OBS[di][t + 1, F0:F1] - OBS[di][t, F0:F1]
    return np.linalg.norm(np.concatenate([d0, d1]))

S = [(di, t) for di in OBS for t in range(8, len(OBS[di]) - 12)]
rng = np.random.RandomState(0)
S = [S[i] for i in rng.choice(len(S), min(5000, len(S)), replace=False)]
Dm = np.array([di for di, t in S])
spd = np.array([speed(di, t) for di, t in S])
paused = spd < np.quantile(spd, 0.25)
moving = spd > np.median(spd)
X = np.stack([OBS[di][t] for di, t in S])
Xs = torch.tensor(X / (X.std(0) + 1e-6))
D = torch.cdist(Xs, Xs)
D[torch.tensor(Dm[:, None] == Dm[None, :])] = 1e9
NN = D.topk(8, largest=False).indices.numpy()
fp = paused[NN].mean(1); fm = moving[NN].mean(1)
aliased = (np.minimum(fp, fm) >= 0.25)
pure = (np.maximum(fp, fm) >= 0.85)
print(f"n={len(S)} aliased={aliased.sum()} pure={pure.sum()} (obs_dim={OBS[0].shape[1]}, obj={obj_dim})", flush=True)

def sig_of(ag, ds, cfg, os_steps, ckpt):
    ag.load(ckpt, load_optimizer=False)
    ag.encoder.eval(); ag.flow_map.eval()
    no = ds.normalizer["obs"]["state"]
    dev = cfg.optimization.device
    out = []
    W = np.stack([OBS[di][t - os_steps + 1:t + 1] for di, t in S])
    with torch.no_grad():
        for b0 in range(0, len(W), 256):
            wb = torch.tensor(no.normalize(W[b0:b0 + 256]), device=dev, dtype=torch.float32)
            emb = ag.encoder(wb, None)
            t0 = torch.zeros(len(wb), device=dev)
            y0 = torch.zeros((len(wb), 16, cfg.task.act_dim), device=dev)
            _, s_raw = ag.flow_map.net(y0, t0, t0, emb)
            sb = torch.nn.functional.softplus(s_raw).reshape(len(wb), -1).mean(dim=1) + 1e-3
            out.append(sb.cpu().numpy())
    return np.concatenate(out)

SNAP = os.environ.get("SNAP", "model_latest")
for name, os_steps, ckpt in [("os2", 2, f"logs/trmha_ht_s1000/models/{SNAP}.pt"),
                              ("os8", 8, f"logs/trmha_os8_s1000/models/{SNAP}.pt")]:
    cfg, ds, ag = (cfg2, ds2, ag2) if os_steps == 2 else build(8)
    sg = sig_of(ag, ds, cfg, os_steps, ckpt)
    sa, sp = np.median(sg[aliased]), np.median(sg[pure])
    def spear(a, b):
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    alias_score = np.minimum(fp, fm)
    print(f"SIGALIAS {name}: sigma p50 aliased={sa:.4f} pure={sp:.4f} RATIO={sa/max(sp,1e-9):.2f} | "
          f"overall p10/50/90={np.percentile(sg,10):.4f}/{np.median(sg):.4f}/{np.percentile(sg,90):.4f} | "
          f"spearman(sigma, alias-score)={spear(sg, alias_score):.3f}", flush=True)
print("SIGALIAS-DONE")
