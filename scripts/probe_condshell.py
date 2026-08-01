"""v3 mediator test: directional-gain spectrum SHAPE in the near shell.
For each arm, at on-support and annulus (2<=d<10) states from the rollout
capture, K=16 random obs directions, FD gains d_i = ||f(x+eps v)-f(x)||^2
(eps=0.05): report CV^2(d) (2/PR of the local gain spectrum; rank-1
domination -> 2) and the dominant-direction share. Prints CSH lines.
Env: CS_ARMS name:loss:ckpt:dataset, CS_TAG (capture), CS_N.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

ARMS = [tuple(a.split(":")) for a in os.environ["CS_ARMS"].split(",")]
TAG = os.environ.get("CS_TAG", "l2mp200v2")
NST = int(os.environ.get("CS_N", "160"))
K = 16
EPS = 0.05

z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
RW = np.concatenate([z[f"W{sd}"] for sd in seeds])
RD = np.concatenate([z[f"D{sd}"] for sd in seeds])
rng = np.random.RandomState(0)
SETS = {}
for lab, m in (("on", RD < 2), ("ann", (RD >= 2) & (RD < 10))):
    idx = np.where(m)[0]
    SETS[lab] = RW[rng.choice(idx, min(NST, len(idx)), replace=False)]
print(f"CSH on {len(SETS['on'])} ann {len(SETS['ann'])}", flush=True)

for name, loss, ck, dset in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset), "network=chiunet",
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
    no = ds.normalizer["obs"]["state"]
    dev = cfg.optimization.device

    def act(WS):
        out = []
        for i in range(0, len(WS), 256):
            xb = torch.tensor(np.asarray(WS[i:i + 256], dtype=np.float32),
                              device=dev)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
            out.append(an.reshape(len(xb), -1).cpu().numpy())
        return np.concatenate(out)

    for lab, WS in SETS.items():
        Xn = np.stack([no.normalize(w) for w in WS])         # (N, 2, 53)
        base = act(Xn)
        D = np.zeros((len(Xn), K))
        for k in range(K):
            v = rng.randn(*Xn.shape).astype(np.float32)
            v = v / (np.linalg.norm(v.reshape(len(Xn), -1), axis=1)
                     .reshape(-1, 1, 1) + 1e-9)
            pert = act(Xn + EPS * v)
            D[:, k] = ((pert - base) ** 2).mean(axis=1)
        cv2 = D.var(axis=1) / (D.mean(axis=1) ** 2 + 1e-12)
        share = D.max(axis=1) / (D.sum(axis=1) + 1e-12)
        print(f"CSH {name} {lab} cv2_p50 {np.median(cv2):.2f} p90 "
              f"{np.percentile(cv2,90):.2f} | eff_rank_p50 "
              f"{np.median(2.0/np.maximum(cv2,1e-6)):.2f} | topshare_p50 "
              f"{np.median(share):.2f}", flush=True)
print("CSH done", flush=True)
