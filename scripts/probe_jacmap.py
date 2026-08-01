"""Jacobian maps for visualization: full J = d(executed pose chunk, 24) /
d(obs window, 106) for several arms at three state categories (approach,
contact pocket, annulus). Saves per-state J matrices + svals to npz.
Env: JM_ARMS name:loss:ckpt (comma list), JM_N per category.
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
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
TAG = "l2mp200v2"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [tuple(a.split(":")) for a in os.environ["JM_ARMS"].split(",")]
NC = int(os.environ.get("JM_N", "8"))
AS_LO, AS_HI = 1, 9

# --- states: approach (prog .05-.25) and pocket (prog .3-.5) from held-out
#     demos; annulus (2<=d<10) from the rollout capture --------------------
rng = np.random.RandomState(0)
h = h5py.File(HELD, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2100]
appr, pock = [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    L = len(S)
    if L < 30:
        continue
    ia = rng.randint(max(1, int(0.05 * L)), int(0.25 * L))
    ip = rng.randint(int(0.30 * L), int(0.50 * L))
    appr.append(np.stack([S[ia - 1], S[ia]]))
    pock.append(np.stack([S[ip - 1], S[ip]]))
h.close()
z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
RW = np.concatenate([z[f"W{sd}"] for sd in seeds])
RD = np.concatenate([z[f"D{sd}"] for sd in seeds])
ann_idx = np.where((RD >= 2) & (RD < 10))[0]
CATS = {"approach": np.stack(appr[:NC]),
        "pocket": np.stack(pock[:NC]),
        "annulus": RW[rng.choice(ann_idx, NC, replace=False)]}
if os.environ.get("JM_CATS"):
    CATS = {k: v for k, v in CATS.items()
            if k in os.environ["JM_CATS"].split(",")}
print(f"JM cats {[ (k, len(v)) for k, v in CATS.items() ]}", flush=True)

out = {}
for name, loss, ck in ARMS:
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
    H = int(cfg.task.horizon)
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    no = ds.normalizer["obs"]["state"]
    dev = cfg.optimization.device

    for cat, WS in CATS.items():
        Js, svs = [], []
        for w in WS:
            xn = torch.tensor(no.normalize(w).reshape(-1), device=dev,
                              dtype=torch.float32)

            def f(inp):
                a0 = torch.zeros((1, H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0,
                             {"state": inp.reshape(1, 2, 53)})
                return an[0, AS_LO:AS_HI, 0:3].reshape(-1)   # 24 outputs

            J = torch.autograd.functional.jacobian(f, xn, vectorize=True)
            J = J.reshape(24, 106).double().cpu().numpy()
            Js.append(J)
            svs.append(np.linalg.svd(J, compute_uv=False))
        out[f"{name}_{cat}_J"] = np.stack(Js)
        out[f"{name}_{cat}_sv"] = np.stack(svs)
        sv = np.stack(svs)
        pr = (sv ** 2).sum(1) ** 2 / (sv ** 4).sum(1)
        print(f"JM {name} {cat} svPR_p50 {np.median(pr):.2f} "
              f"top_sv_share {np.median(sv[:,0]**2/(sv**2).sum(1)):.2f}",
              flush=True)
np.savez_compressed(os.environ.get("JM_OUT", "analysis/manifold/jacmap.npz"), **out)
print("JM done", flush=True)
