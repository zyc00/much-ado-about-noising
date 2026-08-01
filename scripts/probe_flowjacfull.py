"""Pooled full-trajectory encoder-J spectrum + COLCOS for flow endpoint (global counterpart
of FLOWSPEC's settle row); same pooled-state convention as probe_condnum full_traj."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
        "optimization.loss_type=flow", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device
h5f = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
rng = np.random.RandomState(7)
S = []
for di in rng.choice(200, 20, replace=False):
    d = h5f[f"data/demo_{di}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]; T = len(g)
    ov = np.concatenate([np.asarray(d["obs"][q]) for q in OK], axis=1).astype(np.float32)
    for lo, hi in [(c1-40, c1-15), (c1+20, c1+60), (c1+72, T-2)]:
        t = rng.randint(max(1, lo), min(hi, T-1))
        S.append(np.stack([ov[t-1], ov[t]]))
h5f.close()
S = np.stack(S)
ag.load("logs/flow_timeline/models/model_latest.pt", load_optimizer=False); ag.eval()
enc = ag.encoder_ema
def fea(inp):
    return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
k10s, prj, k90s, pcs = [], [], [], []
for w in S:
    x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
    J = torch.autograd.functional.jacobian(fea, x, vectorize=True).squeeze(1).detach().cpu().numpy()
    sv = np.linalg.svd(J, compute_uv=False); e2 = sv ** 2
    cs = np.cumsum(e2) / e2.sum()
    k90s.append(int(np.searchsorted(cs, 0.90) + 1))
    prj.append(float(e2.sum()**2 / (e2**2).sum()))
    k10s.append(float(sv[0] / sv[min(9, len(sv)-1)]))
    gn = np.linalg.norm(J, axis=0); nz = np.where(gn > 1e-8 * gn.max())[0]
    top = nz[np.argsort(gn[nz])[-10:]]
    C = J[:, top] / (np.linalg.norm(J[:, top], axis=0, keepdims=True) + 1e-12)
    Sm = np.abs(C.T @ C); iu = np.triu_indices(10, 1)
    pcs.append(float(Sm[iu].mean()))
print(f"FLOWJACFULL full_traj: k90={np.median(k90s):.0f} PR={np.median(prj):.2f} k10={np.median(k10s):.1f} | COLCOS pair={np.median(pcs):.2f}", flush=True)
print("FLOWJACFULL-DONE")
