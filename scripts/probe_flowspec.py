"""Spectrum battery for flow_timeline endpoint: settle/all PCA (PR/erank/top1) +
encoder-J kappa10/PR/k90 + COLCOS at settle canon. Same conventions as prior probes."""
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
Q = np.load("scripts/jac_queries.npz")

h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase = [], []
for k in keys[:150]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        r = t - c1
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3)))
h.close()
W = np.stack(W); phase = np.array(phase)

def specsum(ev):
    ev = np.clip(ev, 0, None); ev = ev[ev > 1e-12 * max(ev.max(), 1e-30)]
    p = ev / ev.sum()
    return float(1.0/(p**2).sum()), float(np.exp(-(p*np.log(p)).sum())), float(np.sort(p)[-1])

ag.load("logs/flow_timeline/models/model_latest.pt", load_optimizer=False); ag.eval()
enc = ag.encoder_ema
outs = []
for i in range(0, len(W), 512):
    x = torch.tensor(no.normalize(W[i:i+512]), device=dev, dtype=torch.float32)
    with torch.no_grad():
        e = enc({"state": x}, None)
    outs.append(e.reshape(len(x), -1))
E = torch.cat(outs)
En = (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy().astype(np.float64)
line = "FLOWSPEC PCA:"
for sn, idx in [("all", np.arange(len(En))), ("settle", np.where(phase == 1)[0])]:
    X = En[idx] - En[idx].mean(0, keepdims=True)
    pr, er, top1 = specsum(np.linalg.eigvalsh((X.T @ X) / len(X)))
    line += f" | {sn}: PR={pr:.2f} erank={er:.2f} top1={top1:.0%}"
print(line, flush=True)
def fea(inp):
    return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
k10s, prj, k90s, pcs, u1s = [], [], [], [], []
for w in Q["canon"]:
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
    S = np.abs(C.T @ C); iu = np.triu_indices(10, 1)
    pcs.append(float(S[iu].mean()))
    U = np.linalg.svd(J, full_matrices=False)[0]
    u1s.append(float(np.abs(C.T @ U[:, 0]).mean()))
print(f"FLOWSPEC JAC settle_canon: k90={np.median(k90s):.0f} PR={np.median(prj):.2f} k10={np.median(k10s):.1f} | COLCOS pair={np.median(pcs):.2f} vs-u1={np.median(u1s):.2f}", flush=True)
print("FLOWSPEC-DONE")
