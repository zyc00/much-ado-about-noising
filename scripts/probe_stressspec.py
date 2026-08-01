"""Spectrum battery for the stressreg arms: encoder-Jacobian kappa10 / PR(s^2) / k90 at
settle canon + embedding PCA PR/erank (settle and all-phase). Same conventions as
probe_dimcount / probe_erank."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
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
    pr = float(1.0 / (p ** 2).sum())
    er = float(np.exp(-(p * np.log(p)).sum()))
    top1 = float(np.sort(p)[-1])
    return pr, er, top1

for name, ck in [("stress03", "logs/orig_stress03/models/model_latest.pt"),
                 ("stress10", "logs/orig_stress10/models/model_latest.pt")]:
    if not os.path.exists(ck):
        print(f"STRESSSPEC {name}: MISSING {ck}", flush=True); continue
    cfg, ds, ag = load("regression_stressreg")
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema
    outs = []
    for i in range(0, len(W), 512):
        x = torch.tensor(no.normalize(W[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = enc({"state": x}, None)
        outs.append(e.reshape(len(x), -1))
    E = torch.cat(outs)
    E = (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy().astype(np.float64)
    line = f"STRESSSPEC {name} PCA:"
    for sn, idx in [("all", np.arange(len(E))), ("settle", np.where(phase == 1)[0])]:
        X = E[idx] - E[idx].mean(0, keepdims=True)
        pr, er, top1 = specsum(np.linalg.eigvalsh((X.T @ X) / len(X)))
        line += f" | {sn}: PR={pr:.2f} erank={er:.2f} top1={top1:.0%}"
    print(line, flush=True)
    def fea(inp):
        return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
    k10s, prs, k90s = [], [], []
    for w in Q["canon"]:
        x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
        J = torch.autograd.functional.jacobian(fea, x, vectorize=True).squeeze(1).detach().cpu().numpy()
        sv = np.linalg.svd(J, compute_uv=False); e2 = sv ** 2
        cs = np.cumsum(e2) / e2.sum()
        k90s.append(int(np.searchsorted(cs, 0.90) + 1))
        prs.append(float(e2.sum() ** 2 / (e2 ** 2).sum()))
        k10s.append(float(sv[0] / sv[min(9, len(sv) - 1)]))
    print(f"STRESSSPEC {name} JAC settle_canon: k90={np.median(k90s):.0f} PR(s2)={np.median(prs):.2f} kappa10={np.median(k10s):.1f}", flush=True)
print("STRESSSPEC-DONE")
