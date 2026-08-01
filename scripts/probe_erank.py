"""Effective rank (Roy & Vetterli 2007: exp of Shannon entropy of the normalized spectrum)
of the embedding space, alongside PR (= exp Renyi-2), per phase and pooled.
Two conventions reported: spectrum = covariance eigenvalues (lam, = squared singular values)
and spectrum = singular values (sv, the literal Roy-Vetterli definition).
Embeddings unit-normalized then centered — same convention as probe_dimcount PCA numbers.
"""
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
        ph = 0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3))
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(ph)
h.close()
W = np.stack(W); phase = np.array(phase)
print(f"bank={len(W)}", flush=True)
PHN = {0: "approach", 1: "settle", 2: "transit", 3: "insert"}

def eranks(ev):
    ev = np.clip(ev, 0, None)
    ev = ev[ev > 1e-12 * ev.max()]
    out = {}
    for tag, spec in [("lam", ev), ("sv", np.sqrt(ev))]:
        p = spec / spec.sum()
        H1 = -(p * np.log(p)).sum()
        out[f"erank_{tag}"] = float(np.exp(H1))
        out[f"PR_{tag}"] = float(1.0 / (p ** 2).sum())
    return out

MODELS = [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
          ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt"),
          ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt")]
for d in ["orig_sig001", "orig_sig003", "orig_sig03", "orig_sig10", "sig20", "sig50"]:
    p = f"logs/{d}/models/model_latest.pt"
    if os.path.exists(p): MODELS.append((d, "mip", p))

for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
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
    sets = {"all": np.arange(len(E))}
    for pid, pn in PHN.items(): sets[pn] = np.where(phase == pid)[0]
    line = f"ERANK {name}:"
    for sn, idx in sets.items():
        X = E[idx] - E[idx].mean(0, keepdims=True)
        ev = np.linalg.eigvalsh((X.T @ X) / len(X))
        r = eranks(ev)
        line += f" | {sn}: erank={r['erank_lam']:.2f} PR={r['PR_lam']:.2f} erankSV={r['erank_sv']:.1f}"
    print(line, flush=True)
print("ERANK-DONE")
