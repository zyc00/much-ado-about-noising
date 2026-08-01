"""Hetero-t diagnostics (H1-H2): (a) residual level by window x tail (memorization check
vs the healthy ~0.05 floor); (b) learned sigma(s) map: correlation with window and with
kNN-unpredictability; (c) within-chunk residual heterogeneity (per-step-position RMS) —
justifies per-step scale if strong."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys, h5py
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = os.environ.get("DSP", "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
def load():
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            "optimization.loss_type=regression_hetero_t", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
cfg, ds, ag = load()
dev = cfg.optimization.device
ag.load(os.environ.get("CKPT", "logs/hheterot_s5/models/model_latest.pt"), load_optimizer=False); ag.eval()
# raw windows for classification (raw source -> classify in meters); ds for model inputs
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs = []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
h.close()
off = np.median(np.stack(offs), 0)
N = 4096
idx = np.random.RandomState(0).choice(len(ds), N, replace=False)
OW, AC = [], []
for i in idx:
    b = ds[int(i)]
    o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
    OW.append(o[:2].numpy()); AC.append(b["action"][:16].numpy())
OW = np.stack(OW); AC = np.stack(AC)
no = ds.normalizer["obs"]["state"]
RAW = no.unnormalize(OW)
wins = []
for w in RAW:
    s53 = w[1]
    v = s53[FP] - (s53[BP] + off)
    lat = np.linalg.norm(v[:2])
    wins.append("NEAR" if lat < 0.080 else ("MID" if lat < 0.250 else "FAR"))
wins = np.array(wins)
# kNN unpredictability (normalized obs / actions, correct space)
FX = torch.tensor(OW.reshape(N, -1)); D = torch.cdist(FX, FX)
nbr = D.topk(9, largest=False).indices[:, 1:].numpy()
FA = AC.reshape(N, -1)
tail = np.linalg.norm(FA - FA[nbr].mean(1), axis=1)
# forward: residuals + learned sigma
res = np.zeros(N); sig = np.zeros(N); res_step = np.zeros((N, 16))
B = 256
for b0 in range(0, N, B):
    w = torch.tensor(OW[b0:b0+B], device=dev, dtype=torch.float32)
    a = torch.tensor(AC[b0:b0+B], device=dev)
    t = torch.zeros(len(a), device=dev)
    with torch.no_grad():
        e = ag.encoder(w, None)
        pred, s_raw = ag.flow_map.net(torch.zeros_like(a), t, t, e)
    r = (pred - a)
    res[b0:b0+B] = r.reshape(len(a), -1).norm(dim=1).cpu().numpy() / np.sqrt(160)
    res_step[b0:b0+B] = r.norm(dim=2).cpu().numpy()
    sig[b0:b0+B] = (torch.nn.functional.softplus(s_raw).reshape(len(a), -1).mean(dim=1)).cpu().numpy()
q90 = np.quantile(tail, 0.9); istail = tail >= q90
print("HTDIAG residual (per-dim RMS) by window x tail:")
for wnd in ["FAR", "MID", "NEAR"]:
    m = wins == wnd
    print(f"  {wnd}: tail={np.median(res[m & istail]):.3f} typ={np.median(res[m & ~istail]):.3f} (n={m.sum()})", flush=True)
print(f"HTDIAG global med residual={np.median(res):.3f}  [healthy floor ~0.05; chiunet-L2 memorized 0.076->0.015 range; MLP floor 0.17/sqrt? scale differs]")
from scipy.stats import spearmanr
rho_t, _ = spearmanr(sig, tail); 
print(f"HTDIAG sigma(s) diagnostics: med sigma={np.median(sig):.3f} | spearman(sigma, kNN-unpredictability)={rho_t:+.2f}")
for wnd in ["FAR", "MID", "NEAR"]:
    m = wins == wnd
    print(f"  sigma {wnd}: {np.median(sig[m]):.3f}", flush=True)
hetero = res_step.std(axis=1) / (res_step.mean(axis=1) + 1e-9)
print(f"HTDIAG within-chunk residual heterogeneity (CV across 16 steps): p50={np.median(hetero):.2f} p90={np.quantile(hetero,0.9):.2f}")
print("HTDIAG-DONE")
