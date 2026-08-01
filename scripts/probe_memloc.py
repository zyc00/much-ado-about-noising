"""Memorization locus: per-window x tail-decile training residuals at MSE@37k vs MSE@300k
(and MLP-MSE final). Claim under test: the 37k->300k residual reduction is concentrated
in the REACH/CARRY tail (memorizing coarse-phase noise), not the near-gate servo."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys, h5py
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
def load(net):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, f"network={net}",
            "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    cfg.task.horizon = 10 if net == "mlp" else 16
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
# sample windows + tail scores (same construction as probe_tail_location)
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, rows = [], []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    for t in range(1, T - 1, 4):
        rows.append((k, t, ov[t-1], ov[t], a[t], c1 + 5 <= t < r1))
h.close()
off = np.median(np.stack(offs), 0)
X = np.stack([r[3] for r in rows]); A = np.stack([r[4] for r in rows])
wins = []
for (_, _, _, s53, act, held) in rows:
    if not held: wins.append("REACH")
    else:
        v = s53[FP] - (s53[BP] + off)
        wins.append("NEARGATE" if np.linalg.norm(v[:2]) < 0.080 else "CARRY")
wins = np.array(wins)
FX = torch.tensor(X); D = torch.cdist(FX, FX)
nbr = D.topk(9, largest=False).indices[:, 1:].numpy()
tail = np.linalg.norm(A - A[nbr].mean(1), axis=1)
q90 = np.quantile(tail, 0.9)
istail = tail >= q90
MODELS = [("mse37k", "chiunet", "logs/snap_hmse_chi/models/snap_36999.pt"),
          ("mse300k", "chiunet", "logs/snap_hmse_chi/models/snap_299999.pt"),
          ("mlpfinal", "mlp", "logs/hmse_mlp_s5/models/model_latest.pt")]
for name, net, ck in MODELS:
    cfg, ds, ag = load(net)
    ag.load(ck, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    AD = int(cfg.task.act_dim); CH = int(cfg.task.horizon)
    res = np.zeros(len(rows))
    B = 256
    for b in range(0, len(rows), B):
        batch = rows[b:b + B]
        w = np.stack([np.stack([p, c]) for (_, _, p, c, _, _) in batch])
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((len(batch), CH, AD), device=dev), obs=ot, use_ema=True)
        pred = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, 1:2])[:, 0]
        for j, (_, _, _, _, act, _) in enumerate(batch):
            res[b + j] = np.linalg.norm(pred[j] - act)
    line = f"MEMLOC {name}:"
    for w in ["REACH", "CARRY", "NEARGATE"]:
        for tt, tn in [(istail, "tail"), (~istail, "typ")]:
            m = (wins == w) & tt
            if m.sum() > 20:
                line += f" {w}/{tn}={np.median(res[m]):.3f}"
    print(line, flush=True)
print("MEMLOC-DONE")
