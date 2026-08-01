"""Step-1 vs step-2 output comparison on dataset states: the bias pattern.
(1) AMPLITUDE: ||output||/||label|| per window (FAR/MID/NEAR) and channel (pos/rot) —
    conditional-mean attenuation predicts step1 < 1 (worst in noisy fine channels),
    step2 ~ 1 (instance reconstruction).
(2) ERROR: residual RMS per window/channel for both steps (where does step2 improve).
(3) PREDICTABILITY: fraction of each step's error explained by obs-kNN mean of errors —
    step2 should have removed the state-predictable part of step1's error.
(4) CORRECTION DIRECTION: cos(step2-step1, label-step1) — does the correction point at truth."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys, h5py
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = os.environ.get("DSP", "x")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
def load():
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            "optimization.loss_type=mip", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
cfg, ds, ag = load()
dev = cfg.optimization.device
ag.load(os.environ["CK_MIP"], load_optimizer=False); ag.eval()
no = ds.normalizer["obs"]["state"]
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
RAW = no.unnormalize(OW)
wins = []
for w in RAW:
    s53 = w[1]
    v = s53[FP] - (s53[BP] + off)
    lat = np.linalg.norm(v[:2])
    wins.append("NEAR" if lat < 0.080 else ("MID" if lat < 0.250 else "FAR"))
wins = np.array(wins)
P1 = np.zeros_like(AC); P2 = np.zeros_like(AC)
B = 256
for b0 in range(0, N, B):
    w = torch.tensor(OW[b0:b0+B], device=dev, dtype=torch.float32)
    a = torch.tensor(AC[b0:b0+B], device=dev)
    t0 = torch.zeros(len(a), device=dev)
    with torch.no_grad():
        e = ag.encoder(w, None)
        a0 = ag.flow_map.get_velocity(t0, torch.zeros_like(a), e)
        tt = torch.full((len(a),), 0.9, device=dev)
        a2 = ag.flow_map.get_velocity(tt, a0, e)
    P1[b0:b0+B] = a0.cpu().numpy(); P2[b0:b0+B] = a2.cpu().numpy()
CH = {"pos": slice(0, 3), "rot": slice(3, 9)}
for wnd in ["FAR", "MID", "NEAR"]:
    m = wins == wnd
    row = f"S12BIAS {wnd} (n={m.sum()}):"
    for cn, sl in CH.items():
        la = np.linalg.norm(AC[m][:, :, sl], axis=2)
        r1a = np.linalg.norm(P1[m][:, :, sl], axis=2) / (la + 1e-8)
        r2a = np.linalg.norm(P2[m][:, :, sl], axis=2) / (la + 1e-8)
        e1 = np.linalg.norm((P1 - AC)[m][:, :, sl], axis=2).mean()
        e2 = np.linalg.norm((P2 - AC)[m][:, :, sl], axis=2).mean()
        row += f" {cn}: amp1={np.median(r1a):.2f} amp2={np.median(r2a):.2f} err1={e1:.4f} err2={e2:.4f} |"
    print(row, flush=True)
# (3) predictability of errors + (4) correction direction
E1 = (P1 - AC).reshape(N, -1); E2 = (P2 - AC).reshape(N, -1)
FX = torch.tensor(OW.reshape(N, -1)); D = torch.cdist(FX, FX)
nbr = D.topk(9, largest=False).indices[:, 1:].numpy()
for tag, E in [("step1", E1), ("step2", E2)]:
    pred = E[nbr].mean(1)
    ev = 1 - np.sum((E - pred) ** 2) / np.sum(E ** 2)
    print(f"S12BIAS {tag}: err RMS={np.sqrt((E**2).mean()):.4f} | kNN-predictable frac of err var={ev:.2f}", flush=True)
corr = (P2 - P1).reshape(N, -1); tru = (AC.reshape(N, -1) - P1.reshape(N, -1))
cos = np.sum(corr * tru, axis=1) / (np.linalg.norm(corr, axis=1) * np.linalg.norm(tru, axis=1) + 1e-9)
print(f"S12BIAS correction direction: cos(step2-step1, truth-step1) p50={np.median(cos):+.2f} frac>0.5={np.mean(cos>0.5):.2f}", flush=True)
print("S12BIAS-DONE")
