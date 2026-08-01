"""Training-set fit anatomy on human data: hMSE vs hMIP (step1 + 2-step), per task window
and per amplitude bin, against a kNN aleatoric floor.

For each (obs window, label) pair: residual = || pred_firstslot - label || in normalized
10-d action space (pos-only also reported). Floor: k=5 state-NN label mean (excluding self,
excluding same-demo temporal neighbors +-4) — irreducible noise + local smoothness.
Windows: approach1/settle1/carry1/align1/between/settle2/carry2/align2.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
WNAMES = ["approach1", "settle1", "carry1", "align1", "between", "settle2", "carry2", "align2"]

def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; dev = cfg.optimization.device
START = cfg.task.obs_steps - 1

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, wid, A10, demo_id, tstep = [], [], [], [], []
for di, k in enumerate(keys):
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    c2 = next((t for t in cl if r1 and t > r1), None)
    rc = ds.rotation_transformer.forward(a[:, 3:6])
    a10 = na.normalize(np.concatenate([a[:, :3], rc, a[:, 6:7]], axis=1).astype(np.float32))
    for t in range(1, T - 1, 2):
        w = -1
        if t < c1 - 12: w = 0
        elif c1 - 12 <= t < c1 + 5: w = 1
        elif r1 and c1 + 5 <= t < r1 - 30: w = 2
        elif r1 and r1 - 30 <= t < r1: w = 3
        elif r1 and c2 and r1 <= t < c2 - 12: w = 4
        elif c2 and c2 - 12 <= t < c2 + 5: w = 5
        elif c2 and c2 + 5 <= t < T - 40: w = 6
        elif c2 and T - 40 <= t: w = 7
        if w < 0: continue
        W.append(np.stack([ov[t-1], ov[t]])); wid.append(w); A10.append(a10[t]); demo_id.append(di); tstep.append(t)
h.close()
W = np.stack(W); wid = np.array(wid); A10 = np.stack(A10); demo_id = np.array(demo_id); tstep = np.array(tstep)
print(f"pairs={len(W)}", flush=True)

# ---- kNN aleatoric floor
Wn = no.normalize(W).reshape(len(W), -1)
tree = cKDTree(Wn)
dists, idxs = tree.query(Wn, k=12)
floor_res = np.zeros(len(W))
for i in range(len(W)):
    picks = [j for j in idxs[i] if j != i and not (demo_id[j] == demo_id[i] and abs(tstep[j] - tstep[i]) <= 4)][:5]
    floor_res[i] = np.linalg.norm(A10[picks].mean(0) - A10[i])
print("FLOOR per-window RMS: " + ", ".join(
    f"{WNAMES[w]}={np.sqrt((floor_res[wid==w]**2).mean()):.3f}" for w in range(8)), flush=True)

MODELS = [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt")]
amp = np.linalg.norm(A10[:, :3], axis=1)
qs = np.quantile(amp, [0.25, 0.5, 0.75])
ampbin = np.digitize(amp, qs)
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    tts = cfg.optimization.t_two_step
    preds1, preds2 = [], []
    for i in range(0, len(W), 256):
        x = torch.tensor(no.normalize(W[i:i+256]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = enc({"state": x}, None)
            z = torch.zeros(len(x), 16, 10, device=dev)
            t0 = torch.zeros(len(x), device=dev)
            y0 = fm.get_velocity(t0, z, e)
            preds1.append(y0[:, START].cpu().numpy())
            if loss == "mip":
                y1 = fm.get_velocity(torch.full((len(x),), tts, device=dev), y0, e)
                preds2.append(y1[:, START].cpu().numpy())
    P1 = np.concatenate(preds1)
    res1 = np.linalg.norm(P1 - A10, axis=1)
    tag2 = ""
    if preds2:
        P2 = np.concatenate(preds2)
        res2 = np.linalg.norm(P2 - A10, axis=1)
    print(f"FIT {name} step1 per-window RMS: " + ", ".join(
        f"{WNAMES[w]}={np.sqrt((res1[wid==w]**2).mean()):.3f}" for w in range(8)), flush=True)
    if preds2:
        print(f"FIT {name} 2step per-window RMS: " + ", ".join(
            f"{WNAMES[w]}={np.sqrt((res2[wid==w]**2).mean()):.3f}" for w in range(8)), flush=True)
    print(f"FIT {name} step1 amp-bin RMS (Q1..Q4): " + ", ".join(
        f"{np.sqrt((res1[ampbin==b]**2).mean()):.3f}" for b in range(4)) +
        f" | overall {np.sqrt((res1**2).mean()):.3f} p50={np.median(res1):.3f} p90={np.quantile(res1,0.9):.3f}", flush=True)
print("FIT-DONE")
