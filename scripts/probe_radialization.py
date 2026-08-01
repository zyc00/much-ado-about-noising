"""Radialization evidence battery. Sub-claims:
P1 demos cross the gate band multi-directionally (low circular resultant R in much of band)
P2 MSE = local circular MEAN of crossing directions; attenuation grows as dispersion grows
P3 MIP = mode-seeking at high-dispersion states (closer to an individual demo direction,
   further from the mean; magnitude NOT attenuated with dispersion)
Saves per-state arrays to analysis/traj_vis/radialization.npz for the figure script.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from scipy.spatial import cKDTree
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
BP = slice(7, 10); FP = slice(21, 24)

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

# demo band states: frame lateral pos rel gate + demonstrator lateral action
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs = []
demo_xy, demo_axy = [], []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1 - 1, FP] - ov[r1 - 1, BP])
offx = np.median(np.stack(offs), 0)
h.close()
h = h5py.File(DSP, "r")
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    for t in range(c1 + 5, r1):
        gate = ov[t, BP] + offx
        d = ov[t, FP] - gate
        if 0.01 <= np.linalg.norm(d[:2]) < 0.08 and abs(d[2]) <= 0.05:
            demo_xy.append(d[:2]); demo_axy.append(a[t, :2])
h.close()
demo_xy = np.stack(demo_xy); demo_axy = np.stack(demo_axy)
dtree = cKDTree(demo_xy)
print(f"demo band points: {len(demo_xy)}", flush=True)

# P1: local direction dispersion over the band
u = demo_axy / (np.linalg.norm(demo_axy, axis=1, keepdims=True) + 1e-9)
Rbar = []
for i in range(0, len(demo_xy), 5):
    _, idx = dtree.query(demo_xy[i], k=12)
    Rbar.append(np.linalg.norm(u[idx].mean(0)))
Rbar = np.array(Rbar)
print(f"P1 demo direction dispersion: R p50={np.median(Rbar):.2f} | frac R<0.7 (crossing) = {(Rbar<0.7).mean():.0%} | frac R<0.5 = {(Rbar<0.5).mean():.0%}", flush=True)

# shared rollout band states
SHARED = []
for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]
        L = min(len(o), len(a)); gc = a[:L, 6]
        run = 0
        for t in range(L):
            run = run + 1 if gc[t] >= 0 else 0
            if run < 10: continue
            s = o[t]
            d = s[FP] - (s[BP] + offx)
            if 0.01 <= np.linalg.norm(d[:2]) < 0.08 and abs(d[2]) <= 0.05:
                SHARED.append(np.stack([o[max(t-1,0)], o[t]]))
        i += 1
SHARED = np.stack(SHARED[::2])
sxy = np.stack([s[1, FP][:2] - (s[1, BP] + offx)[:2] for s in SHARED])
print(f"shared band states: {len(SHARED)}", flush=True)

# local demo stats at shared states
loc_mean, loc_R, loc_dirs, loc_perpmag = [], [], [], []
for p in sxy:
    _, idx = dtree.query(p, k=12)
    ud = u[idx]
    m = ud.mean(0)
    loc_R.append(np.linalg.norm(m))
    loc_mean.append(m / (np.linalg.norm(m) + 1e-9))
    loc_dirs.append(ud)
    r = p / (np.linalg.norm(p) + 1e-9)
    av = demo_axy[idx]
    loc_perpmag.append(np.median(np.abs(av[:, 0] * r[1] - av[:, 1] * r[0])))
loc_mean = np.stack(loc_mean); loc_R = np.array(loc_R); loc_perpmag = np.array(loc_perpmag)

out = {"sxy": sxy, "loc_R": loc_R}
MODELS = [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt"),
          ("hMSE_s5001", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5001_success60.pt"),
          ("hMIP_s5001", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5001_success80.pt")]
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    acts = []
    for i in range(0, len(SHARED), 512):
        x = torch.tensor(no.normalize(SHARED[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = enc({"state": x}, None)
            y = fm.get_velocity(torch.zeros(len(x), device=dev), torch.zeros(len(x), 16, 10, device=dev), e)
        acts.append(ds.undo_transform_action(na.unnormalize(y.cpu().numpy()))[:, START, :3])
    A = np.concatenate(acts)[:, :2]
    out[f"{name}_axy"] = A
    ua = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    cos_mean = (ua * loc_mean).sum(1)
    minang = np.array([np.degrees(np.arccos(np.clip((ua[i][None] * loc_dirs[i]).sum(1), -1, 1))).min() for i in range(len(ua))])
    r = sxy / (np.linalg.norm(sxy, axis=1, keepdims=True) + 1e-9)
    perp = np.abs(A[:, 0] * r[:, 1] - A[:, 1] * r[:, 0])
    att = perp / (loc_perpmag + 1e-9)
    for tag, m in [("crossing R<0.6", loc_R < 0.6), ("aligned R>0.8", loc_R > 0.8)]:
        if m.sum() < 20: continue
        print(f"RADIAL {name} [{tag}] (n={m.sum()}): cos-to-local-MEAN p50={np.median(cos_mean[m]):+.2f} | min-angle-to-a-demo-dir p50={np.median(minang[m]):.0f}deg | perp/GTperp p50={np.median(att[m]):.2f}", flush=True)
np.savez("analysis/traj_vis/radialization.npz", **out)
print("RADIAL-DONE", flush=True)
