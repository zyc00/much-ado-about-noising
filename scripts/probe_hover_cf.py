"""Hover-attractor counterfactual: at DEMO near-gate states, compare policy output under
moving history [o[t-1], o[t]] vs hover history [o[t], o[t]] (same current state, zero
velocity). If hover history collapses command magnitude/closure for MSE-family but not MIP,
the stall is a velocity-conditioned attractor: slow -> weak output -> stays slow."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys, h5py
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, states = [], []
rows = []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP]); rows.append((ov, c1 + 5, r1))
h.close()
off = np.median(np.stack(offs), 0)
for ov, a0, a1 in rows:
    v = ov[a0:a1, FP] - (ov[a0:a1, BP] + off)
    lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
    for t in range(1, len(v), 6):
        if 0.010 <= lat[t] < 0.060 and 0.005 <= alt[t] < 0.120:
            states.append((ov[a0 + t - 1], ov[a0 + t], -v[t, :2] / (lat[t] + 1e-9)))
print(f"demo near states: {len(states)}", flush=True)
def load(loss, extra=()):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"] + list(extra))
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
MODELS = [("hMSE", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt", ()),
          ("hMIP", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt", ()),
          ("hrotaux", "regression", "logs/hrotaux_s5/models/model_latest.pt", ("+task.rot_indicator=true", "task.act_dim=13"))]
for mname, loss, ck, extra in MODELS:
    cfg, ds, ag = load(loss, extra)
    AD = int(cfg.task.act_dim)
    ag.load(ck, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    torch.manual_seed(0)
    res = {}
    for variant in ["move", "hover"]:
        nets, cosi, rots = [], [], []
        B = 64
        for b in range(0, len(states), B):
            batch = states[b:b + B]
            if variant == "move":
                w = np.stack([np.stack([p, c]) for p, c, u in batch])
            else:
                w = np.stack([np.stack([c, c]) for p, c, u in batch])
            ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                an = ag.sample(act_0=torch.randn((len(batch), 16, AD), device=dev), obs=ot, use_ema=True)
            acts = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])
            for j, (p, c, u) in enumerate(batch):
                net = acts[j][:, 0:2].sum(0); n = np.linalg.norm(net)
                nets.append(n); rots.append(np.linalg.norm(acts[j][:, 3:6], axis=1).mean())
                if n > 1e-6: cosi.append(float(np.dot(net, u)) / n)
        res[variant] = (np.median(nets), np.median(cosi), np.median(rots))
    m, hv = res["move"], res["hover"]
    print(f"HOVER {mname}: |net_xy| move={m[0]:.3f} hover={hv[0]:.3f} ratio={hv[0]/m[0]:.2f}"
          f" | cos_inward move={m[1]:+.2f} hover={hv[1]:+.2f}"
          f" | |rot| move={m[2]:.3f} hover={hv[2]:.3f} ratio={hv[2]/m[2]:.2f}", flush=True)
print("HOVER-DONE")
