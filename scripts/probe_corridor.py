"""Goal-seeking vs corridor-rejoining: at shared 3-8cm band states, does each policy's
command point at the GATE or at the nearest point of the demonstrated frame-approach
corridor? cos(cmd, to-gate) vs cos(cmd, to-corridor), per model."""
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

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, corridor = [], []
for k in keys[:60]:
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
    for t in range(c1 + 5, r1):
        corridor.append(ov[t, FP] - ov[t, BP])   # frame pos rel base = base-frame path
h.close()
off = np.median(np.stack(offs), 0)
corridor = np.stack(corridor)
ctree = cKDTree(corridor)

SHARED = []
for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    f = f"analysis/traj_vis/human_{name}.npz"
    if not os.path.exists(f): continue
    z = np.load(f)
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]
        L = min(len(o), len(a)); gcmd = a[:L, 6]
        run = 0
        for t in range(L):
            run = run + 1 if gcmd[t] >= 0 else 0
            if run < 10: continue
            s = o[t]
            delta = (s[BP] + off) - s[FP]
            dl = np.linalg.norm(delta[:2])
            if 0.03 <= dl < 0.08 and abs(delta[2]) <= 0.05:
                SHARED.append(np.stack([o[max(t-1,0)], o[t]]))
        i += 1
SHARED = np.stack(SHARED[::2])
print(f"outer-band states: {len(SHARED)}", flush=True)

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
    A = np.concatenate(acts)
    cg, cc = [], []
    for s, ap in zip(SHARED[:, 1], A):
        gate = (s[BP] + off) - s[FP]
        rel = s[FP] - s[BP]
        d, idx = ctree.query(rel)
        corr = corridor[idx] - rel      # toward nearest corridor point (rel-base coords)
        n = np.linalg.norm(ap) + 1e-9
        cg.append(float(ap @ gate / (n * np.linalg.norm(gate) + 1e-9)))
        if np.linalg.norm(corr) > 0.005:
            cc.append(float(ap @ corr / (n * np.linalg.norm(corr) + 1e-9)))
    print(f"CORRIDOR {name}: cos(cmd,to-gate) p50={np.median(cg):+.2f} | cos(cmd,to-corridor) p50={np.median(cc):+.2f} (n={len(cg)}/{len(cc)})", flush=True)
print("CORRIDOR-DONE")
