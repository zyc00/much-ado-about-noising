"""THE concrete behavioral contrast: lateral centering command at the insertion gate.

States: pooled from ALL rollout dumps (both models, both seeds) — frame held, pre-assembly,
lateral gate distance in [1,8]cm, vertical within 4cm of gate height. Both policies evaluated
on the SAME states (counterfactual), plus the demonstrators' executed actions in the same
band from the demos (GT reference).
Readout per lateral-distance bin: commanded lateral velocity TOWARD the gate (mm/step,
positive = centering), perpendicular component, and total command magnitude.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
BP = slice(7, 10); FP = slice(21, 24)
BINS = [(0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 0.08)]

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

# gate offset from demos + GT reference behavior in the band
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, gt_states = [], {b: [] for b in range(4)}
demo_rows = []
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
    demo_rows.append((ov, a, c1, r1))
off = np.median(np.stack(offs), 0)
h.close()

def band_metrics_from_actions(states, acts):
    """states: (N,53) current-frame obs; acts: (N,3) raw pos actions. Returns per-bin lists."""
    out = {b: {"cen": [], "perp": [], "mag": []} for b in range(4)}
    for s, ap in zip(states, acts):
        target = s[BP] + off
        delta = target - s[FP]
        dl = np.linalg.norm(delta[:2]); dv = abs(delta[2])
        if dv > 0.04: continue
        for b, (lo, hi) in enumerate(BINS):
            if lo <= dl < hi:
                u = delta[:2] / (dl + 1e-9)
                out[b]["cen"].append(float(ap[:2] @ u) * 1000)
                out[b]["perp"].append(float(abs(ap[0]*u[1] - ap[1]*u[0])) * 1000)
                out[b]["mag"].append(float(np.linalg.norm(ap)) * 1000)
                break
    return out

# GT reference
gt = {b: {"cen": [], "perp": [], "mag": []} for b in range(4)}
for ov, a, c1, r1 in demo_rows:
    res = band_metrics_from_actions(ov[c1+5:r1], a[c1+5:r1, :3])
    for b in range(4):
        for k2 in gt[b]: gt[b][k2] += res[b][k2]
line = "GATESERVO GT-demos:"
for b, (lo, hi) in enumerate(BINS):
    if gt[b]["cen"]:
        line += f" | {int(lo*100)}-{int(hi*100)}cm cen={np.median(gt[b]['cen']):+.1f} perp={np.median(gt[b]['perp']):.1f} mag={np.median(gt[b]['mag']):.1f} (n={len(gt[b]['cen'])})"
print(line, flush=True)

# shared band states from ALL dumps (frame held = post first sustained close, pre-assembly proxy: use all steps, filter by band)
SHARED = []
for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    f = f"analysis/traj_vis/human_{name}.npz"
    if not os.path.exists(f): continue
    z = np.load(f)
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]
        L = min(len(o), len(a))
        gcmd = a[:L, 6]
        held = np.zeros(L, dtype=bool)
        run = 0
        for t in range(L):
            run = run + 1 if gcmd[t] >= 0 else 0
            held[t] = run >= 10
        for t in range(L):
            if not held[t]: continue
            s = o[t]
            target = s[BP] + off
            delta = target - s[FP]
            dl = np.linalg.norm(delta[:2]); dv = abs(delta[2])
            if 0.01 <= dl < 0.08 and dv <= 0.04:
                SHARED.append(np.stack([o[max(t-1,0)], o[t]]))
        i += 1
SHARED = np.stack(SHARED[::2])
print(f"shared band states: {len(SHARED)}", flush=True)

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
    res = band_metrics_from_actions(SHARED[:, 1], A)
    line = f"GATESERVO {name}:"
    for b, (lo, hi) in enumerate(BINS):
        if res[b]["cen"]:
            line += f" | {int(lo*100)}-{int(hi*100)}cm cen={np.median(res[b]['cen']):+.1f} perp={np.median(res[b]['perp']):.1f} mag={np.median(res[b]['mag']):.1f} (n={len(res[b]['cen'])})"
    print(line, flush=True)
print("GATESERVO-DONE")
