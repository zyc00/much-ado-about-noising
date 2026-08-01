"""On-support edge-state inward margin: is MIP already more inward than MSE at
GENUINE data edge states? amp = -n_perp . (pi(s) - a_label(s)) per d-band.
No env: evaluate policies on recorded obs windows of dataset demos."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)
H = 16
DS = os.environ.get("DS", "data/tool_hang_full2ins_2000.hdf5")
PKNOTS = 120
N_DEMOS = int(os.environ.get("DEMOS", "60"))
STRIDE = int(os.environ.get("STRIDE", "4"))

hf = h5py.File(DS, "r")
keys = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
grid = np.linspace(0, 1, PKNOTS)
tube = []
for k in keys[:300]:
    p = np.asarray(hf[f"data/{k}/obs/robot0_eef_pos"])
    t = np.linspace(0, 1, len(p))
    tube.append(np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], 1))
tube = np.stack(tube)
center = tube.mean(0); sigma = tube.std(0).mean(1) + 1e-9
tang = np.gradient(center, axis=0)
tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)

cfgdir = os.path.abspath("examples/configs")
agents = []
for spec in os.environ["MODELS"].replace("|", ":").split(","):
    tag, ckpt, loss = spec.split(":")[:3]
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(os.environ.get("NORMDS", DS)),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    dev = cfg.optimization.device; start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    act0 = torch.randn((1, H, 10), generator=g).to(dev)
    def mk(ag=ag, no=no, na=na, ds=ds, dev=dev, act0=act0, start=start, AS=AS):
        def f(win):
            x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                an = ag.sample(act_0=act0.clone(), obs=x, use_ema=True)
            return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :3]
        return f
    agents.append((tag, mk()))
    print(f"MODEL {tag} loaded", flush=True)

rows = []
for k in keys[300:300 + N_DEMOS]:   # demos NOT used for tube stats
    o = hf[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[kk]) for kk in OK], axis=1).astype(np.float32)
    acts = np.asarray(hf[f"data/{k}/actions"])[:, :3]
    T = len(ov)
    for t in range(2, T - 1, STRIDE):
        p_idx = min(int(round(t / (T - 1) * (PKNOTS - 1))), PKNOTS - 1)
        dv = ov[t, POS] - center[p_idx]
        # tangent-orthogonalized normal
        dvp = dv - (dv @ tang[p_idx]) * tang[p_idx]
        nn = np.linalg.norm(dvp)
        if nn < 1e-9: continue
        n_perp = dvp / nn
        dval = nn / sigma[p_idx]
        a_lab = acts[t]
        Rlab = -float(n_perp @ a_lab)
        win = [ov[t - 1], ov[t]]
        rec = [dval, Rlab]
        for tag, f in agents:
            am = f(win)
            rec.append(-float(n_perp @ am))          # R_model
            rec.append(-float(n_perp @ (am - a_lab)))  # amp vs label
        rows.append(rec)
A = np.array(rows)
print(f"total states={len(A)}")
BANDS = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 9)]
names = [t for t, _ in agents]
for lo, hi in BANDS:
    m = (A[:, 0] >= lo) & (A[:, 0] < hi)
    if m.sum() < 20: continue
    B = A[m]
    line = f"ONSUP d=[{lo},{hi}) n={m.sum()} Rlab_p50={np.median(B[:,1]):+.4f} Rlab_mean={B[:,1].mean():+.4f} | "
    for i, nm in enumerate(names):
        line += f"{nm}: R={np.median(B[:,2+2*i]):+.4f} amp={np.median(B[:,3+2*i]):+.4f} amp_mean={B[:,3+2*i].mean():+.4f}  "
    print(line)
print("ONSUP-DONE")
