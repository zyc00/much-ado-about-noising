"""Why do hetero-t's residual failures fail? For each rollout's near-gate states:
(a) coverage: distance to nearest DEMO obs (normalized space) — are fail states off-support?
(b) the model's own sigma(s) there — does it know it's uncertain?
(c) action quality: cos(policy action, kNN-demo action mean) — coherent-wrong vs weak.
PASS vs FAIL comparison, both hetero-t seeds pooled."""
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
no = ds.normalizer["obs"]["state"]
# demo bank: held near-gate states (raw -> normalized), with actions
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, bank_o, bank_prev, bank_a = [], [], [], []
for k in keys[:200]:
    o = h[f"data/{k}/obs"]
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    for t in range(c1 + 5, r1, 2):
        v = ov[t, FP] - (ov[t, BP] + offs[-1])
        if np.linalg.norm(v[:2]) < 0.10:
            bank_o.append(ov[t]); bank_prev.append(ov[t-1]); bank_a.append(a[t])
h.close()
off = np.median(np.stack(offs), 0)
BO = np.stack(bank_o); BA = np.stack(bank_a)
BOn = no.normalize(np.stack([np.stack([p, c]) for p, c in zip(bank_prev, bank_o)]))
BOflat = torch.tensor(BOn.reshape(len(BO), -1))
print(f"demo near bank: {len(BO)}", flush=True)
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
ag.load(os.environ.get("CKPT", "logs/hheterot_s5/models/model_latest.pt"), load_optimizer=False); ag.eval()
res = {"PASS": {"d": [], "sig": [], "cos": []}, "FAIL": {"d": [], "sig": [], "cos": []}}
for name in ["hheterot_s5", "hheterot_s1000"]:
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        grp = "PASS" if int(m[1]) else "FAIL"
        if g0 is not None:
            v = o[g0:gend, FP] - (o[g0:gend, BP] + off)
            lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
            ts = [t for t in range(1, len(v)) if 0.010 <= lat[t] < 0.080 and 0.005 <= alt[t] < 0.120][::3]
            for t in ts:
                w = np.stack([o[g0 + t - 1], o[g0 + t]])
                wn = no.normalize(w[None])
                q = torch.tensor(wn.reshape(1, -1), dtype=torch.float32)
                dist = torch.cdist(q, BOflat)[0]
                near = dist.topk(8, largest=False)
                res[grp]["d"].append(float(near.values[0]))
                wt = torch.tensor(wn, device=dev, dtype=torch.float32)
                with torch.no_grad():
                    e = ag.encoder(wt, None)
                    t0 = torch.zeros(1, device=dev)
                    pred, s_raw = ag.flow_map.net(torch.zeros((1, 16, 10), device=dev), t0, t0, e)
                sig = float(torch.nn.functional.softplus(s_raw).mean())
                res[grp]["sig"].append(sig)
                # policy action (first exec step, denormalized already in dump a) vs kNN demo action
                pa = a[g0 + t][:6]
                ka = BA[near.indices.numpy()][:, :6].mean(0)
                c = float(np.dot(pa, ka) / (np.linalg.norm(pa) * np.linalg.norm(ka) + 1e-9))
                res[grp]["cos"].append(c)
        i += 1
for grp in ["PASS", "FAIL"]:
    d = np.array(res[grp]["d"]); s = np.array(res[grp]["sig"]); c = np.array(res[grp]["cos"])
    print(f"FAILCOV {grp} (n={len(d)}): support-dist p50={np.median(d):.2f} p90={np.quantile(d,0.9):.2f}"
          f" | sigma p50={np.median(s):.4f} | cos(policy, kNN-demo-action) p50={np.median(c):+.2f} frac<0={np.mean(c<0):.2f}", flush=True)
print("FAILCOV-DONE")
