"""Local Jacobian structure at PASS vs FAIL near-gate rollout states (hetero-t):
J = d(exec action)/d(current obs frame) [10 x 53 for the first exec step].
Report per group: kappa10, PR of singular values, ||J||;
SERVO BLOCKS: lateral gain G_lat = d a_xy / d framepos_xy (2x2) — symmetric-part
eigenvalues (negative = corrective feedback); rot gain norm d a_rot / d framequat."""
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
BP = slice(7, 10); FP = slice(21, 24); FQ = slice(17, 21)
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
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
ag.load(os.environ.get("CKPT", "logs/hheterot_s5/models/model_latest.pt"), load_optimizer=False); ag.eval()
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
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
states = {"PASS": [], "FAIL": []}
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
            ts = [t for t in range(1, len(v)) if 0.010 <= lat[t] < 0.080 and 0.005 <= alt[t] < 0.120]
            step = max(1, len(ts) // 30)
            for t in ts[::step][:30]:
                states[grp].append((o[g0 + t - 1].copy(), o[g0 + t].copy()))
        i += 1
print({k: len(v) for k, v in states.items()}, flush=True)
start = cfg.task.obs_steps - 1
def jac_at(prev, cur):
    w = np.stack([prev, cur])[None]
    wn = no.normalize(w)
    x = torch.tensor(wn, device=dev, dtype=torch.float32)
    def f(cur_frame):
        xx = torch.cat([x[:, :1], cur_frame.view(1, 1, -1)], dim=1)
        e = ag.encoder(xx, None)
        t0 = torch.zeros(1, device=dev)
        pred = ag.flow_map.get_velocity(t0, torch.zeros((1, 16, 10), device=dev), e)
        an = na.unnormalize(pred.cpu().detach().numpy() if False else None) if False else pred
        return pred[0, start + 1]  # first executed action step (10-dim, normalized)
    J = torch.autograd.functional.jacobian(f, x[0, 1].clone(), vectorize=True)
    return J.detach().cpu().numpy()  # (10, 53)
rows = {}
for grp in ["PASS", "FAIL"]:
    K, PR, GLAT_EIG, GROT, JN = [], [], [], [], []
    for prev, cur in states[grp]:
        J = jac_at(prev, cur)
        s = np.linalg.svd(J, compute_uv=False)
        K.append(s[0] / (s[9] + 1e-12)); PR.append((s.sum() ** 2) / (np.sum(s ** 2) + 1e-12)); JN.append(s[0])
        # denormalize direction scales: action cols 0:2 (xy delta), obs dims frame xy = 21:23
        Glat = J[0:2, 21:23]
        Gs = 0.5 * (Glat + Glat.T)
        ev = np.linalg.eigvalsh(Gs)
        GLAT_EIG.append(ev)  # both eigenvalues of symmetric part
        GROT.append(np.linalg.norm(J[3:6, 17:21]))
    GL = np.array(GLAT_EIG)
    rows[grp] = (np.median(K), np.median(PR), np.median(JN),
                 np.median(GL[:, 0]), np.median(GL[:, 1]), np.mean((GL < 0).all(axis=1)),
                 np.median(GROT))
    print(f"FAILJAC {grp}: kappa10={rows[grp][0]:.1f} PR={rows[grp][1]:.1f} |J|={rows[grp][2]:.2f}"
          f" | G_lat sym-eigs p50=({rows[grp][3]:+.3f},{rows[grp][4]:+.3f}) frac both-neg={rows[grp][5]:.2f}"
          f" | |G_rot|={rows[grp][6]:.3f}", flush=True)
print("FAILJAC-DONE")
