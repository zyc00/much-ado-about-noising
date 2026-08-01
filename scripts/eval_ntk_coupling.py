"""NTK/influence diagnostic: does the aux (denoising) slice at EDGE states share
parameter-gradient direction with the MAIN slice at OFF-support states?
cos( grad_theta[-n.f(s_off,0,0)] , grad_theta[-n.f(s_edge,a+(1-t)eps,t)] )
Views: MIP main-off x {aux-edge, main-edge, aux-edge-ufix, aux-edgeMISMATCH};
MSE main-off x main-edge. Scalars = inward component of first action (normalized space).
"""
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
DS = os.environ.get("DS", "data/tool_hang_full2ins_2000.hdf5")
PKNOTS = 120
N_PAIRS = int(os.environ.get("PAIRS", "40"))
OFF_CM = float(os.environ.get("OFF_CM", "3.0"))
KNOISE = 4

hf = h5py.File(DS, "r")
keys = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
grid = np.linspace(0, 1, PKNOTS)
tube = []
for k in keys[:300]:
    p = np.asarray(hf[f"data/{k}/obs/robot0_eef_pos"]); t = np.linspace(0, 1, len(p))
    tube.append(np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], 1))
tube = np.stack(tube); center = tube.mean(0); sigma = tube.std(0).mean(1) + 1e-9
tg = np.gradient(center, axis=0); tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12)

# collect edge anchors from held-out demos: (win(2,53) raw, act chunk raw, n_perp, p_idx)
anchors = []
for k in keys[300:420]:
    o = hf[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[kk]) for kk in OK], axis=1).astype(np.float32)
    acts = np.asarray(hf[f"data/{k}/actions"])
    T = len(ov)
    for t0 in range(4, T - 20, 7):
        p_idx = min(int(round(t0 / (T - 1) * (PKNOTS - 1))), PKNOTS - 1)
        dv = ov[t0, POS] - center[p_idx]
        dvp = dv - (dv @ tg[p_idx]) * tg[p_idx]
        d = np.linalg.norm(dvp) / sigma[p_idx]
        if 1.5 <= d < 2.5:
            n_perp = dvp / (np.linalg.norm(dvp) + 1e-12)
            anchors.append((ov[t0-1:t0+1].copy(), acts[t0-1:t0-1+16].copy(), n_perp, p_idx))
    if len(anchors) >= N_PAIRS: break
anchors = anchors[:N_PAIRS]
print(f"anchors={len(anchors)}")

cfgdir = os.path.abspath("examples/configs")
def load(tag, ckpt, loss):
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(DS),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False)
    return cfg, ds, ag

def grad_vec(ag, scalar):
    ps = [p for p in list(ag.flow_map.parameters()) + list(ag.encoder.parameters()) if p.requires_grad]
    gs = torch.autograd.grad(scalar, ps, retain_graph=False, allow_unused=True)
    return torch.cat([ (g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, ps) ])

results = {}
for spec in os.environ["MODELS"].replace("|", ":").split(","):
    tag, ckpt, loss = spec.split(":")[:3]
    cfg, ds, ag = load(tag, ckpt, loss)
    dev = cfg.optimization.device
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    tts = cfg.optimization.t_two_step
    net, enc = ag.flow_map, ag.encoder
    torch.manual_seed(0)
    def inward_scalar(win_raw, n_perp, view, act_raw=None, ufix=False):
        x = torch.tensor(no.normalize(win_raw[None]), device=dev, dtype=torch.float32)
        emb = enc(x, None)
        B1 = torch.zeros(1, device=dev)
        nvec = torch.tensor(n_perp, device=dev, dtype=torch.float32)
        if view == "main":
            a0 = torch.zeros((1, 16, 10), device=dev)
            f = net.get_velocity(B1, a0, emb)
        else:
            if ufix:
                a_n = torch.zeros((1, 16, 10), device=dev)  # fixed constant u (normalized mid-range)
            else:
                pos_, rot_, grip_ = act_raw[:, :3], act_raw[:, 3:6], act_raw[:, 6:]
                rot6 = ds.rotation_transformer.forward(rot_)
                ch = np.concatenate([pos_, rot6, grip_], -1).astype(np.float32)[None]  # (1,16,10)
                a_n = torch.tensor(na.normalize(ch), device=dev, dtype=torch.float32)
            sc = 0.0
            for _ in range(KNOISE):
                noise = torch.randn_like(a_n)
                at = a_n + (1 - tts) * noise
                f = net.get_velocity(B1 + tts, at, emb)
                sc = sc + (-(f[0, 1, :3] * nvec).sum()) / KNOISE
            return sc
        return -(f[0, 1, :3] * nvec).sum()
    rows = {"aux": [], "main": [], "ufix": [], "mismatch": []}
    rng = np.random.RandomState(1)
    for i, (win, actc, n_perp, p_idx) in enumerate(anchors):
        if len(actc) < 16: continue
        # off-support state: displace both frames along n_perp by OFF_CM
        win_off = win.copy(); win_off[:, POS] += (OFF_CM / 100.0) * n_perp
        s_off = inward_scalar(win_off, n_perp, "main")
        g_off = grad_vec(ag, s_off); g_off = g_off / (g_off.norm() + 1e-12)
        for name in (["aux", "main", "ufix", "mismatch"] if loss == "mip" else ["main"]):
            if name == "mismatch":
                j = (i + len(anchors) // 2) % len(anchors)
                w2, a2, n2, _ = anchors[j]
                if len(a2) < 16: continue
                sc = inward_scalar(w2, n2, "aux", a2)
            elif name == "aux":
                sc = inward_scalar(win, n_perp, "aux", actc)
            elif name == "ufix":
                sc = inward_scalar(win, n_perp, "aux", actc, ufix=True)
            else:
                sc = inward_scalar(win, n_perp, "main")
            g2 = grad_vec(ag, sc); g2 = g2 / (g2.norm() + 1e-12)
            rows[name].append(float((g_off * g2).sum()))
    for name, v in rows.items():
        if v:
            v = np.array(v)
            print(f"NTK {tag} main(off)x{name}(edge): n={len(v)} cos_p50={np.median(v):+.3f} mean={v.mean():+.3f} p10={np.percentile(v,10):+.3f}")
print("NTK-DONE")
