"""Per-window gradient-mass allocation (scripted): does the task-critical window get its
fair share of the training gradient? For each sample: window (REACH / CARRY / NEARGATE)
+ per-sample representation-gradient norm under the loss view. Report per window:
sample share vs gradient-mass share (allocation ratio = mass share / sample share)."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys, h5py
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = os.environ["DSP"]
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
# window classification from raw demos (gate offset from release frames)
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, rows = [], []
for k in keys[:300]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    for t in range(1, T - 1, 3):
        rows.append((ov[t-1], ov[t], t, c1 + 5 <= t < r1))
    if len(rows) > 6000: break
h.close()
off = np.median(np.stack(offs), 0)
wins = []
for (_, s53, t, held) in rows:
    if not held: wins.append("REACH")
    else:
        v = s53[FP] - (s53[BP] + off)
        wins.append("NEARGATE" if np.linalg.norm(v[:2]) < 0.080 else "CARRY")
wins = np.array(wins)
NET = "chiunet"
def load():
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, f"network={NET}",
            "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
cfg, ds, ag = load()
dev = cfg.optimization.device
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
# actions: use dataset normalizer on demo action chunks — approximate with nearest ds sample?
# simpler: rebuild chunks from the same raw demos via ds sampler indices is complex; instead
# normalize raw 7D->10D chunk via dataset pipeline is heavy. Use per-step single actions
# tiled? NO — use the ds directly: sample ds items and classify via their obs window.
N = 6000
idx = np.random.RandomState(0).choice(len(ds), N, replace=False)
OWs, ACs = [], []
for i in idx:
    b = ds[int(i)]
    o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
    OWs.append(o[:2].numpy()); ACs.append(b["action"][:16].numpy())
OW = np.stack(OWs); AC = np.stack(ACs)
# classify each ds sample by its current raw obs (denormalized? ds obs are raw pre-normalizer)
RAW = no.unnormalize(OW)
wins2 = []
for w in RAW:
    s53 = w[1]
    v = s53[FP] - (s53[BP] + off)
    lat = np.linalg.norm(v[:2])
    wins2.append("NEAR" if lat < 0.080 else ("MID" if lat < 0.250 else "FAR"))
wins2 = np.array(wins2)
for tag, ck, view in [("L2@MSEpt", os.environ["CK_MSE"], "v1"), ("MIPv2@MIPpt", os.environ["CK_MIP"], "v2")]:
    ag.load(ck, load_optimizer=False); ag.eval()
    gns = np.zeros(N)
    B = 256
    for b0 in range(0, N, B):
        w = torch.tensor(OW[b0:b0+B], device=dev, dtype=torch.float32)
        a = torch.tensor(AC[b0:b0+B], device=dev)
        e = ag.encoder(w, None).detach().requires_grad_(True)
        if view == "v2":
            tt = torch.full((len(a),), 0.9, device=dev)
            tot = 0
            for r in range(4):
                gg = torch.Generator(device="cpu").manual_seed(700 + r)
                eps = torch.randn(a.shape, generator=gg).to(dev)
                pred = ag.flow_map.get_velocity(tt, a + 0.1 * eps, e)
                tot = tot + (((pred - a) / 0.1) ** 2).sum(dim=(1, 2))
            li = tot / 4
        else:
            t0 = torch.zeros(len(a), device=dev)
            pred = ag.flow_map.get_velocity(t0, torch.zeros_like(a), e)
            li = ((pred - a) ** 2).sum(dim=(1, 2))
        gr = torch.autograd.grad(li.sum(), e)[0]
        gns[b0:b0+B] = gr.reshape(len(a), -1).norm(dim=1).detach().cpu().numpy()
    tot = gns.sum()
    line = f"WINALLOC {tag}:"
    for w in ["FAR", "MID", "NEAR"]:
        m = wins2 == w
        if m.sum() < 10: continue
        sshare = m.mean(); gshare = gns[m].sum() / tot
        line += f" {w}: samples={100*sshare:.0f}% gradmass={100*gshare:.0f}% alloc={gshare/sshare:.2f} med|g|={np.median(gns[m]):.4f} |"
    print(line, flush=True)
print("WINALLOC-DONE")
