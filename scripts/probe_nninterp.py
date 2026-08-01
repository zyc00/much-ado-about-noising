"""Deployed-state NN interpolation test (two-attractor hypothesis).
Queries: obs windows from swdump trajectories at rel in {-24,-16,-8,-2} w.r.t. closure.
For each query: policy action (rotation channels) + 10-NN in the policy's training set
(z-scored obs metric). Reports per rel bin: NN phase composition, and the least-squares
interpolation coefficient alpha solving  pred_rot ~ (1-a)*NN_approach_rot + a*transit_dir_scaled.
Env: CKPT, LOSS, DS, DUMP (swdump dir), TAG."""
import os, sys, glob
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(os.environ["DS"]), "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); ag.load(os.environ["CKPT"], load_optimizer=False); ag.eval()
dev = cfg.optimization.device; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
OKEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

# ---- training bank with phase labels ----
ht = h5py.File(os.environ["DS"], "r")
tkeys = sorted(ht["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:300]
bobs, bact, brel = [], [], []
for k in tkeys:
    d = ht[f"data/{k}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    ov = np.concatenate([np.asarray(d["obs"][q]) for q in OKEYS], axis=1).astype(np.float32)
    A = np.asarray(d["actions"])
    for t in range(1, len(g)):
        bobs.append(ov[t]); bact.append(A[t]); brel.append(t - c1)
ht.close()
BO = np.stack(bobs); BA = np.stack(bact); BR = np.array(brel)
sdv = BO.std(0) + 1e-6
BOz = BO / sdv
def phase_of(r):
    if r < 0: return "approach"
    if r < 16: return "closure"
    if r < 70: return "transit"
    return "insert"
BP = np.array([phase_of(r) for r in BR])
tr = BA[(BP == "transit") & (np.linalg.norm(BA[:, 3:6], axis=1) > 0.005), 3:6]
tdir = tr.mean(0); tdir /= (np.linalg.norm(tdir) + 1e-12)
tmag = np.linalg.norm(tr, axis=1).mean()
print(f"bank n={len(BO)}  transit dir={np.array2string(tdir, precision=3)} mean|rot|={tmag:.4f}")

# ---- deployed queries from dumps ----
RELQ = [-24, -16, -8, -2]
wins, rq = [], []
for f in sorted(glob.glob(os.path.join(os.environ["DUMP"], "sw_*.npz"))):
    z = np.load(f)
    ca = int(z["closed_at"])
    if ca < 0: continue
    traj = z["obs_traj"]
    for r in RELQ:
        t = ca + r + 1
        if 2 <= t < len(traj):
            wins.append(traj[t-1:t+1]); rq.append(r)
W = np.stack(wins); RQ = np.array(rq)
ot = {"state": torch.tensor(no.normalize(W), device=dev, dtype=torch.float32)}
with torch.no_grad():
    an = ag.sample(act_0=torch.randn((len(W), 16, 10), device=dev), obs=ot, use_ema=True)
P = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy()))[:, start]

tag = os.environ.get("TAG", "")
for r in RELQ:
    m = RQ == r
    comp = {p: 0.0 for p in ["approach", "closure", "transit", "insert"]}
    alphas, res_pr = [], []
    for i in np.where(m)[0]:
        qz = W[i, 1] / sdv
        dists = np.linalg.norm(BOz - qz[None], axis=1)
        nn = np.argsort(dists)[:10]
        for j in nn: comp[BP[j]] += 1
        appr = nn[BP[nn] == "approach"]
        base = BA[appr, 3:6].mean(0) if len(appr) else BA[nn, 3:6].mean(0)
        # least-squares alpha: pred_rot = base + alpha * (tmag*tdir - base)
        v = tmag * tdir - base
        alpha = float((P[i, 3:6] - base) @ v / (v @ v + 1e-12))
        alphas.append(alpha)
        res = P[i, 3:6] - base
        res_pr.append(float(res @ tdir))
    tot = sum(comp.values()) + 1e-9
    cs = " ".join(f"{p}={comp[p]/tot:.2f}" for p in comp)
    print(f"NNINT {tag} rel{r:+d}: NN[{cs}] alpha p50={np.median(alphas):+.3f} p90={np.percentile(alphas,90):+.3f}  res·tdir p50={np.median(res_pr):+.5f}")
print("NNINT-DONE")
