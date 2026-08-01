"""Formation dynamics of the two extension-defect components, swept over training snapshots.
Per snapshot:
  (a) FOLD: 10-NN phase composition (cosine, encoder_ema embedding) of the 108
      failure-closure-window queries against a 13k-state phase-labeled bank.
  (b) RECOVERY: restoring component of the predicted action at neutral annulus queries
      (canonical settle states + known physical eef offsets): pullback = a_pos . (-d_hat);
      positive = commanded motion opposes the offset.
Env: SNAP_GLOB, LOSS, TAG, QDIR1, QDIR2.
"""
import os, sys, glob, re
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    _ov = ["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"]
    _ov += [o for o in os.environ.get("EXTRA_OV", "").split(",") if o]
    cfg = compose(config_name="main", overrides=_ov)
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

# ---- bank with phase labels (matches emb_neighbors.py conventions)
h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase, LBL = [], [], []
for k in keys[:150]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        r = t - c1
        ph = 0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3))
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(ph); LBL.append(a[t])
h.close()
W = np.stack(W); phase = np.array(phase); LBL = np.stack(LBL)
# fixed within-phase pairs for metric-alignment statistic
prng = np.random.RandomState(1)
PAIRS = {}
for pname, pid in [("all", None), ("settle", 1)]:
    idx = np.arange(len(W)) if pid is None else np.where(phase == pid)[0]
    ii = idx[prng.randint(0, len(idx), 30000)]; jj = idx[prng.randint(0, len(idx), 30000)]
    m = ii != jj
    if pid is None:
        m &= (phase[ii] == phase[jj])   # within-phase only
    PAIRS[pname] = (ii[m], jj[m], np.linalg.norm(LBL[ii[m]] - LBL[jj[m]], axis=1))
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64); ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-12))

# ---- fold queries: failure closure-window states from deployed dumps
FAILS = {21003,21006,21008,21012,21018,21020,21022,21023,21032,21034,21036,21045,21047,21048,21052,21054,21062,21063}
qF = []
for pat in [os.environ.get("QDIR1", "/tmp/msebulk") + "/ep_*.npz", os.environ.get("QDIR2", "/tmp/cfdump_mse2") + "/ep_*.npz"]:
    for f in sorted(glob.glob(pat)):
        z = np.load(f)
        if "obs" not in z.files: continue
        sd = int(z["seed"]); ki, ch, obs = z["ki"], z["chA"], z["obs"]
        g = ch[:, :, 6]; cl = None
        for i in range(1, len(ki)):
            if g[i-1].max() < 0 and g[i].max() >= 0: cl = i; break
        if cl is None or not (sd in FAILS and int(z["asm"]) == 0): continue
        for i in range(max(0, cl-1), min(cl+5, len(obs))):
            qF.append(obs[i])
qF = np.stack(qF)
# metric-faithfulness reference: obs-space distances from each failure query to a fixed bank subset
frng = np.random.RandomState(2)
FSUB = frng.choice(len(W), 2000, replace=False)
_sd = np.concatenate([W[:, 1]]).std(0) + 1e-6
DOBS = np.stack([np.linalg.norm((W[FSUB, 1] - q[1][None]) / _sd, axis=1) for q in qF])  # (108, 2000)

# ---- recovery queries: canonical states + known eef offsets (exact kinematics)
Q = np.load("scripts/jac_queries.npz")
EEF_POS = [44, 45, 46]; EEF_QUAT = [47, 48, 49, 50]
REL = {"b": [0, 1, 2], "f": [14, 15, 16], "t": [28, 29, 30]}
def quat2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
rng = np.random.RandomState(0)
rq, rdir = [], []
for w0 in Q["canon"]:
    for rep in range(3):
        w = w0.copy()
        d = rng.randn(3); d /= np.linalg.norm(d); mag = rng.uniform(0.010, 0.040)
        for fr in range(2):
            R = quat2R(w[fr, EEF_QUAT])
            w[fr, EEF_POS] += d * mag
            for rel in REL.values(): w[fr, rel] += -R.T @ (d * mag)
        rq.append(w); rdir.append(d)
rq = np.stack(rq); rdir = np.stack(rdir)
print(f"bank={len(W)} foldq={len(qF)} recq={len(rq)}", flush=True)

def emb_of(X):
    outs = []
    for i in range(0, len(X), 512):
        x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = ag.encoder_ema({"state": x}, None)
        outs.append(e.reshape(len(x), -1))
    E = torch.cat(outs)
    return E / (E.norm(dim=1, keepdim=True) + 1e-9)

def first_act(X):
    outs = []
    for i in range(0, len(X), 256):
        ot = {"state": torch.tensor(no.normalize(X[i:i+256]), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((len(ot["state"]), 16, int(os.environ.get("ACT_DIM", "10"))), device=dev), obs=ot, use_ema=True)
        outs.append(ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy()))[:, start, :3])
    return np.concatenate(outs)

snaps = sorted(glob.glob(os.environ["SNAP_GLOB"]),
               key=lambda p: int(re.search(r"snap_(\d+)", p).group(1)) if re.search(r"snap_(\d+)", p) else 10**9)
tag = os.environ.get("TAG", "")
for ck in snaps:
    m = re.search(r"snap_(\d+)", ck)
    step = m.group(1) if m else "latest"
    ag.load(ck, load_optimizer=False); ag.eval()
    EB = emb_of(W); EQ = emb_of(qF)
    nb = (EQ @ EB.T).topk(10, dim=1).indices.cpu().numpy()
    ph = phase[nb]
    settle = (ph == 1).mean(); transit = (ph == 2).mean()
    A = first_act(rq)
    pb = -(A * rdir).sum(1)
    EBn = EB.cpu().numpy()
    # (1) effective rank of the embedding (participation ratio of covariance spectrum)
    Xc = EBn - EBn.mean(0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(np.cov(Xc[::2].T)), 0, None)
    pr = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))
    # (2) state decodability: ridge from embedding to current-frame obs, demo-blocked split
    Yd = W[:, 1]
    A = EBn[:10000]; Yt = Yd[:10000]; mu_y = Yt.mean(0)
    Wr = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ (Yt - mu_y))
    pred = EBn[10000:] @ Wr + mu_y
    Yv = Yd[10000:]
    ss = 1 - ((Yv - pred) ** 2).sum(0) / (((Yv - Yv.mean(0)) ** 2).sum(0) + 1e-12)
    FD = [14, 15, 16, 44, 45, 46, 51, 52]
    r2_all = float(np.mean(ss)); r2_fold = float(np.mean(ss[FD]))
    al = {}
    for pname, (ii, jj, dl) in PAIRS.items():
        de = 1.0 - (EBn[ii] * EBn[jj]).sum(1)
        al[pname] = spearman(de, dl)
    EQn = EQ.cpu().numpy()
    faith = np.mean([spearman(1.0 - EBn[FSUB] @ EQn[i], DOBS[i]) for i in range(len(qF))])
    print(f"FOLDSWEEP {tag} step={step}: settle={settle:.0%} transit={transit:.0%} | pullback p50={np.median(pb):+.4f} mean={pb.mean():+.4f} frac>0={(pb>0).mean():.0%} | align_all={al['all']:+.3f} align_settle={al['settle']:+.3f} faith_fail={faith:+.3f} | PR={pr:.1f} r2_all={r2_all:+.3f} r2_fold={r2_fold:+.3f}", flush=True)
print("SWEEP-DONE")
