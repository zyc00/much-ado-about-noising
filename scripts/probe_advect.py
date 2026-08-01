"""Displacement-dynamics test: is the embedding reorganized by LABEL relationships or
STATE relationships? For fixed bank-state pairs, regress the change in embedding distance
between consecutive snapshots on (z-scored) label distance and state distance,
controlling for the current embedding distance:
    d_phi(t+1) - d_phi(t) = b_phi * d_phi(t) + b_L * d_label + b_S * d_state + c
Also prints the phase-pair mean embedding-distance matrix per snapshot.
Env: SNAP_GLOB, LOSS, TAG."""
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
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device
no = ds.normalizer["obs"]["state"]

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
sdv = W[:, 1].std(0) + 1e-6
# masked arms: restrict the state-distance reference to live dims
_mask_env = os.environ.get("OBS_MASK")
LIVE = np.array([d for d in range(53) if not (_mask_env and d in {int(x) for x in _mask_env.split(",")})])

rng = np.random.RandomState(3)
N = len(W)
ii = rng.randint(0, N, 60000); jj = rng.randint(0, N, 60000)
m = ii != jj
ii, jj = ii[m], jj[m]
# canonical label metric: exact training space (pos + rot6d + grip, pipeline-normalized)
_rot6 = ds.rotation_transformer.forward(LBL[:, 3:6])
_X10 = np.concatenate([LBL[:, :3], _rot6, LBL[:, 6:7]], axis=1).astype(np.float32)
_L10 = ds.normalizer["action"].normalize(_X10)
d_label = np.linalg.norm(_L10[ii] - _L10[jj], axis=1)
d_state = np.linalg.norm(((W[ii, 1] - W[jj, 1]) / sdv)[:, LIVE], axis=1)
zL = (d_label - d_label.mean()) / d_label.std()
zS = (d_state - d_state.mean()) / d_state.std()
PH_NAMES = ["appr", "settle", "transit", "insert"]

def rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)
def partials(d_phi):
    rp, rl, rs = rank(d_phi), rank(d_label), rank(d_state)
    rp = (rp - rp.mean()) / rp.std(); rl = (rl - rl.mean()) / rl.std(); rs = (rs - rs.mean()) / rs.std()
    # residualize
    rl_s = rl - (rl @ rs / (rs @ rs)) * rs
    rs_l = rs - (rs @ rl / (rl @ rl)) * rl
    p_label = float((rp @ rl_s) / (np.linalg.norm(rp) * np.linalg.norm(rl_s)))
    p_state = float((rp @ rs_l) / (np.linalg.norm(rp) * np.linalg.norm(rs_l)))
    return p_label, p_state

def emb_of(X):
    outs = []
    for i in range(0, len(X), 512):
        x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = ag.encoder_ema({"state": x}, None)
        outs.append(e.reshape(len(x), -1))
    E = torch.cat(outs)
    return (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy()

snaps = sorted(glob.glob(os.environ["SNAP_GLOB"]),
               key=lambda p: int(re.search(r"snap_(\d+)", p).group(1)) if re.search(r"snap_(\d+)", p) else 10**9)
tag = os.environ.get("TAG", "")
prev_d = None; prev_step = None
# INIT row: random weights (no checkpoint load)
E0 = emb_of(W)
d0 = 1.0 - (E0[ii] * E0[jj]).sum(1)
pl0, ps0 = partials(d0)
def _sp0(a, b):
    x = rank(a); y = rank(b)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
print(f"PARTIAL {tag} step=0(init): marg_LABEL={_sp0(d0, d_label):+.3f} partial_LABEL={pl0:+.3f} marg_STATE={_sp0(d0, d_state):+.3f} partial_STATE={ps0:+.3f}", flush=True)
for ck in snaps:
    mm = re.search(r"snap_(\d+)", ck)
    step = mm.group(1) if mm else "latest"
    ag.load(ck, load_optimizer=False); ag.eval()
    E = emb_of(W)
    d_phi = 1.0 - (E[ii] * E[jj]).sum(1)
    # phase-pair mean distances
    row = []
    for a_ in range(4):
        for b_ in range(a_, 4):
            sel = ((phase[ii] == a_) & (phase[jj] == b_)) | ((phase[ii] == b_) & (phase[jj] == a_))
            row.append(f"{PH_NAMES[a_]}-{PH_NAMES[b_]}={d_phi[sel].mean():.3f}")
    print(f"PHASEDIST {tag} step={step}: " + " ".join(row), flush=True)
    pl, ps = partials(d_phi)
    # marginal (uncontrolled) Spearman on the same pair set
    def sp(a, b):
        x = rank(a); y = rank(b)
        x = x - x.mean(); y = y - y.mean()
        return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
    ml = sp(d_phi, d_label); ms = sp(d_phi, d_state)
    print(f"PARTIAL {tag} step={step}: marg_LABEL={ml:+.3f} partial_LABEL={pl:+.3f} marg_STATE={ms:+.3f} partial_STATE={ps:+.3f}", flush=True)
    # stratified (nonparametric) conditional correlations: bin the confounder into deciles,
    # Spearman within each bin, report the weighted mean
    def strat(target_ref, confounder):
        qs = np.quantile(confounder, np.linspace(0, 1, 11))
        vals, wts = [], []
        for b in range(10):
            m2 = (confounder >= qs[b]) & (confounder <= qs[b+1])
            if m2.sum() < 200: continue
            x = rank(d_phi[m2]); y = rank(target_ref[m2])
            x = x - x.mean(); y = y - y.mean()
            vals.append(float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
            wts.append(m2.sum())
        return float(np.average(vals, weights=wts))
    sl = strat(d_label, d_state)   # label correlation, conditioning on state bins
    ss = strat(d_state, d_label)   # state correlation, conditioning on label bins
    print(f"STRAT {tag} step={step}: strat_LABEL|state={sl:+.3f} strat_STATE|label={ss:+.3f}", flush=True)
    if prev_d is not None:
        dd = d_phi - prev_d
        zphi = (prev_d - prev_d.mean()) / prev_d.std()
        X = np.stack([zphi, zL, zS, np.ones_like(zL)], 1)
        beta, *_ = np.linalg.lstsq(X, dd, rcond=None)
        print(f"ADVECT {tag} {prev_step}->{step}: b_phi={beta[0]:+.4f} b_LABEL={beta[1]:+.4f} b_STATE={beta[2]:+.4f} (dd_mean={dd.mean():+.4f})", flush=True)
    prev_d = d_phi; prev_step = step
print("ADVECT-DONE")
