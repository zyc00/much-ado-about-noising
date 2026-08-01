"""Label-distance variant check for the partial metric: is partial_LABEL driven by the
gripper channel? Variants: full (original), std (per-dim standardized), nogrip (pos+rot
only), grip (gripper channel only). Env: SNAP_GLOB, LOSS, TAG."""
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
W, LBL = [], []
for k in keys[:150]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    for t in range(1, len(g), 2):
        W.append(np.stack([ov[t-1], ov[t]])); LBL.append(a[t])
h.close()
W = np.stack(W); LBL = np.stack(LBL)
sdv = W[:, 1].std(0) + 1e-6
asd = LBL.std(0) + 1e-6
print("action per-dim std:", np.array2string(asd, precision=3))

rng = np.random.RandomState(3)
N = len(W)
ii = rng.randint(0, N, 60000); jj = rng.randint(0, N, 60000)
m = ii != jj; ii, jj = ii[m], jj[m]
d_state = np.linalg.norm((W[ii, 1] - W[jj, 1]) / sdv, axis=1)
# exact training-space labels: [pos, rot6d, grip] normalized as the loss sees them
rot6 = ds.rotation_transformer.forward(LBL[:, 3:6])
X10 = np.concatenate([LBL[:, :3], rot6, LBL[:, 6:7]], axis=1).astype(np.float32)
na = ds.normalizer["action"]
L10 = na.normalize(X10)
DL = {
    "full":   np.linalg.norm(LBL[ii] - LBL[jj], axis=1),
    "TRAIN":  np.linalg.norm(L10[ii] - L10[jj], axis=1),
    "TRAINnogrip": np.linalg.norm(L10[ii][:, :9] - L10[jj][:, :9], axis=1),
    "std":    np.linalg.norm((LBL[ii] - LBL[jj]) / asd, axis=1),
    "nogrip": np.linalg.norm(LBL[ii][:, :6] - LBL[jj][:, :6], axis=1),
    "grip":   np.abs(LBL[ii][:, 6] - LBL[jj][:, 6]),
}
def rank(x): return np.argsort(np.argsort(x)).astype(np.float64)
def sp(a, b):
    x = rank(a); y = rank(b); x -= x.mean(); y -= y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
def partial(dphi, dl):
    rp, rl, rs = rank(dphi), rank(dl), rank(d_state)
    rp = (rp-rp.mean())/rp.std(); rl = (rl-rl.mean())/rl.std(); rs = (rs-rs.mean())/rs.std()
    rl_s = rl - (rl @ rs / (rs @ rs)) * rs
    return float((rp @ rl_s) / (np.linalg.norm(rp) * np.linalg.norm(rl_s)))

def emb_of(X):
    outs = []
    for i in range(0, len(X), 512):
        x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = ag.encoder_ema({"state": x}, None)
        outs.append(e.reshape(len(x), -1))
    E = torch.cat(outs)
    return (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy()

for ck in sorted(glob.glob(os.environ["SNAP_GLOB"])):
    mm = re.search(r"snap_(\d+)", ck)
    step = mm.group(1) if mm else "latest"
    ag.load(ck, load_optimizer=False); ag.eval()
    E = emb_of(W)
    dphi = 1.0 - (E[ii] * E[jj]).sum(1)
    line = f"LBLVAR {os.environ.get('TAG','')} step={step}:"
    for name, dl in DL.items():
        line += f" {name}: marg={sp(dphi, dl):+.3f} part={partial(dphi, dl):+.3f} |"
    print(line, flush=True)
print("LBLVAR-DONE")
