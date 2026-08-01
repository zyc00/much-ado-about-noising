"""Off-support anatomy on human data.

(A) Response field: at align1/align2/carry1 canon states, apply kinematically consistent
    eef offsets (10-40mm, rel columns adjusted via R_eef), measure the DIFFERENTIAL
    restoring component Rdiff = -n_hat . (a_pert - a_base)[:3] in raw action units
    (differential readout — absolute projection is blind in the near zone, PART LXV/edgejac).
    Models: hMSE_s5 (1-pass), hMIP_s5 (step1 and 2-step deployment path).
(B) Tube-distance stats of the 20-ep rollout dumps vs a 40-demo anchor tree (z-scored obs):
    maxd p50/90/95, frac steps d>2 / d>4, per-model.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
EEF_POS = [44, 45, 46]; EEF_QUAT = [47, 48, 49, 50]
REL = {"b": [0, 1, 2], "f": [14, 15, 16], "t": [28, 29, 30]}
def quat2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

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

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, wid = [], []
anch = []
for di, k in enumerate(keys):
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    if di < 40: anch.append(ov)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    c2 = next((t for t in cl if r1 and t > r1), None)
    for t in range(1, T, 4):
        w = -1
        if r1 and r1 - 30 <= t < r1: w = 3
        elif c1 + 5 <= t < (r1 - 30 if r1 else T): w = 2
        elif c2 and T - 40 <= t: w = 7
        if w >= 0:
            W.append(np.stack([ov[t-1], ov[t]])); wid.append(w)
h.close()
W = np.stack(W); wid = np.array(wid)
rng = np.random.RandomState(0)

# (B) tube stats first (no GPU)
cl_ = np.concatenate(anch, 0)
mu, sig = cl_.mean(0), cl_.std(0) + 1e-6
tree = cKDTree((cl_ - mu) / sig)
for name in ["hMSE_s5", "hMIP_s5"]:
    f = f"analysis/traj_vis/human_{name}.npz"
    if not os.path.exists(f): print(f"TUBE {name}: dump missing"); continue
    z = np.load(f)
    maxds, fr2, fr4 = [], [], []
    i = 0
    while f"ep{i}_obs" in z.files:
        obs = z[f"ep{i}_obs"]
        d, _ = tree.query((obs - mu) / sig)
        maxds.append(d.max()); fr2.append((d >= 2).mean()); fr4.append((d >= 4).mean())
        i += 1
    if maxds:
        print(f"TUBE {name}: maxd p50={np.median(maxds):.1f} p90={np.quantile(maxds,0.9):.1f} | frac d>=2 {np.mean(fr2):.2%} | d>=4 {np.mean(fr4):.2%} (n={len(maxds)} eps)", flush=True)

# (A) response field
MODELS = [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt")]
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    tts = cfg.optimization.t_two_step
    def act_of(X, twostep):
        outs = []
        for i in range(0, len(X), 256):
            x = torch.tensor(no.normalize(X[i:i+256]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                e = enc({"state": x}, None)
                y = fm.get_velocity(torch.zeros(len(x), device=dev), torch.zeros(len(x), 16, 10, device=dev), e)
                if twostep:
                    y = fm.get_velocity(torch.full((len(x),), tts, device=dev), y, e)
            outs.append(ds.undo_transform_action(na.unnormalize(y.cpu().numpy()))[:, START, :3])
        return np.concatenate(outs)
    for wq, wnm in [(3, "align1"), (7, "align2"), (2, "carry1")]:
        idx = np.where(wid == wq)[0]
        canon = W[idx[rng.choice(len(idx), min(40, len(idx)), replace=False)]]
        base, pert, nhat = [], [], []
        for w0 in canon:
            for rep in range(3):
                w = w0.copy()
                d = rng.randn(3); d /= np.linalg.norm(d); mag = rng.uniform(0.010, 0.040)
                for fr in range(2):
                    R = quat2R(w[fr, EEF_QUAT])
                    w[fr, EEF_POS] += d * mag
                    for rel in REL.values(): w[fr, rel] += -R.T @ (d * mag)
                base.append(w0); pert.append(w); nhat.append(d)
        base = np.stack(base); pert = np.stack(pert); nhat = np.stack(nhat)
        for twostep in ([False, True] if loss == "mip" else [False]):
            a0 = act_of(base, twostep); a1 = act_of(pert, twostep)
            rdiff = -np.sum(nhat * (a1 - a0), axis=1)
            lbl = "2step" if twostep else "step1"
            print(f"RESP {name} {lbl} {wnm}: Rdiff p50={np.median(rdiff)*1000:+.2f}mm frac>0={np.mean(rdiff>0):.0%} |a_pert-a_base| p50={np.median(np.linalg.norm(a1-a0,axis=1))*1000:.1f}mm", flush=True)
print("OFF-DONE")
