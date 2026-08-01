"""Consensus-vs-echo probe: does the policy pool across demonstrations or echo the nearest one?

For pairs of nearby align1 states from DIFFERENT demos (z-scored distance < 2.5), walk the
policy along the linear state transect s(alpha), alpha in {0,.25,.5,.75,1} (quats renormalized):
  WIGGLE = sum_k ||pi(s_{k+1}) - pi(s_k)|| / ||pi(s_1) - pi(s_0)||   (1 = monotone morph,
           >1 = non-monotone switching = echo field)
  CONS   = cos( pi(s_mid), (a_i + a_j)/2 )      (consensus readout at the midpoint)
  ECHO   = cos( pi(s_mid), a_nearer ) - cos( pi(s_mid), a_farther )  (>0 = echoes nearer demo)
Cells: chiunet-MSE/MIP x seeds s5,s5001 + mlp-MSE/MIP s5001 (+mlp-MSE s1).
Unified-hypothesis predictions: WIGGLE high for chiunet-MSE on BOTH seeds; low for mlp arms
regardless of objective; low/medium for chiunet-MIP. Must survive the seed-reversal curse.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from scipy.spatial import cKDTree
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
QUATS = [slice(3, 7), slice(10, 14), slice(17, 21), slice(24, 28), slice(31, 35), slice(38, 42), slice(47, 51)]

def load(loss, network):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, f"network={network}",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression", "chiunet")
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; dev = cfg.optimization.device
START = cfg.task.obs_steps - 1

# align1 states with labels, per demo
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
S, A, D = [], [], []
for di, k in enumerate(keys[:120]):
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    for t in range(r1 - 35, r1 - 2, 3):
        if t < 1: continue
        S.append(np.stack([ov[t-1], ov[t]])); A.append(a[t, :3]); D.append(di)
h.close()
S = np.stack(S); A = np.stack(A); D = np.array(D)
flat = S[:, 1]; mu, sig = flat.mean(0), flat.std(0) + 1e-6
tree = cKDTree((flat - mu) / sig)
rng = np.random.RandomState(0)
pairs = []
cand = rng.permutation(len(S))
for i in cand:
    dists, idxs = tree.query((flat[i] - mu) / sig, k=12)
    for dd, j in zip(dists[1:], idxs[1:]):
        if D[j] != D[i] and dd < 2.5:
            pairs.append((i, int(j), float(dd))); break
    if len(pairs) >= 250: break
print(f"cross-demo pairs={len(pairs)}", flush=True)

def transect(si, sj, K=5):
    out = []
    for k in range(K):
        al = k / (K - 1)
        s = (1 - al) * si + al * sj
        for fr in range(2):
            for q in QUATS:
                v = s[fr, q]; s[fr, q] = v / (np.linalg.norm(v) + 1e-9)
        out.append(s)
    return np.stack(out)

MODELS = [("chiunet-MSE-s5", "regression", "chiunet", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("chiunet-MIP-s5", "mip", "chiunet", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt"),
          ("chiunet-MSE-s5001", "regression", "chiunet", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5001_success60.pt"),
          ("chiunet-MIP-s5001", "mip", "chiunet", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5001_success80.pt"),
          ("mlp-MSE-s5001", "regression", "mlp", f"{OLD}/tool_hang_ph_state_regression_mlp_512_seed5001_success90.pt"),
          ("mlp-MIP-s5001", "mip", "mlp", f"{OLD}/tool_hang_ph_state_mip_mlp_512_seed5001_success85.pt"),
          ("mlp-MSE-s1", "regression", "mlp", f"{OLD}/tool_hang_ph_state_regression_mlp_512_seed1_success90.pt")]
for name, loss, network, ck in MODELS:
    cfg, ds, ag = load(loss, network)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    CH = 10 if network == "mlp" else 16
    def pol3(X):
        preds = []
        for i in range(0, len(X), 512):
            x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                e = enc({"state": x}, None)
                y = fm.get_velocity(torch.zeros(len(x), device=dev), torch.zeros(len(x), CH, 10, device=dev), e)
            preds.append(y[:, START].cpu().numpy())
        P = np.concatenate(preds)
        return ds.undo_transform_action(na.unnormalize(np.repeat(P[:, None], CH, 1)))[:, 0, :3]
    X = np.concatenate([transect(S[i].copy(), S[j].copy()) for i, j, _ in pairs])
    P = pol3(X).reshape(len(pairs), 5, 3)
    wig, cons, echo = [], [], []
    for pi_, (i, j, dd) in enumerate(pairs):
        seq = P[pi_]
        tv = np.linalg.norm(np.diff(seq, axis=0), axis=1).sum()
        direct = np.linalg.norm(seq[-1] - seq[0]) + 1e-9
        wig.append(tv / direct)
        mid = seq[2]; nm = np.linalg.norm(mid) + 1e-9
        am = (A[i] + A[j]) / 2
        cons.append(float(mid @ am / (nm * (np.linalg.norm(am) + 1e-9))))
        ci = float(mid @ A[i] / (nm * (np.linalg.norm(A[i]) + 1e-9)))
        cj = float(mid @ A[j] / (nm * (np.linalg.norm(A[j]) + 1e-9)))
        echo.append(abs(ci - cj))
    print(f"CONSENSUS {name}: wiggle p50={np.median(wig):.2f} p90={np.quantile(wig,0.9):.2f} | cons-cos p50={np.median(cons):+.2f} | echo-asym p50={np.median(echo):.2f}", flush=True)
print("CONSENSUS-DONE")
