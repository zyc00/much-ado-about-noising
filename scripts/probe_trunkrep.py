"""Trunk-level representation probe: fold composition + faithfulness computed on the
PENULTIMATE UNet features (input to final_conv) -- the representation headmse proves
load-bearing -- at the deployment operating point (zero action input, t=0).
Models: MSE, MIPs1, fadehint."""
import os, sys, glob
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        _ov = ["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"]
        _ov += [o for o in os.environ.get("EXTRA_OV", "").split(",") if o]
        cfg = compose(config_name="main", overrides=_ov)
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device

h5f = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h5f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase = [], []
for k in keys[:150]:
    o = h5f[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h5f[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        r = t - c1
        ph = 0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3))
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(ph)
h5f.close()
W = np.stack(W); phase = np.array(phase)
sdv = W[:, 1].std(0) + 1e-6

FAILS = {21003,21006,21008,21012,21018,21020,21022,21023,21032,21034,21036,21045,21047,21048,21052,21054,21062,21063}
qF = []
for pat in ["/mnt/pfs/yuchen/embq/msebulk/ep_*.npz", "/mnt/pfs/yuchen/embq/cfdump_mse2/ep_*.npz"]:
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
frng = np.random.RandomState(2)
FSUB = frng.choice(len(W), 2000, replace=False)
DOBS = np.stack([np.linalg.norm((W[FSUB, 1] - q[1][None]) / sdv, axis=1) for q in qF])

def rank(x): return np.argsort(np.argsort(x)).astype(np.float64)
def spearman(x, y):
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-12))

import os as _os
_MODELS = [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
           ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt"),
           ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt")]
if _os.environ.get("ONE_TAG"):
    _MODELS = [(_os.environ["ONE_TAG"], _os.environ["ONE_LOSS"], _os.environ["ONE_CKPT"])]
for name, loss, ck in _MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    feats = {}
    net = fm.network if hasattr(fm, "network") else fm
    # locate the module with final_conv
    mod = None
    for m in [net] + [getattr(net, a) for a in dir(net) if not a.startswith("_") and isinstance(getattr(net, a, None), torch.nn.Module)]:
        if hasattr(m, "final_conv"): mod = m; break
    assert mod is not None, "final_conv holder not found"
    buf = {}
    h = mod.final_conv.register_forward_hook(lambda m, i, o: buf.__setitem__("x", i[0].detach()))
    def trunk_feats(X):
        outs = []
        for i in range(0, len(X), 256):
            x = torch.tensor(no.normalize(X[i:i+256]), device=dev, dtype=torch.float32)
            e = enc({"state": x}, None)
            z = torch.zeros(len(x), 16, int(os.environ.get("ACT_DIM", "10")), device=dev)
            t0 = torch.zeros(len(x), device=dev)
            with torch.no_grad():
                fm.get_velocity(t0, z, e)
            outs.append(buf["x"].reshape(len(x), -1).cpu())
        return torch.nn.functional.normalize(torch.cat(outs), dim=1)
    EB = trunk_feats(W); EQ = trunk_feats(qF)
    h.remove()
    nb = (EQ @ EB.T).topk(10, dim=1).indices.numpy()
    ph = phase[nb]
    settle = (ph == 1).mean(); transit = (ph == 2).mean()
    EBn, EQn = EB.numpy(), EQ.numpy()
    faith = np.mean([spearman(1.0 - EBn[FSUB] @ EQn[i], DOBS[i]) for i in range(len(qF))])
    print(f"TRUNKREP {name}: dim={EBn.shape[1]} settle={settle:.0%} transit={transit:.0%} faith={faith:+.3f}", flush=True)
print("TRUNKREP-DONE")
