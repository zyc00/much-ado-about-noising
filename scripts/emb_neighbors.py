"""For deployed annulus states (failure closure windows + guilty chunks), find kNN in each
model's embedding space among dataset states; report neighbor phase & label statistics."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch, glob
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
def load_agent(loss, ck):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ck, load_optimizer=False); ag.eval()
    return cfg, ds, ag
MSE_CKPT = os.environ.get("MSE_CKPT", "logs/full_regression_2000/models/model_latest.pt")
MIP_CKPT = os.environ.get("MIP_CKPT", "logs/full_mip_2000/models/model_latest.pt")
AGS = {"fullMSE": load_agent("regression", MSE_CKPT),
       "fullMIP": load_agent("mip", MIP_CKPT)}
dev = "cuda"
# ---- dataset bank with phase + labels
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
    for t in range(2, len(ov) - 1, 2):
        p = 0 if t < c1 - 12 else (1 if t < c1 + 4 else (2 if t < c1 + 70 else 3))
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(p); LBL.append(a[t])
h.close()
W = np.stack(W); phase = np.array(phase); LBL = np.stack(LBL)
# ---- queries from deployed dumps
FAILS = {21003,21006,21008,21012,21018,21020,21022,21023,21032,21034,21036,21045,21047,21048,21052,21054,21062,21063}
qF, qS = [], []
for pat in [os.environ.get("QDIR1", "/tmp/msebulk") + "/ep_*.npz", os.environ.get("QDIR2", "/tmp/cfdump_mse2") + "/ep_*.npz"]:
    for f in sorted(glob.glob(pat)):
        z = np.load(f)
        if "obs" not in z.files: continue
        sd = int(z["seed"]); ki, ch, obs = z["ki"], z["chA"], z["obs"]
        g = ch[:, :, 6]; cl = None
        for i in range(1, len(ki)):
            if g[i-1].max() < 0 and g[i].max() >= 0: cl = i; break
        if cl is None: continue
        tgt = qF if (sd in FAILS and int(z["asm"]) == 0) else qS
        for i in range(max(0, cl-1), min(cl+5, len(obs))):
            tgt.append(obs[i])
qF = np.stack(qF); qS = np.stack(qS)
print(f"queries: FAIL closure-window chunks={len(qF)}, SUCCESS={len(qS)}; bank={len(W)}")
def emb_of(name, X):
    cfg, ds, ag = AGS[name]
    no = ds.normalizer["obs"]["state"]
    outs = []
    for i in range(0, len(X), 512):
        x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = ag.encoder_ema({"state": x}, None)
        outs.append(e.reshape(len(x), -1))
    E = torch.cat(outs)
    return E / (E.norm(dim=1, keepdim=True) + 1e-9)
for name in AGS:
    EB = emb_of(name, W)
    for qname, Q in [("FAIL", qF), ("SUCC", qS)]:
        EQ = emb_of(name, Q)
        sims = EQ @ EB.T
        top = sims.topk(10, dim=1)
        nb = top.indices.cpu().numpy(); dist = (1 - top.values).cpu().numpy()
        ph = phase[nb]                                  # (n,10)
        comp = [ (ph == p).mean() for p in range(4) ]
        lg = LBL[nb][:, :, 6]                            # neighbor label gripper
        open_frac = (lg < 0).mean()
        lxy = np.linalg.norm(LBL[nb][:, :, :2], axis=2).mean()
        print(f"{name} {qname}: NN-phase appr/settle/lift/ins = {comp[0]:.0%}/{comp[1]:.0%}/{comp[2]:.0%}/{comp[3]:.0%} | nb-label g<0 frac={open_frac:.0%} | nb-label |a_xy| mean={lxy:.3f} | emb-dist p50={np.median(dist[:,0]):.4f}")
print("DONE")
