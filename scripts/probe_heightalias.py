"""Height-aliasing battery. (1) cross-phase pair d_phi binned by |dh| (h = eef world z);
(2) embedding-NN height match at the 108 failure queries; (3) 1-D height-lookup surrogate:
compare the policy's action at failure states to the mean action of height-matched bank
states (any phase) vs settle-phase states. Models: MSE, fadehint."""
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
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; dev = cfg.optimization.device
start = cfg.task.obs_steps - 1; 

h5 = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase, LBL, H = [], [], [], []
for k in keys[:150]:
    o = h5[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h5[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        r = t - c1
        ph = 0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3))
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(ph); LBL.append(a[t]); H.append(ov[t, 46])
h5.close()
W = np.stack(W); phase = np.array(phase); LBL = np.stack(LBL); H = np.array(H)

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
qH = qF[:, 1, 46]

rng = np.random.RandomState(5)
ii = rng.randint(0, len(W), 80000); jj = rng.randint(0, len(W), 80000)
m = (ii != jj) & (phase[ii] != phase[jj])
ii, jj = ii[m], jj[m]
dh = np.abs(H[ii] - H[jj])
BINS = [(0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 1.0)]

for name, loss, ck in [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
                        ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt")]:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    def emb_of(X):
        outs = []
        for i in range(0, len(X), 512):
            x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                e = ag.encoder_ema({"state": x}, None)
            outs.append(e.reshape(len(x), -1))
        E = torch.cat(outs)
        return (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy()
    EB = emb_of(W); EQ = emb_of(qF)
    dphi = 1.0 - (EB[ii] * EB[jj]).sum(1)
    same = rng.randint(0, len(W), 40000); same2 = rng.randint(0, len(W), 40000)
    ms = (same != same2) & (phase[same] == phase[same2])
    dsame = 1.0 - (EB[same[ms]] * EB[same2[ms]]).sum(1)
    line = f"HALIAS {name} crossphase d_phi by |dh|:"
    for lo, hi in BINS:
        mb = (dh >= lo) & (dh < hi)
        line += f" [{lo*1000:.0f},{hi*1000:.0f})mm={np.median(dphi[mb]):.3f}"
    line += f" | samephase ref={np.median(dsame):.3f}"
    print(line, flush=True)
    # (2) NN height match at failure queries
    nb = (EQ @ EB.T).argsort(axis=1)[:, -10:]
    dh_nn = np.abs(H[nb] - qH[:, None])
    rand_nb = rng.randint(0, len(W), nb.shape)
    dh_rand = np.abs(H[rand_nb] - qH[:, None])
    print(f"HALIAS {name} failNN |dh|: p50={np.median(dh_nn)*1000:.1f}mm (random-bank ref {np.median(dh_rand)*1000:.1f}mm)", flush=True)
    # (3) height-lookup surrogate for the policy action at failure states
    ot = {"state": torch.tensor(no.normalize(qF), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((len(qF), 16, 10), device=dev), obs=ot, use_ema=True)
    A = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy()))[:, start]
    d_lookup, d_settle = [], []
    for i in range(len(qF)):
        hm = np.abs(H - qH[i]) < 0.005
        if hm.sum() < 5: continue
        a_lookup = LBL[hm].mean(0)
        a_settle = LBL[(phase == 1)].mean(0)
        d_lookup.append(np.linalg.norm(A[i] - a_lookup))
        d_settle.append(np.linalg.norm(A[i] - a_settle))
    print(f"HALIAS {name} action at failq: |a - heightlookup| p50={np.median(d_lookup):.3f}  |a - settlemean| p50={np.median(d_settle):.3f}", flush=True)
print("HALIAS-DONE")
