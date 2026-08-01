"""Cross-phase interpolation verification: does the policy's emitted LATERAL action at
failure states equal the conditional mean of its own embedding neighborhood?
Per query: cos_xy(policy, embNN label mean), |policy_xy|/|embNN_xy|, vs baselines
(obs-space NN mean, settle-phase mean). Models: MSE, fadehint."""
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
start = cfg.task.obs_steps - 1

h5f = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h5f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase, LBL = [], [], []
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
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(ph); LBL.append(a[t])
h5f.close()
W = np.stack(W); phase = np.array(phase); LBL = np.stack(LBL)
sdv = W[:, 1].std(0) + 1e-6
settle_mean = LBL[phase == 1].mean(0)

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
        return (E / (E.norm(dim=1, keepdim=True) + 1e-9))
    EB = emb_of(W); EQ = emb_of(qF)
    nb = (EQ @ EB.T).topk(10, dim=1).indices.cpu().numpy()
    ot = {"state": torch.tensor(no.normalize(qF), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((len(qF), 16, 10), device=dev), obs=ot, use_ema=True)
    A = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy()))[:, start]
    # obs-space NN (z-scored, current frame)
    QZ = (qF[:, 1] / sdv); BZ = (W[:, 1] / sdv)
    obnb = np.argsort(((BZ[None] - QZ[:, None]) ** 2).sum(-1), axis=1)[:, :10]
    def stats(nbrs):
        cs, rt = [], []
        for i in range(len(qF)):
            mean_xy = LBL[nbrs[i], :2].mean(0)
            pxy = A[i, :2]
            nm, npq = np.linalg.norm(mean_xy), np.linalg.norm(pxy)
            if nm < 1e-4: continue
            cs.append(float(pxy @ mean_xy / (nm * npq + 1e-12)))
            rt.append(float(npq / nm))
        return np.median(cs), np.median(rt), len(cs)
    ce, re_, ne = stats(nb)
    co, ro, no_ = stats(obnb)
    # settle-mean baseline: cos with settle mean xy (tiny norm -> often skipped)
    smxy = settle_mean[:2]; sm = np.linalg.norm(smxy)
    cset = np.median([float(A[i,:2] @ smxy / (sm*np.linalg.norm(A[i,:2])+1e-12)) for i in range(len(qF))]) if sm > 1e-4 else float("nan")
    pmag = np.median(np.linalg.norm(A[:, :2], axis=1))
    embmag = np.median([np.linalg.norm(LBL[nb[i], :2].mean(0)) for i in range(len(qF))])
    comp = (phase[nb] == 2).mean()
    print(f"XPINT {name}: |policy_xy| p50={pmag:.4f} | embNN: cos_xy p50={ce:+.2f} amp_ratio={re_:.2f} (n={ne}, transit-frac={comp:.2f}, |mean_xy|={embmag:.4f}) | obsNN: cos={co:+.2f} ratio={ro:.2f} | cos(settle_mean)={cset:+.2f}", flush=True)
print("XPINT-DONE")
