"""Resolution-band test: within-embedding-neighborhood label dispersion at settle states.

Theory: denoising at scale sigma requires the encoder to resolve state pairs whose labels
differ at scale <= sigma (coarser differences are absorbed by the anchor). Prediction:
the label dispersion inside embedding 10-NN neighborhoods at settle grows monotonically
with sigma, saturating at the MSE value for sigma >= 1; fadehint comparable to small sigma.

Dispersion = RMS distance (training action space, 10-dim normalized) between each neighbor's
label and the query's label; median/p90 over 500 settle queries. Bank NN uses the same
cosine-normalized embedding convention as the fold probes.
"""
import os, sys
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
rc = ds.rotation_transformer.forward(LBL[:, 3:6])
L10 = na.normalize(np.concatenate([LBL[:, :3], rc, LBL[:, 6:7]], axis=1).astype(np.float32))
sidx = np.where(phase == 1)[0]
rng = np.random.RandomState(3)
Q = sidx[rng.choice(len(sidx), 500, replace=False)]
print(f"bank={len(W)} settleq={len(Q)}", flush=True)

MODELS = [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
          ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt"),
          ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt")]
for d in ["orig_sig001", "orig_sig003", "orig_sig03", "orig_sig10", "sig20", "sig50"]:
    p = f"logs/{d}/models/model_latest.pt"
    if os.path.exists(p): MODELS.append((d, "mip", p))

for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema
    outs = []
    for i in range(0, len(W), 512):
        x = torch.tensor(no.normalize(W[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = enc({"state": x}, None)
        outs.append(e.reshape(len(x), -1))
    E = torch.cat(outs)
    E = (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy()
    disp, dispnm = [], []
    for q in Q:
        simq = E @ E[q]
        simq[q] = -2
        nb = np.argsort(simq)[-10:]
        d = np.linalg.norm(L10[nb] - L10[q][None], axis=1)
        disp.append(np.sqrt((d ** 2).mean()))
        mu_nb = L10[nb].mean(0)
        dispnm.append(np.sqrt(((L10[nb] - mu_nb) ** 2).sum(1).mean()))
    disp = np.array(disp); dispnm = np.array(dispnm)
    print(f"LABELDISP {name}: toquery p50={np.median(disp):.4f} p90={np.percentile(disp,90):.4f} | withinNN p50={np.median(dispnm):.4f} p90={np.percentile(dispnm,90):.4f}", flush=True)
print("LABELDISP-DONE")
