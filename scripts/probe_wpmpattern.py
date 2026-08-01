"""Does the representation pattern replicate on the wpmatch (5.05mm waypoint-noise) dataset?
Battery per (model, dataset): settle PCA PR/erank/top1; encoder-J kappa10/PR(s2)/k90 at 30
settle states; COLCOS top-10 pairwise |cos| and vs-u1; fold-analog = 10-NN phase composition
of kinematic-offset settle queries (10-40mm eef offsets with rel-dim consistency, off-support).
Original-data models included as same-instrument reference rows.
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
EEF_POS = [44, 45, 46]; EEF_QUAT = [47, 48, 49, 50]
REL = {"b": [0, 1, 2], "f": [14, 15, 16], "t": [28, 29, 30]}
def quat2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def load(loss, dsp):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dsp), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

BANKS = {}
def bank_of(dsp):
    if dsp in BANKS: return BANKS[dsp]
    h = h5py.File(dsp, "r")
    keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    W, phase = [], []
    for k in keys[:150]:
        o = h[f"data/{k}/obs"]
        ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
        cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
        if not cl: continue
        c1 = cl[0]
        for t in range(1, len(g), 2):
            r = t - c1
            W.append(np.stack([ov[t-1], ov[t]])); phase.append(0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3)))
    h.close()
    BANKS[dsp] = (np.stack(W), np.array(phase))
    return BANKS[dsp]

def specsum(ev):
    ev = np.clip(ev, 0, None); ev = ev[ev > 1e-12 * max(ev.max(), 1e-30)]
    p = ev / ev.sum()
    return float(1.0 / (p**2).sum()), float(np.exp(-(p*np.log(p)).sum())), float(np.sort(p)[-1])

DS_ORIG = "data/tool_hang_full2ins_2000.hdf5"
DS_WPM = "data/tool_hang_full2ins_wpmatch_2000.hdf5"
MODELS = [
    ("wpm_MSE", "regression", "logs/wpmatch_mse/models/model_latest.pt", DS_WPM),
    ("wpm_MIP", "mip", "logs/wpmatch_mip/models/model_latest.pt", DS_WPM),
    ("wpm_lam3", "mip_lambda", "logs/wpm_lam3/models/model_latest.pt", DS_WPM),
    ("orig_MSE", "regression", "logs/full_regression_2000/models/model_latest.pt", DS_ORIG),
    ("orig_MIP", "mip", "logs/full_mip_2000_s1/models/model_latest.pt", DS_ORIG),
]
rng = np.random.RandomState(0)
for name, loss, ck, dsp in MODELS:
    if not os.path.exists(ck):
        print(f"WPMPAT {name}: MISSING {ck}", flush=True); continue
    cfg, ds, ag = load(loss, dsp)
    no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema
    W, phase = bank_of(dsp)
    outs = []
    for i in range(0, len(W), 512):
        x = torch.tensor(no.normalize(W[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = enc({"state": x}, None)
        outs.append(e.reshape(len(x), -1))
    E = torch.cat(outs)
    En = (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy().astype(np.float64)
    sidx = np.where(phase == 1)[0]
    Xs = En[sidx] - En[sidx].mean(0, keepdims=True)
    pr, er, top1 = specsum(np.linalg.eigvalsh((Xs.T @ Xs) / len(Xs)))
    # settle canon states for J/COLCOS + offset fold queries
    canon = W[rng.choice(sidx, 30, replace=False)]
    def fea(inp):
        return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
    k10s, prj, k90s, pcs, u1cs = [], [], [], [], []
    for w in canon:
        x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
        J = torch.autograd.functional.jacobian(fea, x, vectorize=True).squeeze(1).detach().cpu().numpy()
        sv = np.linalg.svd(J, compute_uv=False); e2 = sv ** 2
        cs = np.cumsum(e2) / e2.sum()
        k90s.append(int(np.searchsorted(cs, 0.90) + 1))
        prj.append(float(e2.sum()**2 / (e2**2).sum()))
        k10s.append(float(sv[0] / sv[min(9, len(sv)-1)]))
        g = np.linalg.norm(J, axis=0); nz = np.where(g > 1e-8 * g.max())[0]
        top = nz[np.argsort(g[nz])[-10:]]
        C = J[:, top] / (np.linalg.norm(J[:, top], axis=0, keepdims=True) + 1e-12)
        S = np.abs(C.T @ C); iu = np.triu_indices(10, 1)
        pcs.append(float(S[iu].mean()))
        U = np.linalg.svd(J, full_matrices=False)[0]
        u1cs.append(float(np.abs(C.T @ U[:, 0]).mean()))
    # fold-analog: kinematic offsets 10-40mm from canon
    qs = []
    for w0 in canon:
        for rep in range(3):
            w = w0.copy()
            d = rng.randn(3); d /= np.linalg.norm(d); mag = rng.uniform(0.010, 0.040)
            for fr in range(2):
                R = quat2R(w[fr, EEF_QUAT])
                w[fr, EEF_POS] += d * mag
                for rel in REL.values(): w[fr, rel] += -R.T @ (d * mag)
            qs.append(w)
    qs = np.stack(qs)
    xq = torch.tensor(no.normalize(qs), device=dev, dtype=torch.float32)
    with torch.no_grad():
        eq = enc({"state": xq}, None).reshape(len(qs), -1)
    eq = (eq / (eq.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy()
    nb = np.argsort(eq @ En.T, axis=1)[:, -10:]
    ph = phase[nb]
    print(f"WPMPAT {name}: settlePCA PR={pr:.2f} erank={er:.2f} top1={top1:.0%} | JAC k90={np.median(k90s):.0f} PR={np.median(prj):.2f} k10={np.median(k10s):.1f} | COLCOS pair={np.median(pcs):.2f} vs-u1={np.median(u1cs):.2f} | offsetfold settle={(ph==1).mean():.0%} transit={(ph==2).mean():.0%}", flush=True)
print("WPMPAT-DONE")
