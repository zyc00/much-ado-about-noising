"""How many embedding dimensions carry the model's decision structure?
Tests: is the neighborhood structure (the readout substrate) reproducible from ONE
embedding coordinate (single-column-dominance hypothesis for MSE) or does it require
several (multi-dimension hypothesis for MIP/fadehint)?

Per model:
 (a) SPECTRUM: PCA of bank embeddings (all-phase + settle-only): PR, k90, k99, top-5 shares.
 (b) TRUNC: 10-NN of the 108 failure queries computed in the top-k PC subspace,
     k in {1,2,4,8,16,32}: overlap with the model's OWN full-embedding 10-NN +
     phase composition (settle%/transit%).
 (c) RESID: same with PC1 REMOVED (PCs 2..end). If the model is 1-D, this destroys
     the neighborhoods; if multi-D, they survive.
 (d) DECODE: ridge decodability from PC1-only vs PC1-removed vs full embedding:
     R2 for eef-z / frame-rel xy / gripper + settle-vs-transit linear accuracy.
 (e) JAC: encoder-Jacobian spectrum at settle canon: k90 (90% of spectral energy),
     PR(s^2), kappa10  -- the advisor's condition-number quantity.
Sigma-ladder models included for the dose-response of (a)/(e).
"""
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
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device

# ---- bank with phase labels (same conventions as probe_foldsweep)
h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
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
        ph = 0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3))
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(ph)
h.close()
W = np.stack(W); phase = np.array(phase)

# ---- 108 failure closure-window queries (same as probe_foldsweep)
FAILS = {21003,21006,21008,21012,21018,21020,21022,21023,21032,21034,21036,21045,21047,21048,21052,21054,21062,21063}
qF = []
for pat in [os.environ.get("QDIR1", "/mnt/pfs/yuchen/embq/msebulk") + "/ep_*.npz",
            os.environ.get("QDIR2", "/mnt/pfs/yuchen/embq/cfdump_mse2") + "/ep_*.npz"]:
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
Q = np.load("scripts/jac_queries.npz")
print(f"bank={len(W)} foldq={len(qF)}", flush=True)

def spec_stats(ev):
    ev = np.sort(np.clip(ev, 0, None))[::-1]
    tot = ev.sum() + 1e-12
    pr = float(tot**2 / ((ev**2).sum() + 1e-12))
    cs = np.cumsum(ev) / tot
    k90 = int(np.searchsorted(cs, 0.90) + 1); k99 = int(np.searchsorted(cs, 0.99) + 1)
    return pr, k90, k99, (ev[:5] / tot)

def ph_comp(nb):
    p = phase[nb]
    return (p == 1).mean(), (p == 2).mean()

def knn(BP, QP, k=10):
    d = -2 * QP @ BP.T + (BP**2).sum(1)[None, :] + (QP**2).sum(1)[:, None]
    return np.argsort(d, axis=1)[:, :k]

MODELS = [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
          ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt"),
          ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt")]
for d in ["orig_sig001", "orig_sig003", "orig_sig03", "orig_sig10", "sig20", "sig50"]:
    for base in ["model_latest.pt", "model_best.pt"]:
        p = f"logs/{d}/models/{base}"
        if os.path.exists(p):
            MODELS.append((d, "mip", p)); break

FD_Z = 46; FD_XY = [14, 15]; FD_G = [51, 52]
for name, loss, ck in MODELS:
    if not os.path.exists(ck):
        print(f"DIMCOUNT {name}: MISSING {ck}", flush=True); continue
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema

    def emb_of(X):
        outs = []
        for i in range(0, len(X), 512):
            x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                e = enc({"state": x}, None)
            outs.append(e.reshape(len(x), -1))
        E = torch.cat(outs)
        return (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy()

    EB = emb_of(W); EQ = emb_of(qF)
    mu = EB.mean(0, keepdims=True)
    Xc = EB - mu; Qc = EQ - mu
    # (a) spectrum
    C = (Xc.T @ Xc) / len(Xc)
    ev, V = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]; ev = ev[order]; V = V[:, order]
    pr, k90, k99, top5 = spec_stats(ev)
    sidx = np.where(phase == 1)[0]
    Xs = EB[sidx] - EB[sidx].mean(0, keepdims=True)
    evs = np.linalg.eigvalsh((Xs.T @ Xs) / len(Xs))
    prs, k90s, k99s, top5s = spec_stats(evs)
    print(f"DIMCOUNT {name} SPECTRUM all: PR={pr:.2f} k90={k90} k99={k99} top5={np.round(top5,3).tolist()}", flush=True)
    print(f"DIMCOUNT {name} SPECTRUM settle: PR={prs:.2f} k90={k90s} k99={k99s} top5={np.round(top5s,3).tolist()}", flush=True)
    # full-embedding NN baseline
    nb_full = knn(Xc, Qc)
    s_f, t_f = ph_comp(nb_full)
    print(f"DIMCOUNT {name} FULLFOLD: settle={s_f:.0%} transit={t_f:.0%}", flush=True)
    # (b) truncation sweep
    for k in [1, 2, 4, 8, 16, 32]:
        BP = Xc @ V[:, :k]; QP = Qc @ V[:, :k]
        nb = knn(BP, QP)
        ov = np.mean([len(set(nb[i]) & set(nb_full[i])) / 10.0 for i in range(len(nb))])
        s, t = ph_comp(nb)
        print(f"DIMCOUNT {name} TRUNC k={k}: overlap={ov:.2f} settle={s:.0%} transit={t:.0%}", flush=True)
    # (c) residual: PC1 removed
    BP = Xc - (Xc @ V[:, :1]) @ V[:, :1].T; QP = Qc - (Qc @ V[:, :1]) @ V[:, :1].T
    nb = knn(BP, QP)
    ov = np.mean([len(set(nb[i]) & set(nb_full[i])) / 10.0 for i in range(len(nb))])
    s, t = ph_comp(nb)
    print(f"DIMCOUNT {name} RESID pc1-removed: overlap={ov:.2f} settle={s:.0%} transit={t:.0%}", flush=True)
    # (d) decodability: PC1-only vs residual vs full
    Yd = W[:, 1]
    reps = {"pc1": Xc @ V[:, :1], "resid": Xc - (Xc @ V[:, :1]) @ V[:, :1].T, "full": Xc}
    line = f"DIMCOUNT {name} DECODE"
    for rn, R in reps.items():
        A, Yt = R[:10000], Yd[:10000]
        mu_y = Yt.mean(0)
        Wr = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ (Yt - mu_y))
        pred = R[10000:] @ Wr + mu_y
        Yv = Yd[10000:]
        ss = 1 - ((Yv - pred) ** 2).sum(0) / (((Yv - Yv.mean(0)) ** 2).sum(0) + 1e-12)
        # settle-vs-transit linear accuracy
        m_tr = (phase[:10000] == 1) | (phase[:10000] == 2)
        m_te = (phase[10000:] == 1) | (phase[10000:] == 2)
        ycls = np.where(phase == 1, 1.0, -1.0)
        Ac = R[:10000][m_tr]; yc = ycls[:10000][m_tr]
        Wc = np.linalg.solve(Ac.T @ Ac + 1e-3 * np.eye(Ac.shape[1]), Ac.T @ yc)
        acc = float((np.sign(R[10000:][m_te] @ Wc) == ycls[10000:][m_te]).mean())
        line += f" | {rn}: z={ss[FD_Z]:+.2f} xy={np.mean(ss[FD_XY]):+.2f} grip={np.mean(ss[FD_G]):+.2f} phacc={acc:.2f}"
    base = float((phase[10000:][(phase[10000:]==1)|(phase[10000:]==2)] == 1).mean())
    print(line + f" | base={max(base,1-base):.2f}", flush=True)
    # (e) encoder-Jacobian spectrum at settle canon
    def fea(inp):
        return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
    k90j, prj, k10j = [], [], []
    for w in Q["canon"]:
        x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
        J = torch.autograd.functional.jacobian(fea, x, vectorize=True).squeeze(1).detach().cpu().numpy()
        sv = np.linalg.svd(J, compute_uv=False); e2 = sv ** 2
        cs = np.cumsum(e2) / e2.sum()
        k90j.append(int(np.searchsorted(cs, 0.90) + 1))
        prj.append(float(e2.sum() ** 2 / ((e2 ** 2).sum() + 1e-12)))
        k10j.append(float(sv[0] / sv[min(9, len(sv)-1)]))
    print(f"DIMCOUNT {name} JAC settle_canon: k90={np.median(k90j):.0f} PR(s2)={np.median(prj):.2f} k10={np.median(k10j):.1f}", flush=True)
print("DIMCOUNT-DONE")
