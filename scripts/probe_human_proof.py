"""Proof battery for the human-data hypotheses: hMSE vs hMIP (historical seed-5 pair).

SMOOTHNESS (is MIP's deployed function smoother / does MSE chase tremor?):
 S1 temporal TV along GT demo state sequences: mean ||pi(s_{t+1}) - pi(s_t)|| vs the
    label TV (labels include tremor; a noise-chasing fit has policy TV ~ label TV).
 S2 high-frequency label tracking: split labels a_t = smooth(a) + hf(a) (moving avg width 9);
    corr( hf(pi), hf(a) ) per window — direct 'fits the tremor' coefficient.
 S3 small-scale local Lipschitz: ||pi(s+ds) - pi(s)|| / ||ds|| for kinematic eef offsets
    of 1-5mm at align states (complement of the 10-40mm RESP numbers).

DIMENSIONALITY (is the feature-conditioning story load-bearing? does engaging more
dimensions help the task?):
 D1 trunk-feature PCA per window (all / align1 / align2): PR, erank, top-1 share.
 D2 TRUNCATION-FIT CURVES: ridge readout of the action label from top-k trunk-feature PCs,
    k in {1,2,4,8,16,32,64,128}, train/test split by demos; targets = (a) raw labels,
    (b) smoothed labels (clean-servo proxy). If hMIP's align fit keeps improving with k
    beyond hMSE's saturation, 'engaging more dimensions carries the fine servo' is measured.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
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

# ---- demo sequences with window labels
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
SEQS = []          # per demo: (W (T,2,53), A10 (T,10), wid (T,), demo_id)
for di, k in enumerate(keys[:120]):
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    c2 = next((t for t in cl if r1 and t > r1), None)
    rc = ds.rotation_transformer.forward(a[:, 3:6])
    a10 = na.normalize(np.concatenate([a[:, :3], rc, a[:, 6:7]], axis=1).astype(np.float32))
    wide = np.full(T, -1)
    for t in range(1, T - 1):
        if t < c1 - 12: wide[t] = 0
        elif c1 - 12 <= t < c1 + 5: wide[t] = 1
        elif r1 and c1 + 5 <= t < r1 - 30: wide[t] = 2
        elif r1 and r1 - 30 <= t < r1: wide[t] = 3
        elif r1 and c2 and r1 <= t < c2 - 12: wide[t] = 4
        elif c2 and c2 - 12 <= t < c2 + 5: wide[t] = 5
        elif c2 and c2 + 5 <= t < T - 40: wide[t] = 6
        elif c2 and T - 40 <= t: wide[t] = 7
    Wd = np.stack([np.stack([ov[t-1], ov[t]]) for t in range(1, T - 1)])
    SEQS.append((Wd, a10[1:T-1], wide[1:T-1], di))
h.close()
print(f"demos={len(SEQS)}", flush=True)

def smooth_ma(x, w=9):
    k = np.ones(w) / w
    return np.stack([np.convolve(x[:, d], k, mode="same") for d in range(x.shape[1])], 1)

WNAMES = ["approach1", "settle1", "carry1", "align1", "between", "settle2", "carry2", "align2"]
MODELS = [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt")]
rng = np.random.RandomState(0)
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    tts = cfg.optimization.t_two_step
    net = fm.network if hasattr(fm, "network") else fm
    mod = None
    for m in [net] + [getattr(net, a) for a in dir(net) if not a.startswith("_") and isinstance(getattr(net, a, None), torch.nn.Module)]:
        if hasattr(m, "final_conv"): mod = m; break
    buf = {}
    hk = mod.final_conv.register_forward_hook(lambda m, i, o: buf.__setitem__("x", i[0].detach()))
    def fwd(X, twostep=False, want_feats=False):
        preds, feats = [], []
        for i in range(0, len(X), 256):
            x = torch.tensor(no.normalize(X[i:i+256]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                e = enc({"state": x}, None)
                y = fm.get_velocity(torch.zeros(len(x), device=dev), torch.zeros(len(x), 16, 10, device=dev), e)
                if want_feats: feats.append(buf["x"].reshape(len(x), -1).cpu().numpy())
                if twostep:
                    y = fm.get_velocity(torch.full((len(x),), tts, device=dev), y, e)
            preds.append(y[:, START].cpu().numpy())
        return np.concatenate(preds), (np.concatenate(feats) if want_feats else None)

    # ---- S1/S2 over demo sequences (collect preds + feats in one pass, step1)
    tv_pol, tv_pol2, tv_lbl, hf_corr = {w: [] for w in range(8)}, {w: [] for w in range(8)}, {w: [] for w in range(8)}, {w: [] for w in range(8)}
    FE, LB, WD, DID = [], [], [], []
    for Wd, a10, wide, di in SEQS[:60]:
        p1, ft = fwd(Wd, twostep=False, want_feats=True)
        p2, _ = fwd(Wd, twostep=True) if loss == "mip" else (p1, None)
        FE.append(ft[::2]); LB.append(a10[::2]); WD.append(wide[::2]); DID.append(np.full(len(ft[::2]), di))
        sm = smooth_ma(a10); hf_l = a10 - sm
        sp1 = smooth_ma(p1); hf_p = p1 - sp1
        dp1 = np.linalg.norm(np.diff(p1, axis=0), axis=1)
        dp2 = np.linalg.norm(np.diff(p2, axis=0), axis=1)
        dl = np.linalg.norm(np.diff(a10, axis=0), axis=1)
        for w in range(8):
            m = wide[:-1] == w
            if m.sum() < 8: continue
            tv_pol[w].append(dp1[m].mean()); tv_pol2[w].append(dp2[m].mean()); tv_lbl[w].append(dl[m].mean())
            mm = wide == w
            num = (hf_p[mm] * hf_l[mm]).sum()
            den = np.sqrt((hf_p[mm]**2).sum() * (hf_l[mm]**2).sum()) + 1e-12
            hf_corr[w].append(num / den)
    line1 = f"SMOOTH {name} TV(policy)/TV(label) step1:"
    line2 = f"SMOOTH {name} hf-corr(policy,label):"
    for w in [1, 2, 3, 7]:
        line1 += f" {WNAMES[w]}={np.mean(tv_pol[w])/np.mean(tv_lbl[w]):.2f}"
        line2 += f" {WNAMES[w]}={np.mean(hf_corr[w]):.3f}"
    if loss == "mip":
        line1 += " | 2step: " + " ".join(f"{WNAMES[w]}={np.mean(tv_pol2[w])/np.mean(tv_lbl[w]):.2f}" for w in [1, 2, 3, 7])
    print(line1, flush=True); print(line2, flush=True)

    # ---- S3 small-scale Lipschitz at align1 states
    FEc = np.concatenate(FE); LBc = np.concatenate(LB); WDc = np.concatenate(WD); DIDc = np.concatenate(DID)
    al_idx = np.where(WDc == 3)[0]
    Wall = np.concatenate([s[0][::2] for s in SEQS[:60]])
    canon = Wall[al_idx[rng.choice(len(al_idx), 40, replace=False)]]
    base, pert, mags = [], [], []
    for w0 in canon:
        for rep in range(2):
            w = w0.copy()
            d = rng.randn(3); d /= np.linalg.norm(d); mag = rng.uniform(0.001, 0.005)
            for fr in range(2):
                R = quat2R(w[fr, EEF_QUAT])
                w[fr, EEF_POS] += d * mag
                for rel in REL.values(): w[fr, rel] += -R.T @ (d * mag)
            base.append(w0); pert.append(w); mags.append(mag)
    b_, p_ = np.stack(base), np.stack(pert)
    for twostep in ([False, True] if loss == "mip" else [False]):
        a0, _ = fwd(b_, twostep); a1, _ = fwd(p_, twostep)
        lip = np.linalg.norm((a1 - a0)[:, :3], axis=1) / np.array(mags)
        print(f"SMOOTH {name} {'2step' if twostep else 'step1'} align1 local-Lip (1-5mm): p50={np.median(lip):.2f} p90={np.quantile(lip,0.9):.2f} (norm-act units / m)", flush=True)

    # ---- D1 trunk PCA per window
    def specsum(ev):
        ev = np.clip(ev, 0, None); ev = ev[ev > 1e-12 * max(ev.max(), 1e-30)]
        p = ev / ev.sum()
        return float(1/(p**2).sum()), float(np.exp(-(p*np.log(p)).sum())), float(np.sort(p)[-1])
    Fn = FEc / (np.linalg.norm(FEc, axis=1, keepdims=True) + 1e-9)
    line = f"DIMS {name} trunk PCA:"
    for wn, sel in [("all", np.arange(len(Fn))), ("align1", np.where(WDc == 3)[0]), ("align2", np.where(WDc == 7)[0])]:
        X = Fn[sel] - Fn[sel].mean(0, keepdims=True)
        pr, er, t1 = specsum(np.linalg.eigvalsh((X.T @ X) / len(X)))
        line += f" | {wn}: PR={pr:.1f} erank={er:.1f} top1={t1:.0%}"
    print(line, flush=True)

    # ---- D2 truncation-fit curves (align1), demo-blocked split
    sel = np.where(WDc == 3)[0]
    tr = sel[DIDc[sel] % 5 != 0]; te = sel[DIDc[sel] % 5 == 0]
    Xtr = Fn[tr] - Fn[tr].mean(0, keepdims=True); Xte = Fn[te] - Fn[tr].mean(0, keepdims=True)
    C = (Xtr.T @ Xtr) / len(Xtr)
    ev, V = np.linalg.eigh(C); order = np.argsort(ev)[::-1]; V = V[:, order]
    for tgt_name, Y in [("raw", LBc), ("smooth", None)]:
        if tgt_name == "smooth":
            Y = np.concatenate([smooth_ma(l) for l in LB])
        Ytr, Yte = Y[tr], Y[te]
        line = f"DIMS {name} align1 truncfit ({tgt_name}) test-RMS:"
        for k in [1, 2, 4, 8, 16, 32, 64, 128]:
            Ptr = Xtr @ V[:, :k]; Pte = Xte @ V[:, :k]
            Wr = np.linalg.solve(Ptr.T @ Ptr + 1e-3 * np.eye(k), Ptr.T @ (Ytr - Ytr.mean(0)))
            pred = Pte @ Wr + Ytr.mean(0)
            line += f" k{k}={np.sqrt(((pred - Yte) ** 2).sum(1).mean()):.3f}"
        print(line, flush=True)
    hk.remove()
print("PROOF-DONE")
