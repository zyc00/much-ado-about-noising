"""Gradient-stream anatomy: per-sample gradient signal entering the encoder under
L2 / Cauchy / MIP-view-2, at IDENTICAL parameters and data.
Influence proxy: g_i = || d loss_i / d emb_i || (per-sample by construction).
Metrics:
  tail domination: share of sum(g) from top-1% / top-10% samples ranked by g
  concentration: kurtosis of g
  alignment with label-tail: median g of samples in top-decile |label - kNN-label-mean|
    vs bottom-half (does the gradient stream follow the tremor?)
  channel scale: per-channel |d loss / d pred|, rot(3:9) / pos(0:3) ratio.
Parameter points: (a) trained hMSE_s5 chiunet, (b) random init (seed 0)."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = os.environ.get("DSP", "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4")
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
NET = os.environ.get("NET", "chiunet")
def load():
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, f"network={NET}",
            "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    if NET == "chiunet": cfg.task.horizon = 16
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
cfg, ds, ag = load()
dev = cfg.optimization.device
N = 4096
idx = np.random.RandomState(0).choice(len(ds), N, replace=False)
obs = torch.stack([ds[i]["obs"]["state"] if isinstance(ds[i]["obs"], dict) else ds[i]["obs"] for i in idx[:8]])
# build batch properly
def get_batch(ids):
    xs, ys = [], []
    for i in ids:
        b = ds[int(i)]
        o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
        xs.append(o[:2] if o.shape[0] > 2 else o)  # obs window (To=2)
        ys.append(b["action"])
    return torch.stack(xs).to(dev), torch.stack(ys).to(dev)
# label-tail score: |label - kNN-mean over obs-space neighbors| using a subsample
Xs, Ys = get_batch(idx)
flatX = Xs.reshape(N, -1).cpu().numpy(); flatY = Ys.reshape(N, -1).cpu().numpy()
import torch as _t
FX = _t.tensor(flatX)
D = _t.cdist(FX, FX)
nbr = D.topk(9, largest=False).indices[:, 1:].numpy()
knnmean = flatY[nbr].mean(1)
tail_score = np.linalg.norm(flatY - knnmean, axis=1)
tail_hi = tail_score >= np.quantile(tail_score, 0.9)
tail_lo = tail_score <= np.quantile(tail_score, 0.5)

if os.environ.get("CKPT"):
    POINTS = [(os.environ.get("TAG", "pt"), os.environ["CKPT"])]
else:
    POINTS = [("hMSE-point", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
              ("hMIP-point", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt")]
for tag, ck in POINTS:
    ag.load(ck, load_optimizer=False)
    enc, fm = ag.encoder, ag.flow_map
    enc.eval(); fm.eval()
    res = {}
    for loss_name in os.environ.get("GLOSSES", "L2 Cauchy MIPv2").split():
        gs, ch, rs = [], [], []
        B = 256
        for b0 in range(0, N, B):
            xb, yb = get_batch(idx[b0:b0 + B])
            emb = enc(xb, None)
            emb_r = emb.detach().requires_grad_(True)
            t = torch.zeros(len(xb), device=dev)
            if loss_name == "HT":
                t = torch.zeros(len(xb), device=dev)
                pred, s_raw = fm.net(torch.zeros_like(yb), t, t, emb_r)
                r = pred - yb
                sb = torch.nn.functional.softplus(s_raw).reshape(len(xb), -1).mean(dim=1) + 1e-3
                Dn = r[0].numel()
                z2 = r.reshape(len(xb), -1).pow(2).sum(dim=1) / (sb ** 2 * Dn)
                li = 0.5 * (2.0 + 1.0) * torch.log1p(z2 / 2.0) * Dn + Dn * torch.log(sb)
            elif loss_name == "MIPv2":
                sig = 0.1
                eps = torch.randn_like(yb)
                a0 = yb + sig * eps
                tt = torch.full((len(xb),), 0.9, device=dev)
                pred = fm.get_velocity(tt, a0, emb_r)
            else:
                a0 = torch.zeros_like(yb)
                pred = fm.get_velocity(t, a0, emb_r)
            r = pred - yb
            if loss_name == "Cauchy":
                li = torch.log1p((r / 0.2) ** 2).sum(dim=(1, 2))
            elif loss_name != "HT":
                li = (r ** 2).sum(dim=(1, 2))
            g = torch.autograd.grad(li.sum(), emb_r)[0]
            gs.append(g.reshape(len(xb), -1).norm(dim=1).detach().cpu().numpy())
            rs.append(r.reshape(len(xb), -1).norm(dim=1).detach().cpu().numpy())
            # channel-scale: |dloss/dpred| per channel group
            dpred = 2 * r if loss_name != "Cauchy" else 2 * r / (0.04 + r ** 2) * 0.04 / 0.04
            if loss_name == "Cauchy":
                dpred = 2 * r / (1 + (r / 0.2) ** 2)
            ch.append(dpred.abs().mean(dim=(0, 1)).detach().cpu().numpy())
        g = np.concatenate(gs); c = np.mean(ch, 0); rr = np.concatenate(rs)
        srt = np.sort(g)[::-1]
        top1 = srt[:max(1, N // 100)].sum() / g.sum(); top10 = srt[:N // 10].sum() / g.sum()
        kurt = float(((g - g.mean()) ** 4).mean() / (g.var() ** 2))
        tail_ratio = np.median(g[tail_hi]) / np.median(g[tail_lo])
        rot_pos = c[3:9].mean() / c[0:3].mean()
        rres = np.median(rr[tail_hi]) / np.median(rr[tail_lo])
        print(f"GRAD {tag} {loss_name}: top1%={100*top1:.1f}% top10%={100*top10:.1f}% kurt={kurt:.1f}"
              f" | g(tail)/g(typ)={tail_ratio:.2f} | |resid| tail/typ={rres:.2f} med={np.median(rr):.3f}"
              f" | rot/pos grad ratio={rot_pos:.2f}", flush=True)
print("GRAD-DONE")
