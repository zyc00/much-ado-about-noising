"""THE CAUSAL FORCE PROBE (scripted): per-pair merge pressure of the actual training
gradient. For fusion-prone pairs (settle-vs-transit, label distance in the sigma band):
  mp = -(g_i - g_j) . (e_i - e_j) / ||e_i - e_j||   (per descent step; >0 = separation
INCREASES = separating force; <0 = merging force). [sign verified: de = -eta*g]
Views measured separately: L2 (=MSE view / MIP view-1) and MIP view-2 (sigma=0.1).
Across training snapshots of orig_msesnap / orig_mipsnap."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = os.environ["DSP"]
def load():
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
cfg, ds, ag = load()
dev = cfg.optimization.device
N = 4000
idx = np.random.RandomState(0).choice(len(ds), N, replace=False)
obs_w, acts = [], []
for i in idx:
    b = ds[int(i)]
    o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
    obs_w.append(o[:2].numpy()); acts.append(b["action"][:16].numpy())
OW = np.stack(obs_w); AC = np.stack(acts)
# settle = small-action chunks; transit = large
amag = np.linalg.norm(AC.reshape(N, -1), axis=1)
q_lo, q_hi = np.quantile(amag, 0.25), np.quantile(amag, 0.5)
settle = np.where(amag < q_lo)[0]; transit = np.where(amag > q_hi)[0]
import math
D_ = AC.reshape(N, -1).shape[1]
lo_d, hi_d = 0.5 * 0.1 * math.sqrt(D_), 2.0 * 0.1 * math.sqrt(D_)
FA = torch.tensor(AC.reshape(N, -1))
rs = np.random.RandomState(1)
cross, ctrl = [], []
tries = 0
while (len(cross) < 256 or len(ctrl) < 256) and tries < 2000000:
    tries += 1
    i = settle[rs.randint(len(settle))]; j = transit[rs.randint(len(transit))]
    la = float(torch.norm(FA[i] - FA[j]))
    if lo_d < la < hi_d and len(cross) < 256: cross.append((i, j))
    i2, j2 = transit[rs.randint(len(transit))], transit[rs.randint(len(transit))]
    if i2 != j2:
        la2 = float(torch.norm(FA[i2] - FA[j2]))
        if lo_d < la2 < hi_d and len(ctrl) < 256: ctrl.append((i2, j2))
print(f"pairs: cross(settle-transit)={len(cross)} ctrl(transit-transit)={len(ctrl)}", flush=True)
no = ds.normalizer["obs"]["state"]
def merge_pressure(view, pairs):
    mps, aligns, shares, gns = [], [], [], []
    for b0 in range(0, len(pairs), 64):
        chunk = pairs[b0:b0 + 64]
        ii = [p[0] for p in chunk]; jj = [p[1] for p in chunk]
        wi = torch.tensor(OW[ii], device=dev, dtype=torch.float32)
        wj = torch.tensor(OW[jj], device=dev, dtype=torch.float32)
        ai = torch.tensor(AC[ii], device=dev); aj = torch.tensor(AC[jj], device=dev)
        ei = ag.encoder(wi, None).detach().requires_grad_(True)
        ej = ag.encoder(wj, None).detach().requires_grad_(True)
        def L(e, a):
            B = len(a)
            if view == "v2":
                tt = torch.full((B,), 0.9, device=dev)
                tot = 0
                for r in range(4):
                    g = torch.Generator(device="cpu").manual_seed(500 + r)
                    eps = torch.randn(a.shape, generator=g).to(dev)
                    pred = ag.flow_map.get_velocity(tt, a + 0.1 * eps, e)
                    tot = tot + (((pred - a) / 0.1) ** 2).sum()
                return tot / 4
            t0 = torch.zeros(B, device=dev)
            pred = ag.flow_map.get_velocity(t0, torch.zeros_like(a), e)
            return ((pred - a) ** 2).sum()
        loss = L(ei, ai) + L(ej, aj)
        gi, gj = torch.autograd.grad(loss, [ei, ej])
        d = (ei - ej).detach().reshape(len(ai), -1)
        dn = d / (d.norm(dim=1, keepdim=True) + 1e-9)
        gd = (gi - gj).reshape(len(ai), -1)
        mp = (-gd * dn).sum(dim=1)
        gtot = gi.reshape(len(ai), -1).norm(dim=1) + gj.reshape(len(ai), -1).norm(dim=1)
        mps.extend(mp.detach().cpu().numpy().tolist())
        aligns.extend((mp / (gd.norm(dim=1) + 1e-12)).detach().cpu().numpy().tolist())
        shares.extend((mp.abs() / (gtot + 1e-12)).detach().cpu().numpy().tolist())
        gns.extend(gtot.detach().cpu().numpy().tolist())
    return np.median(mps), np.median(aligns), np.median(shares), np.median(gns)
import glob
for tag, run in [("MSE", "orig_msesnap"), ("MIP", "orig_mipsnap")]:
    snaps = sorted(glob.glob(f"logs/{run}/models/snap_*.pt"),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    picks = [snaps[0], snaps[len(snaps)//4], snaps[len(snaps)//2], snaps[-1]] if len(snaps) >= 4 else snaps
    for ck in picks:
        st = ck.split("_")[-1].split(".")[0]
        ag.load(ck, load_optimizer=False); ag.eval()
        for view in (["v1"] if tag == "MSE" else ["v1", "v2"]):
            mc, ac, sc, gc_ = merge_pressure(view, cross)
            mk, ak, sk, gk_ = merge_pressure(view, ctrl)
            print(f"MERGEFORCE {tag}@{st} {view}: S_cross={mc:+.4f} S_ctrl={mk:+.4f} | align_cross={ac:+.2f} align_ctrl={ak:+.2f} | share_cross={100*sc:.1f}% share_ctrl={100*sk:.1f}% | |g|_pair={gc_:.4f}", flush=True)
print("MERGEFORCE-DONE")
