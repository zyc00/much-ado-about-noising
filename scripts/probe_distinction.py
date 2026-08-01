"""Per-distinction incentive table (the scripted analog of the gradient-stream table).
Fusion-prone pairs: label-distance bottom 20% AND state-distance top 50% (cross-phase,
delta ~ 0). Controls: both distances bottom 20% (genuinely similar states).
For each loss: collapse ratio = [L(midpoint emb, i) + L(midpoint emb, j)] /
[L(own emb, i) + L(own emb, j)]. ~1.0 = blind to the distinction; >>1 = full-scale
incentive to keep the states separate."""
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
OW = torch.tensor(np.stack(obs_w)); AC = torch.tensor(np.stack(acts))
flatS = OW.reshape(N, -1); flatA = AC.reshape(N, -1)
DS_ = torch.cdist(flatS, flatS); DA = torch.cdist(flatA, flatA)
# delta ~ sigma band: per-dim RMS label distance in [0.5, 2]x sigma (sigma=0.1)
import math
D_ = flatA.shape[1]
lo_d, hi_d = 0.5 * 0.1 * math.sqrt(D_), 2.0 * 0.1 * math.sqrt(D_)
qShi = torch.quantile(DS_.flatten()[:200000], 0.5)
qSlo = torch.quantile(DS_[DS_ > 0].flatten()[:200000], 0.2)
cross, ctrl = [], []
rs = np.random.RandomState(1)
tries = 0
while (len(cross) < 300 or len(ctrl) < 300) and tries < 3000000:
    tries += 1
    i, j = rs.randint(N), rs.randint(N)
    if i == j: continue
    la, st = DA[i, j], DS_[i, j]
    if lo_d < la < hi_d and st > qShi and len(cross) < 300: cross.append((i, j))
    elif lo_d < la < hi_d and 0 < st < qSlo and len(ctrl) < 300: ctrl.append((i, j))
print(f"pairs: cross={len(cross)} ctrl={len(ctrl)}", flush=True)
def per_loss(loss_name, emb, a):
    B = len(a)
    if loss_name == "MIPv2":
        tt = torch.full((B,), 0.9, device=dev)
        tot = 0
        for r in range(4):
            g = torch.Generator(device="cpu").manual_seed(1000 + r)
            eps = torch.randn(a.shape, generator=g).to(dev)
            pred = ag.flow_map.get_velocity(tt, a + 0.1 * eps, emb)
            tot = tot + ((pred - a) ** 2).sum(dim=(1, 2))
        return tot / 4
    t = torch.zeros(B, device=dev)
    pred = ag.flow_map.get_velocity(t, torch.zeros_like(a), emb)
    r = pred - a
    if loss_name == "Cauchy":
        return torch.log1p((r / 0.2) ** 2).sum(dim=(1, 2))
    return (r ** 2).sum(dim=(1, 2))
CKPTS = [("L2", "@init", None), ("MIPv2", "@init", None),
         ("L2", "@MSEpt", os.environ["CK_MSE"]), ("Cauchy", "@MSEpt", os.environ["CK_MSE"]),
         ("MIPv2", "@MIPpt", os.environ["CK_MIP"])]
import copy
init_state = copy.deepcopy(ag.flow_map.state_dict()), copy.deepcopy(ag.encoder.state_dict())
for loss_name, ptag, ck in CKPTS:
    if ck is None:
        ag.flow_map.load_state_dict(init_state[0]); ag.encoder.load_state_dict(init_state[1])
    else:
        ag.load(ck, load_optimizer=False)
    ag.eval()
    no = ds.normalizer["obs"]["state"]
    out = {}
    for tag, pairs in [("CROSS", cross), ("CTRL", ctrl)]:
        ratios, grads = [], []
        for b0 in range(0, len(pairs), 64):
            chunk = pairs[b0:b0 + 64]
            ii = [p[0] for p in chunk]; jj = [p[1] for p in chunk]
            wi = torch.tensor(no.normalize(OW[ii].numpy()), device=dev, dtype=torch.float32)
            wj = torch.tensor(no.normalize(OW[jj].numpy()), device=dev, dtype=torch.float32)
            ai = AC[ii].to(dev); aj = AC[jj].to(dev)
            ei = ag.encoder(wi, None).detach(); ej = ag.encoder(wj, None).detach()
            m = ((ei + ej) / 2).requires_grad_(True)
            base = per_loss(loss_name, ei, ai) + per_loss(loss_name, ej, aj)
            merged = per_loss(loss_name, m, ai) + per_loss(loss_name, m, aj)
            # separation gradient: gradient of merged loss wrt the shared embedding —
            # its per-pair norm is the pressure to pull the two states apart
            g = torch.autograd.grad(merged.sum(), m)[0]
            gn = g.reshape(len(ai), -1).norm(dim=1)
            ratios.extend((merged.detach() / (base.detach() + 1e-9)).cpu().numpy().tolist())
            grads.extend(gn.detach().cpu().numpy().tolist())
        out[tag] = (np.median(ratios), np.median(grads))
    print(f"DISTINCT {loss_name}{ptag}: merge-cost ratio CROSS={out['CROSS'][0]:.2f} CTRL={out['CTRL'][0]:.2f} | sep-gradient CROSS={out['CROSS'][1]:.3f} CTRL={out['CTRL'][1]:.3f} | CROSS/CTRL grad={out['CROSS'][1]/max(out['CTRL'][1],1e-9):.2f}", flush=True)
print("DISTINCT-DONE")
