"""Bridge check: per-sample ||dL/demb|| vs ||dL/dpred|| — is the Jacobian factor
flat across deciles/phases (metrics interchangeable) or systematic?"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import numpy as np
import torch
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

DSP = os.environ["DSP"]
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + DSP, "network=chiunet",
        "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); dev = cfg.optimization.device
ag.load(os.environ["CKPT"], load_optimizer=False)
ag.encoder.eval(); ag.flow_map.eval()
N = 4096
idx = np.random.RandomState(0).choice(len(ds), N, replace=False)
def get_batch(ids):
    xs, ys = [], []
    for i in ids:
        b = ds[int(i)]
        o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
        xs.append(o[:2]); ys.append(b["action"])
    return torch.stack(xs).to(dev), torch.stack(ys).to(dev)
Xs, Ys = get_batch(idx)
flatX = Xs.reshape(N, -1).cpu().numpy(); flatY = Ys.reshape(N, -1).cpu().numpy()
D = torch.cdist(torch.tensor(flatX), torch.tensor(flatX))
nbr = D.topk(9, largest=False).indices[:, 1:].numpy()
knn_dev = np.linalg.norm(flatY - flatY[nbr].mean(1), axis=1)
gs_emb, gs_out = [], []
for b0 in range(0, N, 256):
    xb, yb = get_batch(idx[b0:b0 + 256])
    emb = ag.encoder(xb, None)
    emb_r = emb.detach().requires_grad_(True)
    t0 = torch.zeros(len(xb), device=dev)
    pred = ag.flow_map.get_velocity(t0, torch.zeros_like(yb), emb_r)
    li = ((pred - yb) ** 2).sum(dim=(1, 2))
    g = torch.autograd.grad(li.sum(), emb_r)[0]
    gs_emb.append(g.reshape(len(xb), -1).norm(dim=1).detach().cpu().numpy())
    gs_out.append((2 * (pred - yb)).reshape(len(xb), -1).norm(dim=1).detach().cpu().numpy())
ge = np.concatenate(gs_emb); go = np.concatenate(gs_out)
ratio = ge / (go + 1e-12)
from scipy.stats import spearmanr
print(f"BRIDGE spearman(emb,out)={spearmanr(ge, go).correlation:.3f} | ratio p10/50/90 = {np.percentile(ratio,10):.3f}/{np.percentile(ratio,50):.3f}/{np.percentile(ratio,90):.3f} (spread {np.percentile(ratio,90)/np.percentile(ratio,10):.2f}x)")
order = np.argsort(knn_dev)
for d in range(10):
    sel = order[d*N//10:(d+1)*N//10]
    print(f"BRIDGE decile D{d}: ratio_med={np.median(ratio[sel]):.3f} emb_med={np.median(ge[sel]):.4f} out_med={np.median(go[sel]):.4f}")
print("BRIDGE-DONE")
