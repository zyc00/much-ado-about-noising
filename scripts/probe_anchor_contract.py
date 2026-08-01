"""Anchor-argument contraction of MIP step-2: at fixed states, feed anchors t*(a+delta)
with ||delta|| ~ typical step-1 error; rho = ||F(s, anchor) - a|| / ||delta||.
rho < 1 => step-2 shrinks action errors (the +13 mechanism candidate).
Measured at demo support states and at hetero-t fail states."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys, h5py
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = os.environ.get("DSP", "x")
def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
cfg, ds, ag = load("mip")
dev = cfg.optimization.device
ag.load(os.environ["CK_MIP"], load_optimizer=False); ag.eval()
N = 512
idx = np.random.RandomState(0).choice(len(ds), N, replace=False)
OW, AC = [], []
for i in idx:
    b = ds[int(i)]
    o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
    OW.append(o[:2].numpy()); AC.append(b["action"][:16].numpy())
OW = np.stack(OW); AC = np.stack(AC)
for DN in [0.05, 0.1, 0.2, 0.5]:
    rhos = []
    for b0 in range(0, N, 128):
        w = torch.tensor(OW[b0:b0+128], device=dev, dtype=torch.float32)
        a = torch.tensor(AC[b0:b0+128], device=dev)
        e = ag.encoder(w, None)
        tt = torch.full((len(a),), 0.9, device=dev)
        d = torch.randn_like(a); d = d / d.reshape(len(a), -1).norm(dim=1).view(-1, 1, 1) * DN * np.sqrt(160)
        with torch.no_grad():
            out = ag.flow_map.get_velocity(tt, a + d, e)
        num = (out - a).reshape(len(a), -1).norm(dim=1)
        den = d.reshape(len(a), -1).norm(dim=1)
        rhos.extend((num / den).cpu().numpy().tolist())
    print(f"CONTRACT |delta|={DN} (per-dim): rho p50={np.median(rhos):.2f} p90={np.quantile(rhos,0.9):.2f}", flush=True)
print("CONTRACT-DONE")
