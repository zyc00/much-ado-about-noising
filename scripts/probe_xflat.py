"""Is the trained function x-flat? Compare f(s,0,0) vs f(s,eps,0) vs f(s,3eps,0). Env: CKPT, LOSS."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
DS = "data/tool_hang_full2ins_2000.hdf5"
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        f"+task.dataset_path={os.path.abspath(DS)}", "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); ag.load(os.environ["CKPT"], load_optimizer=False); ag.eval()
dev = cfg.optimization.device
no = ds.normalizer["obs"]["state"]
enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
h = h5py.File(DS, "r")
wins = []
for i in range(6):
    o = h[f"data/demo_{i}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    for t in range(10, len(ov)-2, 60): wins.append(np.stack([ov[t-1], ov[t]]))
h.close()
wins = np.stack(wins)
x = torch.tensor(no.normalize(wins), device=dev, dtype=torch.float32)
with torch.no_grad():
    e = enc({"state": x}, None); t0 = torch.zeros(len(wins), device=dev)
    y0 = fm.get_velocity(t0, torch.zeros((len(wins),16,10), device=dev), e)
    ds1, ds3 = [], []
    for k in range(3):
        torch.manual_seed(k)
        eps = torch.randn_like(y0)
        y1 = fm.get_velocity(t0, eps, e); y3 = fm.get_velocity(t0, 3*eps, e)
        ds1.append((y1-y0).flatten(1).norm(dim=1)); ds3.append((y3-y0).flatten(1).norm(dim=1))
    n0 = y0.flatten(1).norm(dim=1).median().item()
    d1 = torch.stack(ds1).median().item(); d3 = torch.stack(ds3).median().item()
print(f"XFLAT {os.environ.get('TAG','')}: |f(s,0)|={n0:.2f}  |f(s,eps)-f(s,0)| p50={d1:.4f}  |f(s,3eps)-f(s,0)| p50={d3:.4f}")
