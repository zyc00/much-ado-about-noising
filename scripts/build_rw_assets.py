"""Build RW_ASSETS npz: tube center/sigma (120 knots, 300 demos) + exact affine
coefs mapping NORMALIZED obs eef dims -> raw meters (probed from the actual
normalizer, so any affine normalizer form works). Round-trip verified."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset

DS = os.environ.get("DS", "data/tool_hang_full2ins_2000.hdf5")
PKNOTS = 120
hf = h5py.File(DS, "r")
keys = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
grid = np.linspace(0, 1, PKNOTS)
tube = []
for k in keys[:300]:
    p = np.asarray(hf[f"data/{k}/obs/robot0_eef_pos"]); t = np.linspace(0, 1, len(p))
    tube.append(np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], 1))
tube = np.stack(tube); center = tube.mean(0); sigma = tube.std(0).mean(1) + 1e-9
hf.close()

with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(DS), "network=chiunet",
        "optimization.loss_type=regression", "optimization.auto_resume=false",
        "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
ds = make_dataset(cfg.task)
no = ds.normalizer["obs"]["state"]
# probe affine: raw = A*n + B per dim (exact for any affine normalizer)
z = np.zeros((1, 53), dtype=np.float32)
B = no.unnormalize(z)[0]
A = np.zeros(53, dtype=np.float32)
for i in range(53):
    e = z.copy(); e[0, i] = 1.0
    A[i] = no.unnormalize(e)[0, i] - B[i]
# round-trip check on real data
hf = h5py.File(DS, "r")
OKk = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ov = np.concatenate([np.asarray(hf["data/demo_5/obs/" + k]) for k in OKk], axis=1).astype(np.float32)
hf.close()
n = no.normalize(ov[:50])
rec = n * A[None] + B[None]
err = np.abs(rec - ov[:50]).max()
print(f"roundtrip max err = {err:.2e}")
assert err < 1e-4, "affine probe failed"
tg = np.gradient(center, axis=0)
tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12)
na = ds.normalizer["action"]
z10 = np.zeros((1, 16, 10), dtype=np.float32)
Ba = na.unnormalize(z10)[0, 0]
Aa = np.zeros(10, dtype=np.float32)
for i in range(10):
    e = z10.copy(); e[0, :, i] = 1.0
    Aa[i] = na.unnormalize(e)[0, 0, i] - Ba[i]
out = os.environ.get("RW_OUT", "logs/rw_assets.npz")
np.savez(out, center=center, sigma=sigma, tangent=tg,
         obs_A=A[44:47], obs_B=B[44:47], act_A=Aa, act_B=Ba)
print(f"saved {out} | sigma_p50", float(np.median(sigma)))
