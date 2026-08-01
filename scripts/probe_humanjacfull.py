"""Pooled full-trajectory encoder-Jacobian kappa10/PR/k90 for the human models."""
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
def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
rng = np.random.RandomState(11)
S = []
for k in [keys[i] for i in rng.choice(len(keys), 30, replace=False)]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    t = rng.randint(1, len(ov) - 1)
    S.append(np.stack([ov[t-1], ov[t]]))
h.close()
S = np.stack(S)
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
NOBASE = ",".join(str(i) for i in range(0, 14))
MODELS = [
    ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt", None),
    ("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt", None),
    ("hMIP_s5001", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5001_success80.pt", None),
    ("hMSE_s5001", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5001_success60.pt", None),
    ("lhbase", "regression", "logs/lhbase/models/model_latest.pt", None),
    ("lhnobase", "regression", "logs/lhnobase/models/model_latest.pt", NOBASE),
]
for name, loss, ck, mask in MODELS:
    if mask: os.environ["OBS_MASK"] = mask
    else: os.environ.pop("OBS_MASK", None)
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema
    def fea(inp):
        return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
    k10s, prj, k90s = [], [], []
    for w in S:
        x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
        J = torch.autograd.functional.jacobian(fea, x, vectorize=True).squeeze(1).detach().cpu().numpy()
        sv = np.linalg.svd(J, compute_uv=False); e2 = sv ** 2
        cs = np.cumsum(e2) / e2.sum()
        k90s.append(int(np.searchsorted(cs, 0.90) + 1))
        prj.append(float(e2.sum()**2 / (e2**2).sum()))
        k10s.append(float(sv[0] / sv[min(9, len(sv)-1)]))
    print(f"HJACFULL {name}: k90={np.median(k90s):.0f} PR={np.median(prj):.2f} k10={np.median(k10s):.1f}", flush=True)
print("HJACFULL-DONE")
