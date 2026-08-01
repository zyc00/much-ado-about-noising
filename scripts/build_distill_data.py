"""Distillation dataset: relabel the human demos (and one jittered clone each) with the
MLP-MSE teacher's per-step actions. Output: robomimic-format hdf5 -> train chiunet
regression on it unchanged. The teacher is the relabeling oracle that fixed-label
augmentation lacked."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, h5py, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = os.environ["DSP"]; OUT = os.environ.get("OUT", "data/tool_hang_distmlp.hdf5")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
DIMS = {"object": 44, "robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2}
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + DSP, "network=mlp",
        "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 10
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
ag.load(os.environ["TEACHER"], load_optimizer=False); ag.eval()
dev = cfg.optimization.device
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
start = cfg.task.obs_steps - 1
hin = h5py.File(DSP, "r")
hout = h5py.File(OUT, "w")
grp = hout.create_group("data")
keys = sorted(hin["data"].keys(), key=lambda k: int(k.split("_")[-1]))
JIT = float(os.environ.get("JIT", "0.08"))
wrote = 0
for k in keys:
    src = hin[f"data/{k}"]
    ov = np.concatenate([np.asarray(src["obs"][q]) for q in OK], axis=1).astype(np.float32)
    T = len(ov)
    for variant in range(1 if os.environ.get('NOJIT') else 2):
        obs_v = ov.copy()
        if variant == 1:
            obs_n = no.normalize(obs_v)
            obs_n = obs_n + JIT * np.random.randn(*obs_n.shape).astype(np.float32)
            obs_v = no.unnormalize(obs_n)
        # teacher per-step actions: windows (t-1, t), batched
        wins = np.stack([np.stack([obs_v[max(t - 1, 0)], obs_v[t]]) for t in range(T)])
        acts = np.zeros((T, 7), dtype=np.float32)
        B = 512
        for b in range(0, T, B):
            w = torch.tensor(no.normalize(wins[b:b + B]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                an = ag.sample(act_0=torch.randn((len(w), 10, 10), device=dev), obs={"state": w}, use_ema=True)
            a7 = ds.undo_transform_action(na.unnormalize(an.cpu().numpy())[:, start:start + 1])
            acts[b:b + B] = a7[:, 0]
        g = grp.create_group(f"demo_{wrote}")
        og = g.create_group("obs")
        c = 0
        for q in OK:
            og.create_dataset(q, data=obs_v[:, c:c + DIMS[q]])
            c += DIMS[q]
        g.create_dataset("actions", data=acts)
        g.attrs["num_samples"] = T
        wrote += 1
hout.close(); hin.close()
print(f"DISTILL wrote {wrote} demos -> {OUT}")
