"""Single-policy rollouts with eef dump. Env: CKPT, LOSS, OUT, N_SEEDS."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, robosuite
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
from scripted_tool_hang_v2 import ENV_KWARGS
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS = "data/tool_hang_full2ins_2000.hdf5"
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        f"+task.dataset_path={os.path.abspath(DS)}", "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); ag.load(os.environ["CKPT"], load_optimizer=False); ag.eval()
dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
def chunk_of(hist):
    w = np.stack(hist[-2:])[None]
    ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((1, 16, 10), device=dev), obs=ot, use_ema=True)
    return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]
env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
out = os.environ.get("OUT", "logs/roll_dump"); os.makedirs(out, exist_ok=True)
succ = 0; n = int(os.environ.get("N_SEEDS", "24"))
for sd in range(21000, 21000 + n):
    np.random.seed(sd); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(10): env.step(np.zeros(7))
    o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
    steps = 0; asm = False
    while steps < 700 and not asm:
        for a in chunk_of(hist):
            o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
            if env._check_frame_assembled(): asm = True; break
            if steps >= 700: break
    succ += int(asm)
    np.savez(f"{out}/ep_{sd}.npz", eef=np.array([h[44:47] for h in hist]), asm=int(asm), seed=sd)
print(f"ROLLDUMP {os.environ.get('LOSS')}: SR={succ}/{n}")
