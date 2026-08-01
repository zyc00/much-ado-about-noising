"""3-seed action-averaged ensemble of hetero-t policies, standalone rollout eval
(committee smoothing of annulus wiggle; deployment-side)."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import robosuite
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = os.environ.get("DSP")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
def load():
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            "optimization.loss_type=regression_hetero_t", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
CKS = os.environ["CKS"].split(",")
agents = []
cfg, ds, _ = load()
dev = cfg.optimization.device
for ck in CKS:
    _, _, ag = load()
    ag.load(ck, load_optimizer=False); ag.eval()
    agents.append(ag)
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
def ov(o): return np.concatenate([np.asarray(o[KM.get(k, k)]) for k in OK]).astype(np.float32)
def chunk(hist):
    w = np.stack(hist[-2:])[None]
    ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
    outs = []
    with torch.no_grad():
        for ag in agents:
            outs.append(ag.sample(act_0=torch.randn((1, 16, 10), device=dev), obs=ot, use_ema=True))
    an = torch.stack(outs).mean(0)
    return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]
env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
succ_n = 0
NEP = int(os.environ.get("NEP", "50"))
for j, sd in enumerate(range(31000, 31000 + NEP)):
    np.random.seed(sd); torch.manual_seed(sd); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(10): env.step(np.zeros(7))
    o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
    steps, succ = 0, False
    while steps < 800 and not succ:
        for a in chunk(hist):
            o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
            if env._check_success(): succ = True; break
            if steps >= 800: break
    succ_n += int(succ)
    print(f"ENS ep{j} seed={sd} succ={int(succ)} running={succ_n}/{j+1}", flush=True)
print(f"ENSEMBLE FINAL: {succ_n}/{NEP}")
