"""Splice deployment with video rendering: policy A until gripper closure+OFF, then B.
Same logic as eval_splice_g2.py; saves one mp4 per episode to OUTDIR.
Env: A_CKPT/A_LOSS/A_DS/B_CKPT/B_LOSS/SW_OFF/TAG/OUTDIR/SEED_N."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py, imageio
import robosuite
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
from scripted_tool_hang_v2 import ENV_KWARGS

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS = "data/tool_hang_full2ins_2000.hdf5"

def load_agent(loss, ck, ds_path=None):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(ds_path or DS)}", "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ck, load_optimizer=False); ag.eval()
    return cfg, ds, ag

cfgA, dsA, agA = load_agent(os.environ["A_LOSS"], os.environ["A_CKPT"], os.environ.get("A_DS"))
cfgB, dsB, agB = load_agent(os.environ["B_LOSS"], os.environ["B_CKPT"])
dev = cfgA.optimization.device; AS = cfgA.task.act_steps; start = cfgA.task.obs_steps - 1
OFF = int(os.environ.get("SW_OFF", "16"))
TAG = os.environ.get("TAG", "splice"); OUTDIR = os.environ.get("OUTDIR", "logs/splice_vids")
os.makedirs(OUTDIR, exist_ok=True)

def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
def chunk_of(ag, ds, cfg, hist):
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    w = np.stack(hist[-2:])[None]
    ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((1, 16, 10), device=dev), obs=ot, use_ema=True)
    return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

EK = dict(ENV_KWARGS); EK["has_offscreen_renderer"] = True
env = robosuite.make("ToolHang", horizon=4000, **EK)
succ = 0; N = 0
for sd in range(21000, 21000 + int(os.environ.get("SEED_N", "12"))):
    np.random.seed(sd); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(10): env.step(np.zeros(7))
    o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
    steps = 0; asm = False; locked = False; closed_at = None; frames = []
    while steps < 700 and not asm:
        if not locked and closed_at is not None and steps - closed_at >= OFF: locked = True
        useA = not locked
        ch = chunk_of(agA, dsA, cfgA, hist) if useA else chunk_of(agB, dsB, cfgB, hist)
        if useA and closed_at is None and ch[:, 6].max() >= 0:
            closed_at = steps
            if OFF == 0:
                locked = True; ch = chunk_of(agB, dsB, cfgB, hist); useA = False
        for a in ch:
            o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
            if steps % 2 == 0:
                frames.append(np.ascontiguousarray(env.sim.render(camera_name="sideview", width=512, height=512)[::-1]))
            if env._check_frame_assembled(): asm = True; break
            if steps >= 700: break
    succ += int(asm); N += 1
    out = f"{OUTDIR}/{TAG}_seed{sd}_{'succ' if asm else 'FAIL'}.mp4"
    imageio.mimsave(out, frames, fps=18, macro_block_size=1)
    print(f"{out} steps={steps}", flush=True)
print(f"RENDER-SPLICE {TAG}: SR={succ}/{N}")
print("RENDER-DONE")
