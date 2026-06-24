"""Render two-stage specialist stitching with the SAME logic as
eval_twostage_faithful (stage1=policy to grasped(), reset_settle init), to mp4.
Renders the first N held-out seeds; filenames tag seed + OK/FAIL + handoff frame.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import imageio
import robosuite
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}


def load(ds_path, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(ds_path)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    return cfg, make_dataset(cfg.task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp_ckpt", required=True); ap.add_argument("--grasp_ds", required=True)
    ap.add_argument("--back_ckpt", required=True); ap.add_argument("--back_ds", required=True)
    ap.add_argument("--loss", default="regression"); ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--grasp_budget", type=int, default=160)
    ap.add_argument("--max_steps", type=int, default=700)
    ap.add_argument("--out", default="logs/stitch"); ap.add_argument("--cam", default="sideview"); ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    cfgG, dsG = load(args.grasp_ds, args.loss); cfgB, dsB = load(args.back_ds, args.loss)
    dev = cfgG.optimization.device; H = 16; AS = cfgG.task.act_steps; start = cfgG.task.obs_steps - 1
    aG = TrainingAgent(cfgG); aG.load(args.grasp_ckpt, load_optimizer=False); aG.eval()
    aB = TrainingAgent(cfgB); aB.load(args.back_ckpt, load_optimizer=False); aB.eval()
    noG = dsG.normalizer["obs"]["state"]; naG = dsG.normalizer["action"]
    noB = dsB.normalizer["obs"]["state"]; naB = dsB.normalizer["action"]

    kw = dict(ENV_KWARGS); kw["has_offscreen_renderer"] = True; kw["use_camera_obs"] = False
    kw["camera_names"] = args.cam; kw["camera_heights"] = args.res; kw["camera_widths"] = args.res
    env = robosuite.make("ToolHang", horizon=4000, **kw)
    df = h5py.File(args.demos, "r"); keys = list(df["demos"].keys())[:args.n]

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def fr(): return env.sim.render(camera_name=args.cam, width=args.res, height=args.res)[::-1]
    def grasped():
        g = env._check_grasp(gripper=env.robots[0].gripper["right"], object_geoms=env.frame.contact_geoms)
        fz = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")][2]
        return bool(g and fz > 0.86)
    def chunk(agent, no, na, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return dsG.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]

    for k in keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"])
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; frames = [fr()]
        steps = 0; gdone = False; asm = False
        while steps < args.grasp_budget and not gdone:
            for a in chunk(aG, noG, naG, hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o)); frames.append(fr())
                if grasped(): gdone = True; break
                if steps >= args.grasp_budget: break
        handoff = len(frames)
        while steps < args.max_steps and not asm:
            for a in chunk(aB, noB, naB, hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o)); frames.append(fr())
                if env._check_frame_assembled(): asm = True; break
                if steps >= args.max_steps: break
        path = f"{args.out}_seed{sd}_{'OK' if asm else 'FAIL'}.mp4"
        imageio.mimsave(path, frames, fps=30, macro_block_size=1)
        print(f"RENDERED {path} handoff@{handoff}/{len(frames)} grasped={gdone} assembled={asm}")
    df.close()


if __name__ == "__main__":
    main()
