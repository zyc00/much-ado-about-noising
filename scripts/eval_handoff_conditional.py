"""Conditional handoff analysis: stage1 grasp policy -> grasped(); record the
handoff deviation (frame-in-EEF vs the expert grasp mean); stage2 insert
specialist takes over; record assembled. Then bin insert success rate by handoff
deviation, to test whether failures concentrate in the large-deviation tail.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import robosuite
import robosuite.utils.transform_utils as T
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}


def load(ds, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(ds)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    return cfg, make_dataset(cfg.task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp_ckpt", required=True); ap.add_argument("--grasp_ds", required=True)
    ap.add_argument("--back_ckpt", required=True); ap.add_argument("--back_ds", required=True)
    ap.add_argument("--loss", default="regression"); ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--grasp_budget", type=int, default=160)
    ap.add_argument("--max_steps", type=int, default=700); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfgG, dsG = load(args.grasp_ds, args.loss); cfgB, dsB = load(args.back_ds, args.loss)
    dev = cfgG.optimization.device; H = 16; AS = cfgG.task.act_steps; start = cfgG.task.obs_steps - 1
    aG = TrainingAgent(cfgG); aG.load(args.grasp_ckpt, load_optimizer=False); aG.eval()
    aB = TrainingAgent(cfgB); aB.load(args.back_ckpt, load_optimizer=False); aB.eval()
    noG = dsG.normalizer["obs"]["state"]; naG = dsG.normalizer["action"]
    noB = dsB.normalizer["obs"]["state"]; naB = dsB.normalizer["action"]
    df = h5py.File(args.demos, "r"); keys = list(df["demos"].keys())[:args.n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def grasped():
        g = env._check_grasp(gripper=env.robots[0].gripper["right"], object_geoms=env.frame.contact_geoms)
        fz = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")][2]
        return bool(g and fz > 0.86)
    def fie():
        o = env._get_observations(force_update=True)
        ep = o["robot0_eef_pos"]; eR = T.quat2mat(o["robot0_eef_quat"])
        fp = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")]
        return eR.T @ (fp - ep)
    def chunk(agent, no, na, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return dsG.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]

    # global expert grasp mean (frame-in-eef)
    epose = []
    for k in keys[:20]:
        d = df["demos/" + k]; acts = np.clip(d["actions"][:], -1, 1); s0 = d["state0"][:]; sd = int(d.attrs["seed"])
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for t in range(len(acts)):
            env.step(acts[t])
            if grasped(): epose.append(fie()); break
    exp_mean = np.mean(epose, axis=0)

    recs = []  # (dev_cm, asm)
    for k in keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"])
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; gd = False
        while steps < args.grasp_budget and not gd:
            for a in chunk(aG, noG, naG, hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))  # step-return obs (match eval_twostage; force_update diverges rollout)
                if grasped(): gd = True; break
                if steps >= args.grasp_budget: break
        if not gd:
            continue
        devcm = float(np.linalg.norm(fie() - exp_mean) * 100)
        asm = False
        while steps < args.max_steps and not asm:
            for a in chunk(aB, noB, naB, hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                if env._check_frame_assembled(): asm = True; break
                if steps >= args.max_steps: break
        recs.append((devcm, int(asm)))
    df.close()
    recs = np.array(recs); dvs = recs[:, 0]; asms = recs[:, 1]
    print(f"HANDOFF_COND {args.tag} n={len(recs)} overall={int(asms.sum())}/{len(recs)}={100*asms.mean():.0f}%")
    print(f"  dev success={dvs[asms==1].mean():.2f}±{dvs[asms==1].std():.2f}cm | dev FAIL={dvs[asms==0].mean():.2f}±{dvs[asms==0].std():.2f}cm")
    for lo, hi in [(0, .3), (.3, .6), (.6, 1.0), (1.0, 99)]:
        m = (dvs >= lo) & (dvs < hi)
        if m.sum():
            print(f"  dev[{lo:.1f},{hi if hi<99 else 'inf'}) n={m.sum():2d} SR={100*asms[m].mean():.0f}%")


if __name__ == "__main__":
    main()
