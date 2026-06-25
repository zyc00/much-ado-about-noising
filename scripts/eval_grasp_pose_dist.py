"""Quantify how far a learned policy's GRASP distribution deviates from the
expert's. At the grasped() moment, record the frame pose in the EEF frame
(the grasp configuration that determines downstream insertion geometry), for
(a) expert (replay demo actions) and (b) a learned policy (rollout from init).
Reports systematic bias (|gen_mean - expert_mean|) and spread (std) in cm / deg.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss", default="regression"); ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--grasp_budget", type=int, default=160)
    ap.add_argument("--out_npz", default=None)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(args.dataset)}",
            "network=chiunet", f"optimization.loss_type={args.loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device; H = 16; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); keys = list(df["demos"].keys())[:args.n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def grasped():
        g = env._check_grasp(gripper=env.robots[0].gripper["right"], object_geoms=env.frame.contact_geoms)
        fz = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")][2]
        return bool(g and fz > 0.86)
    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]

    def frame_in_eef():
        o = env._get_observations(force_update=True)
        ep = o["robot0_eef_pos"]; eR = T.quat2mat(o["robot0_eef_quat"])
        fp = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")]
        fR = T.quat2mat(o["object-state"][3:7]) if False else T.quat2mat(o["robot0_eef_quat"])  # placeholder
        # frame orientation: use frame body quat
        fq = o["object-state"][3:7] if len(o.get("object-state", [])) >= 7 else None
        rel_p = eR.T @ (fp - ep)                       # frame pos in eef frame (m)
        # relative orientation angle (frame vs eef), via frame_quat from obs
        try:
            fRq = T.quat2mat(o["frame_quat"]); ang = np.degrees(np.arccos(np.clip((np.trace(eR.T @ fRq) - 1) / 2, -1, 1)))
        except Exception:
            ang = np.nan
        return rel_p, ang

    exp, gen = [], []
    for k in keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); acts = np.clip(d["actions"][:], -1, 1)
        state0 = d["state0"][:]
        # expert: replay to grasped()
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(state0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for t in range(len(acts)):
            env.step(acts[t])
            if grasped():
                p, a = frame_in_eef(); exp.append(np.r_[p, a]); break
        # learned policy: rollout from init to grasped()
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; got = False
        while steps < args.grasp_budget and not got:
            for a in chunk(hist):
                env.step(a); steps += 1; hist.append(ov(env._get_observations(force_update=True)))
                if grasped():
                    p, an = frame_in_eef(); gen.append(np.r_[p, an]); got = True; break
                if steps >= args.grasp_budget: break
    df.close()
    exp = np.array(exp); gen = np.array(gen)
    lbl = ["x(cm)", "y(cm)", "z(cm)", "ang(deg)"]
    sc = np.array([100, 100, 100, 1])
    print(f"GRASPDIST n_exp={len(exp)} n_gen={len(gen)}")
    for i, l in enumerate(lbl):
        em, es = np.nanmean(exp[:, i])*sc[i], np.nanstd(exp[:, i])*sc[i]
        gm, gs = np.nanmean(gen[:, i])*sc[i], np.nanstd(gen[:, i])*sc[i]
        print(f"  {l:9s} expert {em:7.2f}±{es:5.2f} | policy {gm:7.2f}±{gs:5.2f} | bias|Δmean|={abs(gm-em):5.2f} spread×={gs/(es+1e-6):.1f}")
    # overall positional deviation
    dp = np.linalg.norm((gen[:, :3].mean(0) - exp[:, :3].mean(0)))*100
    gen_spread = np.linalg.norm(gen[:, :3].std(0))*100; exp_spread = np.linalg.norm(exp[:, :3].std(0))*100
    print(f"  POS bias |Δmean|={dp:.2f}cm | expert spread={exp_spread:.2f}cm policy spread={gen_spread:.2f}cm")
    if args.out_npz:
        np.savez(args.out_npz, exp=exp, gen=gen)
        print(f"  saved {args.out_npz}")


if __name__ == "__main__":
    main()
