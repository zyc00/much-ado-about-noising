"""Rescue at INSERT onset (not early): MSE drives reach+grasp+approach; the moment
the frame enters the hole region (grasped & frame_mount xy-to-hole < D) -- i.e.
BEFORE the insertion push / final rise -- hand control to MIP for the rest.
Separates failure modes: grasp-phase fails never trigger the handoff (MIP can't
rescue them); insert-phase fails do -> shows whether MIP-insertion rescues them.

  MUJOCO_GL=egl python scripts/eval_switch_insert.py --n 40 --D 0.10,0.06
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import robosuite
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS = "data/tool_hang_full2ins_2000.hdf5"


def load(ckpt, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(DS)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    return cfg, ds, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--D", default="0.10,0.06")
    args = ap.parse_args()
    cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def grasped():
        g = env._check_grasp(gripper=env.robots[0].gripper["right"], object_geoms=env.frame.contact_geoms)
        fz = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")][2]
        return bool(g and fz > 0.86)
    def hole_xy():
        hc = np.mean([env.sim.data.geom_xpos[env.sim.model.geom_name2id(f"stand_wall{i}")] for i in range(4)], axis=0)
        return float(np.linalg.norm((env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")] - hc)[:2]))
    def pred(ag, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    man = []
    for k in man_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); acts = np.clip(d["actions"][:], -1, 1); s0 = d["state0"][:]
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        man.append(ov(env._get_observations(force_update=True)))
        for t in range(len(acts)):
            o, _, _, _ = env.step(acts[t]); man.append(ov(o))
    man = np.array(man); mu, sig = man.mean(0), man.std(0) + 1e-6
    tree = cKDTree((man - mu) / sig)

    def run(D):  # D<0 -> pure MSE; D>=99 -> pure MIP; else switch at grasped & hole_xy<D
        res = []
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
            steps = 0; asm = False; switched = (D >= 99); sw_step = (0 if switched else -1)
            while steps < args.max_steps and not asm:
                ag = mip if switched else mse
                for a in pred(ag, hist):
                    o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                    if (not switched) and D >= 0 and D < 99 and grasped() and hole_xy() < D:
                        switched = True; sw_step = steps
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            res.append((asm, sw_step))
        return res

    base = run(-1); pmip = run(99)
    failmse = [not a for a, _ in base]
    nf = sum(failmse)
    print(f"pure MSE {sum(a for a,_ in base)}/{len(base)}   pure MIP {sum(a for a,_ in pmip)}/{len(pmip)}   MSE-fail={nf}")
    print("\n  handoff @ hole_xy<D | overall SR | rescued/MSE-fail | never-switched(=grasp-fail) | mean sw step")
    for D in [float(x) for x in args.D.split(",")]:
        r = run(D)
        sr = sum(a for a, _ in r)
        rescued = sum(1 for i in range(len(r)) if failmse[i] and r[i][0])
        never = sum(1 for a, s in r if s < 0)  # never reached insert region
        sws = [s for a, s in r if s > 0]
        print(f"  D={D:<5}            | {sr:2d}/{len(r)}      | {rescued}/{nf}              | {never}                          | {np.mean(sws):.0f}")
    df.close()


if __name__ == "__main__":
    main()
