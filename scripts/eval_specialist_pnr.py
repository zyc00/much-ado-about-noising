"""Does a 20k SPECIALIST stay in-support / avoid PNR? Roll out the specialist on its
segment (insert: replay to align_done then policy inserts; grasp: from init) and at each
POLICY step track distance to (a) its OWN training support and (b) the FULL-clean support
(the PNR=4 metric used for the full task). If it rarely leaves its support / never nears
PNR, that confirms the 97% comes from short-horizon in-support operation, not robustness.
Distances z-scored by FULL-task stats (comparable to the full-task PNR=4)."""
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


def stack(path, ndemo, stride=1):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g])[:ndemo]
    out = [np.concatenate([np.asarray(h[f"{g}/{k}/obs"][key]) for key in OK], axis=1).astype(np.float32)[::stride] for k in ks]
    h.close(); return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)         # normalizer (specialist training data)
    ap.add_argument("--support_data", required=True)     # own-support cloud source
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--warm_to", default="align_done")   # align_done (insert) | 0 (grasp)
    ap.add_argument("--success", default="assembled")    # assembled | grasp
    ap.add_argument("--init_mode", default="state0")     # state0 | reset_settle
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--max_steps", type=int, default=700)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.dataset)}", "network=chiunet",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device; H = 16; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    # clouds, z-scored by FULL-task stats (comparable to full-task PNR=4)
    full = stack("data/tool_hang_full2ins_20000.hdf5", 200)
    mu, sig = full.mean(0), full.std(0) + 1e-6
    own = stack(args.support_data, 150)
    tree_own = cKDTree((own[np.random.RandomState(0).choice(len(own), min(12000, len(own)), replace=False)] - mu) / sig)
    tree_full = cKDTree((full[np.random.RandomState(1).choice(len(full), 12000, replace=False)] - mu) / sig)
    def d_own(v): return float(tree_own.query((v - mu) / sig)[0])
    def d_full(v): return float(tree_full.query((v - mu) / sig)[0])

    df = h5py.File(args.demos if False else "data/warmstart_demos.hdf5", "r")
    keys = list(df["demos"].keys())[:args.n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def is_success():
        if args.success == "grasp":
            grasped = env._check_grasp(gripper=env.robots[0].gripper["right"], object_geoms=env.frame.contact_geoms)
            fz = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")][2]
            return bool(grasped and fz > 0.86)
        return bool(env._check_frame_assembled())

    succ = 0; per_seed = []
    for k in keys:
        d = df["demos/" + k]; seed = int(d.attrs["seed"]); acts = d["actions"][:]; state0 = d["state0"][:]
        K = int(d.attrs[args.warm_to]) if args.warm_to in ("c1", "align_done") else (min(int(args.warm_to), len(acts)) if args.warm_to.isdigit() else 0)
        np.random.seed(seed); env.reset()
        if args.init_mode == "state0":
            env.sim.set_state_from_flattened(state0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        o = env._get_observations(force_update=True)
        if args.init_mode == "reset_settle":
            for _ in range(args.settle):
                o, _, _, _ = env.step(np.zeros(7))
        prev = ov(o); cur = ov(o); asm = False; steps = 0
        for t in range(K):
            o, _, _, _ = env.step(np.clip(acts[t], -1, 1)); steps += 1; prev = cur; cur = ov(o)
            if is_success(): asm = True; break
        do_max = 0.0; df_max = 0.0; policy_steps = 0
        if not asm:
            hist = [prev, cur]
            while steps < args.max_steps and not asm:
                w = np.stack(hist[-2:])[None]
                ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
                with torch.no_grad():
                    an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
                a7 = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]
                for a in a7:
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
                    do_max = max(do_max, d_own(v)); df_max = max(df_max, d_full(v)); policy_steps += 1
                    if is_success(): asm = True; break
                    if steps >= args.max_steps: break
        succ += int(asm); per_seed.append((asm, do_max, df_max, policy_steps))
    df.close()
    n = len(per_seed)
    print(f"\nSPECIALIST SR = {succ}/{n} = {100*succ/n:.1f}%  (policy steps/seed median {int(np.median([p[3] for p in per_seed]))})")
    dm_own = np.array([p[1] for p in per_seed]); dm_full = np.array([p[2] for p in per_seed])
    print(f"max distance to OWN support per seed:  median {np.median(dm_own):.2f}  p90 {np.percentile(dm_own,90):.2f}  max {dm_own.max():.2f}")
    print(f"max distance to FULL support per seed: median {np.median(dm_full):.2f}  p90 {np.percentile(dm_full,90):.2f}  max {dm_full.max():.2f}")
    for thr in [2, 3, 4]:
        print(f"  seeds ever reaching d_full>{thr} (full-task PNR={4}): {100*np.mean(dm_full>thr):.0f}%   d_own>{thr}: {100*np.mean(dm_own>thr):.0f}%")


if __name__ == "__main__":
    main()
