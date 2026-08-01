"""Closed-loop MANIFOLD-DEVIATION probe: does the policy DRIFT off the expert
state manifold during a full-task rollout, and does it RETURN?

Hypothesis (from val-loss result): MIP's full-task win over MSE is NOT better
open-loop single-step prediction (MIP's teacher-forced val loss is equal/worse),
but better CLOSED-LOOP recovery from covariate shift. Signature:
  - MSE: nearest-expert-state distance grows monotonically (drifts away, no return)
  - MIP: distance rises then FALLS (drifts then recovers to the manifold)

Method: build the expert manifold = pooled, z-scored obs vectors from held-out
scripted demos (replayed). Roll out the policy closed-loop from init (reset +
10-step settle), and at every step record the cKDTree nearest-neighbour distance
to the manifold. Aggregate per normalized rollout-progress bin, split by whether
the episode ultimately assembled. Saves an npz for MSE-vs-MIP plotting.

Usage:
  MUJOCO_GL=egl python scripts/eval_manifold_deviation.py \
    --ckpt logs/full_mip_20kB/models/model_latest.pt \
    --dataset data/tool_hang_full2ins_20kB.hdf5 --loss mip \
    --demos data/warmstart_demos.hdf5 --n 50 --manifold_n 40 --out_npz mip.npz --tag gen_MIP
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)  # normalizer (model's training data)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=50)            # eval rollouts (eval seeds)
    ap.add_argument("--manifold_n", type=int, default=40)   # demos to build manifold (disjoint slice)
    ap.add_argument("--manifold_stride", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=700)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--out_npz", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.dataset)}",
            "network=chiunet", f"optimization.loss_type={args.loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device
    H = 16; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    df = h5py.File(args.demos, "r")
    allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]
    man_keys = allk[args.n:args.n + args.manifold_n] or allk[:args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    # ---- build expert manifold (replay held-out demos, pool obs) ----
    man = []
    for k in man_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); acts = np.clip(d["actions"][:], -1, 1)
        state0 = d["state0"][:]
        np.random.seed(sd); env.reset()
        env.sim.set_state_from_flattened(state0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        man.append(ov(env._get_observations(force_update=True)))
        for t in range(len(acts)):
            o, _, _, _ = env.step(acts[t])
            if t % args.manifold_stride == 0:
                man.append(ov(o))
    man = np.array(man)
    mu = man.mean(0); sigma = man.std(0) + 1e-6
    tree = cKDTree((man - mu) / sigma)
    print(f"[manifold] {len(man)} states from {len(man_keys)} demos, dim={man.shape[1]}", flush=True)

    # ---- closed-loop rollouts, record nearest-manifold distance per step ----
    per_ep = []  # list of (progress[], dist[], success)
    n_succ = 0
    for k in eval_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"])
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        dists = []; steps = 0; asm = False
        while steps < args.max_steps and not asm:
            for a in chunk(hist):
                o, _, _, _ = env.step(a); steps += 1
                v = ov(o); hist.append(v)
                dd, _ = tree.query((v - mu) / sigma)
                dists.append(float(dd))
                if env._check_frame_assembled():
                    asm = True; break
                if steps >= args.max_steps:
                    break
        per_ep.append((np.array(dists), bool(asm)))
        n_succ += int(asm)
    df.close()

    # ---- aggregate: bin each episode's dist curve onto [0,1] progress ----
    B = args.bins
    def binned(curve):
        if len(curve) < 2:
            return np.full(B, np.nan)
        xp = np.linspace(0, 1, len(curve))
        return np.interp(np.linspace(0, 1, B), xp, curve)
    succ_curves = np.array([binned(c) for c, s in per_ep if s]) if n_succ else np.zeros((0, B))
    fail_curves = np.array([binned(c) for c, s in per_ep if not s]) if (len(per_ep) - n_succ) else np.zeros((0, B))
    allc = np.array([binned(c) for c, _ in per_ep])

    print(f"MANIFOLD {args.tag} n={len(per_ep)} success={n_succ}/{len(per_ep)}")
    print(f"  ALL  dist@start={np.nanmean(allc[:,0]):.3f} @mid={np.nanmean(allc[:,B//2]):.3f} @end={np.nanmean(allc[:,-1]):.3f} peak={np.nanmax(np.nanmean(allc,0)):.3f}")
    if len(succ_curves):
        print(f"  SUCC dist@start={np.nanmean(succ_curves[:,0]):.3f} @mid={np.nanmean(succ_curves[:,B//2]):.3f} @end={np.nanmean(succ_curves[:,-1]):.3f} peak={np.nanmax(np.nanmean(succ_curves,0)):.3f}")
    if len(fail_curves):
        print(f"  FAIL dist@start={np.nanmean(fail_curves[:,0]):.3f} @mid={np.nanmean(fail_curves[:,B//2]):.3f} @end={np.nanmean(fail_curves[:,-1]):.3f} peak={np.nanmax(np.nanmean(fail_curves,0)):.3f}")
    # full mean curve (for the recover-vs-drift signature)
    mc = np.nanmean(allc, 0)
    print("  mean-curve(20bins): " + " ".join(f"{x:.2f}" for x in mc))
    if args.out_npz:
        np.savez(args.out_npz, all=allc, succ=succ_curves, fail=fail_curves,
                 n=len(per_ep), n_succ=n_succ, tag=args.tag)
        print(f"  saved {args.out_npz}")


if __name__ == "__main__":
    main()
