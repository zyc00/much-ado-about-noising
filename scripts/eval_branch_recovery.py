"""Matched-OOD-state branch test: the FALSIFIABLE version of "MIP recovers from
OOD where MSE can't". Removes the confound that MIP/MSE visit DIFFERENT OOD states.

Procedure:
  1. Roll out a SOURCE policy from init; at the first step its state drifts into a
     mild-but-OOD band (obs_dist in [lo,hi]), SNAPSHOT the full mujoco state.
     (Collect one snapshot per seed -> a set of matched OOD states.)
  2. From each identical snapshot, BRANCH: run MIP for K steps, and (separately,
     restoring the same state) run MSE for K steps.
  3. Recovery = min obs_dist over the branch < tau_lo (returned to manifold) OR
     frame assembled. Compare MIP vs MSE recovery FROM THE SAME STATES.

If MIP recovers from matched OOD states and MSE does not -> the recovery edge is
real and causal. If they're equal -> the H1 "action prior -> recovery" story is wrong.

  MUJOCO_GL=egl python scripts/eval_branch_recovery.py --mip_ckpt ... --mse_ckpt ... \
      --dataset data/tool_hang_full2ins_2000.hdf5 --demos data/warmstart_demos.hdf5 \
      --source mse --n 50 --lo 1.5 --hi 4.0 --K 48
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


def load_agent(ckpt, ds_path, loss, dev_holder):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(ds_path)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    return cfg, ds, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mip_ckpt", required=True); ap.add_argument("--mse_ckpt", required=True)
    ap.add_argument("--dataset", required=True)  # shared normalizer source (full2ins, same for both)
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--source", default="mse", choices=["mse", "mip"])
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--manifold_n", type=int, default=40)
    ap.add_argument("--lo", type=float, default=1.5); ap.add_argument("--hi", type=float, default=4.0)
    ap.add_argument("--K", type=int, default=48); ap.add_argument("--src_budget", type=int, default=400)
    ap.add_argument("--tau_lo", type=float, default=1.0); ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()

    cfg, ds, a_mip = load_agent(args.mip_ckpt, args.dataset, "mip", None)
    _, _, a_mse = load_agent(args.mse_ckpt, args.dataset, "regression", None)
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def chunk(agent, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    # manifold
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
    def odist(v): return float(tree.query((v - mu) / sig)[0])
    print(f"[manifold] {len(man)} states", flush=True)

    src_agent = a_mse if args.source == "mse" else a_mip

    # 1. collect matched OOD snapshots from the source policy's own drift
    snaps = []  # (sim_state_flat, hist_last2)
    for k in eval_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"])
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; grabbed = False
        while steps < args.src_budget and not grabbed:
            for a in chunk(src_agent, hist):
                o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
                dd = odist(v)
                if args.lo <= dd <= args.hi:
                    snaps.append((env.sim.get_state().flatten().copy(), [hist[-2].copy(), hist[-1].copy()], dd))
                    grabbed = True; break
                if steps >= args.src_budget: break
    print(f"[snapshots] collected {len(snaps)} matched OOD states (obs_dist in [{args.lo},{args.hi}], source={args.source})", flush=True)

    # 2. branch each snapshot with MIP and MSE from the SAME state
    def branch(agent, sflat, hist0):
        env.reset()
        env.sim.set_state_from_flattened(sflat); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        hist = [hist0[0].copy(), hist0[1].copy()]
        mind = odist(hist[-1]); steps = 0; asm = False
        while steps < args.K and not asm:
            for a in chunk(agent, hist):
                o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
                mind = min(mind, odist(v))
                if env._check_frame_assembled(): asm = True; break
                if steps >= args.K: break
        return mind, asm

    rec_mip = rec_mse = asm_mip = asm_mse = 0
    dmip, dmse = [], []
    for sflat, hist0, d0 in snaps:
        m_min, m_asm = branch(a_mip, sflat, hist0)
        s_min, s_asm = branch(a_mse, sflat, hist0)
        rec_mip += int(m_min < args.tau_lo or m_asm); asm_mip += int(m_asm)
        rec_mse += int(s_min < args.tau_lo or s_asm); asm_mse += int(s_asm)
        dmip.append(m_min); dmse.append(s_min)
    df.close()
    N = len(snaps)
    print(f"BRANCH source={args.source} n={N} band[{args.lo},{args.hi}] K={args.K}")
    print(f"  MIP: recover(min<{args.tau_lo} or asm)={rec_mip}/{N}={100*rec_mip/max(N,1):.0f}%  assembled={asm_mip}/{N}  min_obs_dist mean={np.mean(dmip):.2f}")
    print(f"  MSE: recover(min<{args.tau_lo} or asm)={rec_mse}/{N}={100*rec_mse/max(N,1):.0f}%  assembled={asm_mse}/{N}  min_obs_dist mean={np.mean(dmse):.2f}")


if __name__ == "__main__":
    main()
