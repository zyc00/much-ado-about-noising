"""Rescue test: roll out with MSE; the first time OOD-score crosses tau, hand
control to MIP for the rest of the episode. Does switching to MIP BEFORE the
point-of-no-return (~3.5-4) rescue the rollout? Sweep tau.
tau=inf -> pure MSE; tau=0 -> pure MIP. Same normalizer/obs -> clean handoff."""
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
    args = ap.parse_args()
    cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
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
    def ood(v): return float(tree.query((v - mu) / sig)[0])

    def run(tau):  # MSE until OOD>=tau, then MIP. tau=inf->MSE, tau<=0->MIP from start
        res = []
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
            steps = 0; asm = False; switched = (tau <= 0); sw_step = (0 if switched else -1)
            while steps < args.max_steps and not asm:
                ag = mip if switched else mse
                for a in pred(ag, hist):
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
                    if not switched and ood(v) >= tau:
                        switched = True; sw_step = steps
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
                if switched and ag is mse:  # re-enter loop with mip on next chunk
                    pass
            res.append((asm, sw_step))
        return res

    base_mse = run(float("inf")); base_mip = run(0.0)
    fail_mse = [not a for a, _ in base_mse]
    print(f"pure MSE: {sum(a for a,_ in base_mse)}/{len(base_mse)}   pure MIP: {sum(a for a,_ in base_mip)}/{len(base_mip)}")
    print(f"MSE-fail seeds: {sum(fail_mse)}")
    print("\n  switch tau | overall SR | rescued (of MSE-fail) | mean switch step")
    for tau in [3.0, 2.5, 2.0, 1.5]:
        r = run(tau)
        sr = sum(a for a, _ in r)
        rescued = sum(1 for i in range(len(r)) if fail_mse[i] and r[i][0])
        sws = [s for a, s in r if s > 0]
        print(f"  tau={tau:<4}   | {sr:2d}/{len(r)}      | {rescued}/{sum(fail_mse)}                | {np.mean(sws):.0f}")
    df.close()


if __name__ == "__main__":
    main()
