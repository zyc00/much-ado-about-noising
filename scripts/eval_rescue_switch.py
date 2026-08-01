"""#1: TWO-WAY oracle rescue-switch. Base policy = MSE-clean-2k. When OOD distance to
clean support crosses theta_on, hand control to a RESCUER (MIP-2k or DART-6k); when it
drops back below theta_off (hysteresis), hand control BACK to MSE-2k. If MSE-2k + (off-
support-only rescuer) reaches the rescuer's own SR, that is conclusive evidence that the
ONLY thing MSE-2k lacks is off-support behavior -- in-support it is already fine.

Each policy uses its OWN normalizer (DART-6k differs). Distance ruler = clean cloud.
Reports: pure MSE-2k, pure rescuer, and the switched policy; plus % of control steps
spent on the rescuer (should be small if the rescuer only acts off-support)."""
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
DS_DECODE = "data/tool_hang_full2ins_2000.hdf5"


def build_cfg(loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(DS_DECODE)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--rescuer", default="MIP", choices=["MIP", "DART"])
    ap.add_argument("--theta_on", type=float, default=2.0)
    ap.add_argument("--theta_off", type=float, default=1.79)
    args = ap.parse_args()

    base_cfg = build_cfg("regression")
    ds = make_dataset(base_cfg.task)
    dev = base_cfg.optimization.device; AS = base_cfg.task.act_steps; start = base_cfg.task.obs_steps - 1
    n2 = ds.normalizer
    nD = torch.load("analysis/recovery/norm_puredart6k.pt", weights_only=False)

    mse = TrainingAgent(build_cfg("regression")); mse.load("logs/full_regression_2000/models/model_latest.pt", load_optimizer=False); mse.eval()
    if args.rescuer == "MIP":
        resc = TrainingAgent(build_cfg("mip")); resc.load("logs/full_mip_2000/models/model_latest.pt", load_optimizer=False); resc.eval()
        resc_norm = n2
    else:
        resc = TrainingAgent(build_cfg("regression")); resc.load("logs/puredart_mse/models/model_latest.pt", load_optimizer=False); resc.eval()
        resc_norm = nD
    POL = {"MSE": dict(ag=mse, no=n2["obs"]["state"], na=n2["action"]),
           "RESC": dict(ag=resc, no=resc_norm["obs"]["state"], na=resc_norm["action"])}

    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def pred(p, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(p["no"].normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = p["ag"].sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(p["na"].unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

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

    def rollout(mode):  # mode: 'MSE' | 'RESC' | 'SWITCH'
        succ = 0; resc_steps = 0; tot = 0; seedlog = []
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
            steps = 0; asm = False; on_resc = False; ever_resc = False
            cur = ood(hist[-1])
            while steps < args.max_steps and not asm:
                if mode == "SWITCH":
                    if not on_resc and cur >= args.theta_on:
                        on_resc = True
                    elif on_resc and cur <= args.theta_off:
                        on_resc = False
                    p = POL["RESC"] if on_resc else POL["MSE"]
                    if on_resc: ever_resc = True
                else:
                    p = POL[mode]
                for a in pred(p, hist):
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v); cur = ood(v)
                    tot += 1; resc_steps += int(mode == "SWITCH" and on_resc)
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            succ += int(asm)
            seedlog.append((sd, asm, ever_resc))
        return succ, 100 * resc_steps / max(tot, 1), seedlog

    sr_mse, _, log_mse = rollout("MSE")
    sr_resc, _, _ = rollout("RESC")
    sr_sw, resc_frac, log_sw = rollout("SWITCH")
    n = len(eval_keys)
    mse_fail = {sd for sd, a, _ in log_mse if not a}
    print(f"\nRESCUER = {args.rescuer}   theta_on={args.theta_on} theta_off={args.theta_off}  (n={n})")
    print(f"  pure MSE-2k          : {sr_mse}/{n} = {100*sr_mse/n:.0f}%")
    print(f"  pure {args.rescuer:<15}: {sr_resc}/{n} = {100*sr_resc/n:.0f}%")
    print(f"  MSE + {args.rescuer}-rescue  : {sr_sw}/{n} = {100*sr_sw/n:.0f}%   "
          f"(rescuer drove {resc_frac:.1f}% of all control steps)")
    nresc = sum(1 for _, _, er in log_sw if er)
    rescued_fails = sum(1 for sd, a, er in log_sw if a and er and sd in mse_fail)
    print(f"  seeds triggering rescue: {nresc}/{n}; assembled among them: "
          f"{sum(1 for _, a, er in log_sw if er and a)}/{nresc}")
    print(f"  of {len(mse_fail)} pure-MSE failures, rescued to success: {rescued_fails}")
    df.close()


if __name__ == "__main__":
    main()
