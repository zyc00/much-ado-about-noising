"""Sweep the handoff threshold theta_on for the two-way rescue-switch (base MSE-2k,
rescuer takes over when d_clean>=theta_on, hands back when d_clean<=theta_on-0.3).
Maps SR vs theta_on from pure-MSE (theta=inf, 68%) to pure-rescuer (theta=0).
Distinguishes 'theta=2 was too late' from 'recovery is not modular / advantage is
trajectory-wide'. If SR only approaches the rescuer's SR as theta->0 (rescuer drives
everything), the advantage is trajectory-wide, NOT an off-support-only rescue skill."""
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
    ap.add_argument("--thetas", default="2.5,1.79,1.3,0.8")
    args = ap.parse_args()

    base_cfg = build_cfg("regression")
    ds = make_dataset(base_cfg.task)
    dev = base_cfg.optimization.device; AS = base_cfg.task.act_steps; start = base_cfg.task.obs_steps - 1
    n2 = ds.normalizer
    nD = torch.load("analysis/recovery/norm_puredart6k.pt", weights_only=False)
    mse = TrainingAgent(build_cfg("regression")); mse.load("logs/full_regression_2000/models/model_latest.pt", load_optimizer=False); mse.eval()
    mip = TrainingAgent(build_cfg("mip")); mip.load("logs/full_mip_2000/models/model_latest.pt", load_optimizer=False); mip.eval()
    dart = TrainingAgent(build_cfg("regression")); dart.load("logs/puredart_mse/models/model_latest.pt", load_optimizer=False); dart.eval()
    P_MSE = dict(ag=mse, no=n2["obs"]["state"], na=n2["action"])
    P_MIP = dict(ag=mip, no=n2["obs"]["state"], na=n2["action"])
    P_DART = dict(ag=dart, no=nD["obs"]["state"], na=nD["action"])

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

    def run(resc, theta_on):  # theta_on=inf -> pure MSE ; theta_on<=0 -> pure resc
        theta_off = theta_on - 0.3
        succ = 0; rsteps = 0; tot = 0
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
            steps = 0; asm = False; on = (theta_on <= 0); cur = ood(hist[-1])
            while steps < args.max_steps and not asm:
                if not on and cur >= theta_on: on = True
                elif on and cur <= theta_off: on = False
                p = resc if on else P_MSE
                for a in pred(p, hist):
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v); cur = ood(v)
                    tot += 1; rsteps += int(on)
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            succ += int(asm)
        return 100 * succ / len(eval_keys), 100 * rsteps / max(tot, 1)

    thetas = [float(x) for x in args.thetas.split(",")]
    print(f"n={args.n}.  pure MSE=68, pure MIP=92, pure DART=98 (reference).")
    for name, resc in [("MIP", P_MIP), ("DART", P_DART)]:
        print(f"\n  RESCUER={name}:  theta_on -> SR (rescuer drove %)")
        sr0, _ = run(resc, 0.0)
        print(f"    theta=0   (pure {name}) | SR={sr0:.0f}%", flush=True)
        for th in thetas:
            sr, frac = run(resc, th)
            print(f"    theta={th:<5}             | SR={sr:.0f}%   (rescuer drove {frac:.0f}%)", flush=True)
    df.close()


if __name__ == "__main__":
    main()
