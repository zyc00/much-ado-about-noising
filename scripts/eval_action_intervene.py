"""Causal test: does reducing JERK (or action magnitude) at INFERENCE reduce the
off-support escape rate and improve SR? Distinguishes 'jerk is causal' from
'jerk is a symptom; the off-support action is just WRONG (wrong direction)'.

Interventions applied to the executed action (dims 0:6; gripper dim 6 untouched):
  --smooth BETA : EMA across steps  a_exec = b*a_pred + (1-b)*a_prev  (b=1 -> none) -> cuts jerk
  --scale ALPHA : a_exec[:6] = alpha * a_pred[:6]                      (a=1 -> none) -> cuts magnitude

Reports per config: SR (frame_assembled), escape rate (% steps OOD>tau to clean
cloud), and realized jerk (to confirm the intervention worked).

  MUJOCO_GL=egl python scripts/eval_action_intervene.py --model mse --configs base,smooth0.5,smooth0.3,scale0.7,scale0.5
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


def load(ckpt, ds_path, loss):
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
    ap.add_argument("--model", default="mse", choices=["mse", "mip"])
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--configs", default="base,smooth0.5,smooth0.3,scale0.7,scale0.5")
    args = ap.parse_args()

    spec = {"mse": ("logs/full_regression_2000/models/model_latest.pt", "data/tool_hang_full2ins_2000.hdf5", "regression"),
            "mip": ("logs/full_mip_2000/models/model_latest.pt", "data/tool_hang_full2ins_2000.hdf5", "mip")}[args.model]
    cfg, ds, ag = load(*spec)
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def predict(window):
        w = np.stack(window[-2:])[None]
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

    def run(beta, alpha):
        succ = 0; esc_steps = 0; tot_steps = 0; jerks = []
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
            steps = 0; asm = False; prev_exec = None
            while steps < args.max_steps and not asm:
                for a in predict(hist):
                    ae = a.copy()
                    ae[:6] = alpha * ae[:6]                       # magnitude scale
                    if prev_exec is not None:
                        ae[:6] = beta * ae[:6] + (1 - beta) * prev_exec[:6]   # EMA smooth (jerk)
                    if prev_exec is not None:
                        jerks.append(float(np.linalg.norm(ae[:6] - prev_exec[:6])))
                    prev_exec = ae.copy()
                    o, _, _, _ = env.step(np.clip(ae, -1, 1)); steps += 1; v = ov(o); hist.append(v)
                    tot_steps += 1; esc_steps += int(ood(v) > args.tau)
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            succ += int(asm)
        N = len(eval_keys)
        return 100 * succ / N, 100 * esc_steps / max(tot_steps, 1), float(np.mean(jerks))

    print(f"INTERVENE model={args.model} n={len(eval_keys)} tau={args.tau}")
    print(f"  {'config':12s}  SR%    escape%   mean_jerk")
    for c in args.configs.split(","):
        beta, alpha = 1.0, 1.0
        if c.startswith("smooth"): beta = float(c[6:])
        elif c.startswith("scale"): alpha = float(c[5:])
        elif c == "base": pass
        sr, esc, jk = run(beta, alpha)
        print(f"  {c:12s}  {sr:5.1f}  {esc:6.1f}   {jk:.3f}", flush=True)


if __name__ == "__main__":
    main()
