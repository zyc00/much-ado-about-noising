"""Causal sufficiency test: inject distance-calibrated noise into MIP so its
error-vs-distance MATCHES MSE's (sigma(d)=sqrt(MSE_err(d)^2-MIP_err(d)^2) from the
keystone). Does MIP then inherit MSE's escape/failure? If yes -> error-MAGNITUDE
vs distance is the sufficient cause. If MIP stays robust -> the error STRUCTURE
(systematic/directional), not just magnitude, matters.

Noise added per-element to executed action dims [:6] (pos+axisangle; gripper untouched).
sigma(d) piecewise from keystone, scaled by --scale. scale=0 -> pure MIP.

  MUJOCO_GL=egl python scripts/eval_distnoise.py --n 40 --scales 0,0.5,1,2
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

# calibrated sigma(d) to match MSE error-vs-distance (from eval_support_error keystone)
def sigma_of(d):
    if d < 1: return 0.035
    if d < 2: return 0.025
    if d < 4: return 0.315
    return 0.291


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
    ap.add_argument("--tau", type=float, default=5.0); ap.add_argument("--scales", default="0,0.5,1,2")
    ap.add_argument("--noise_seed", type=int, default=0)
    args = ap.parse_args()
    cfg, ds, mip = load("logs/full_mip_2000/models/model_latest.pt", "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def pred(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = mip.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
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

    def run(scale):
        rng = np.random.RandomState(args.noise_seed)
        succ = 0; esc = 0; tot = 0
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
            steps = 0; asm = False; cur_d = ood(hist[-1])
            while steps < args.max_steps and not asm:
                for a in pred(hist):
                    ae = a.copy()
                    s = scale * sigma_of(cur_d)
                    if s > 0:
                        ae[:6] = np.clip(ae[:6] + rng.randn(6) * s, -1, 1)   # noise gated by CURRENT distance
                    o, _, _, _ = env.step(ae); steps += 1; v = ov(o); hist.append(v)
                    cur_d = ood(v); tot += 1; esc += int(cur_d > args.tau)
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            succ += int(asm)
        return 100 * succ / len(eval_keys), 100 * esc / max(tot, 1)

    print(f"DISTNOISE inject sigma(d) x scale into MIP  (sigma calibrated to MSE-MIP error gap)")
    print(f"  scale | MIP SR% | escape%   (scale=0 pure MIP; scale=1 -> MIP error ~ MSE error)")
    for sc in [float(x) for x in args.scales.split(",")]:
        sr, e = run(sc)
        print(f"  {sc:<5} | {sr:5.1f}  | {e:5.1f}", flush=True)
    print("  (ref: pure MSE ~67/40=67.5% escape~34%)")
    df.close()


if __name__ == "__main__":
    main()
