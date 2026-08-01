"""Braking-point (competing risks): from the first time the OOD-score crosses a
level A, which happens FIRST -- return below LO=1 (BRAKE / back to support) or
reach HI=4 (ESCALATE / point of no return)? P(brake|A) vs A, per method.
The braking point = largest A where the policy can still mostly brake."""
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
    ap.add_argument("--lo", type=float, default=1.0); ap.add_argument("--hi", type=float, default=4.0)
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

    def rollout(ag):
        seqs = []
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; asm = False; dd = []
            while steps < args.max_steps and not asm:
                for a in pred(ag, hist):
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v); dd.append(ood(v))
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            seqs.append(np.array(dd))
        return seqs

    LO, HI = args.lo, args.hi
    grid = [1.5, 2.0, 2.5, 3.0, 3.5]
    for nm, ag in [("MSE", mse), ("MIP", mip)]:
        seqs = rollout(ag)
        print(f"=== {nm}  (LO={LO} brake, HI={HI} escalate) ===")
        print("  level A  | n_cross  P(brake)  P(escalate)  P(neither)")
        for A in grid:
            nb = ne = nn = 0
            for d in seqs:
                idx = np.argmax(d >= A) if (d >= A).any() else -1
                if idx < 0: continue
                seg = d[idx:]
                tb = np.argmax(seg < LO) if (seg < LO).any() else 1e9
                te = np.argmax(seg >= HI) if (seg >= HI).any() else 1e9
                if tb == 1e9 and te == 1e9: nn += 1
                elif tb < te: nb += 1
                else: ne += 1
            tot = nb + ne + nn
            if tot == 0: continue
            print(f"  A={A:<4}   | {tot:3d}      {100*nb/tot:5.0f}%     {100*ne/tot:5.0f}%      {100*nn/tot:5.0f}%")
    df.close()


if __name__ == "__main__":
    main()
