"""OOD-robustness curve for an insertion specialist: replay expert to c1 (in-dist
grasped state), then PERTURB the eef by a random direction (gripper stays closed,
carrying the frame to an off-distribution handoff pose), then the insert specialist
takes over. Sweep perturbation magnitude -> closed-loop assembled SR. Expect MSE
to collapse fast, MIP to degrade slowly (OOD robustness).
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
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss", default="regression"); ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--mag", type=float, default=0.0)       # perturbation pos-delta magnitude
    ap.add_argument("--psteps", type=int, default=5)        # perturbation steps
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(args.dataset)}",
            "network=chiunet", f"optimization.loss_type={args.loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device; H = 16; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); keys = list(df["demos"].keys())[:args.n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]

    succ = 0
    for k in keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); c1 = int(d.attrs["c1"]); acts = np.clip(d["actions"][:], -1, 1)
        state0 = d["state0"][:]
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(state0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        # replay expert to c1 (in-dist grasped state)
        for t in range(c1):
            env.step(acts[t])
        # perturb: push eef in a fixed random dir (gripper closed) -> OOD handoff pose
        rng = np.random.RandomState(sd + 7); dvec = rng.randn(3); dvec /= (np.linalg.norm(dvec) + 1e-9)
        for _ in range(args.psteps):
            a = np.concatenate([dvec * args.mag, [0, 0, 0], [1.0]]).astype(np.float64)
            env.step(np.clip(a, -1, 1))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        asm = False; steps = 0
        while steps < 600 and not asm:
            for a in chunk(hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                if env._check_frame_assembled(): asm = True; break
                if steps >= 600: break
        succ += int(asm)
    N = len(keys)
    print(f"OODPERTURB {args.tag} mag={args.mag} psteps={args.psteps} {succ}/{N} = {100*succ/N:.1f}%")


if __name__ == "__main__":
    main()
