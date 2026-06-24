"""Per-phase validation loss (open-loop action-prediction MSE) on held-out demos.

Replays each held-out expert trajectory (set_state(state0) -> reset_goal -> replay
expert actions), and at strided timesteps has the model predict the action chunk
from the current obs window. Error = MSE(pred_chunk, expert_chunk) in RAW action
space (normalizer-independent -> comparable across generalist/specialist & MSE/MIP).
Grouped by phase: t < c1 = grasp, t >= c1 = insertion.

Usage:
  MUJOCO_GL=egl python scripts/eval_val_loss.py --ckpt <m> --dataset <ds> --loss mip|regression
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)  # for normalizer (model's training data)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--stride", type=int, default=2)
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
    keys = list(df["demos"].keys())[:args.n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def predict(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]

    se = {"grasp": 0.0, "insert": 0.0}; cnt = {"grasp": 0, "insert": 0}
    for k in keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); c1 = int(d.attrs["c1"])
        acts = np.clip(d["actions"][:], -1, 1); state0 = d["state0"][:]
        np.random.seed(sd); env.reset()
        env.sim.set_state_from_flattened(state0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]
        arm.update(); arm.reset_goal()
        o = env._get_observations(force_update=True)
        hist = [ov(o), ov(o)]
        T = len(acts)
        for t in range(T):
            if t >= 1 and (t % args.stride == 0) and (t + AS <= T):
                pred = predict(hist)                       # (AS, 7) raw
                tgt = acts[t:t + AS]                        # (AS, 7) raw
                err = float(np.mean((pred - tgt) ** 2))
                ph = "grasp" if t < c1 else "insert"
                se[ph] += err; cnt[ph] += 1
            o, _, _, _ = env.step(acts[t]); hist.append(ov(o))
    df.close()
    g = se["grasp"] / max(cnt["grasp"], 1); i = se["insert"] / max(cnt["insert"], 1)
    print(f"VALLOSS {args.tag} grasp_phase={g:.5f} (n={cnt['grasp']}) "
          f"insert_phase={i:.5f} (n={cnt['insert']})")


if __name__ == "__main__":
    main()
