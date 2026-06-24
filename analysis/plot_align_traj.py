"""Roll out a policy from two start distances and record the ALIGNMENT dynamics
(frame_mount -> hole xy distance, and eef height) per step, to show why far
starts (B60) diverge in the align phase while near starts (B50) converge.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import robosuite
import matplotlib.pyplot as plt
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
    ap.add_argument("--ckpt", default="logs/rr80_regression_2000/models/model_latest.pt")
    ap.add_argument("--dataset", default="data/tool_hang_rr80_2000.hdf5")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max_chunks", type=int, default=30)
    args = ap.parse_args()

    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.dataset)}",
            "network=chiunet", "optimization.loss_type=regression",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device
    H = 16; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def feats():
        sim = env.sim
        hc = np.mean([sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")] for i in range(4)], axis=0)
        fm = sim.data.site_xpos[sim.model.site_name2id("frame_mount_site")]
        eef = env._get_observations(force_update=True)["robot0_eef_pos"]
        return float(np.linalg.norm((fm - hc)[:2])), float(eef[2])

    def rollout(states, seeds, i):
        np.random.seed(int(seeds[i])); env.reset()
        env.sim.set_state_from_flattened(states[i]); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]
        arm.update(); arm.reset_goal()
        o = env._get_observations(force_update=True)
        for _ in range(5):
            o, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        hist = [ov(o), ov(o)]; xy = []; ez = []; asm = False
        for _ in range(args.max_chunks):
            w = np.stack(hist[-2:])[None]
            ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
            act7 = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]
            for a in act7:
                o, _, _, _ = env.step(a); hist.append(ov(o))
                d, z = feats(); xy.append(d); ez.append(z)
                if env._check_frame_assembled():
                    asm = True; break
            if asm:
                break
        return np.array(xy), np.array(ez), asm

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex='col')
    for col, tag in enumerate(["b50", "b60"]):
        es = h5py.File(f"data/{tag}_eval_states.hdf5", "r")
        st = es["states"][:]; sd = es["seeds"][:]; es.close()
        for i in range(args.n):
            xy, ez, asm = rollout(st, sd, i)
            t = np.arange(len(xy))
            c = "tab:green" if asm else "tab:red"
            axes[0, col].plot(t, xy, color=c, alpha=0.6, lw=1.3)
            axes[1, col].plot(t, ez, color=c, alpha=0.6, lw=1.3)
        axes[0, col].axhline(0.002, color="k", ls="--", lw=0.8)
        axes[0, col].set_title(f"start {tag.upper()}  (green=success, red=fail)")
        axes[0, col].set_ylabel("frame->hole xy dist (m)"); axes[0, col].set_ylim(0, 0.35)
        axes[1, col].set_ylabel("eef height z (m)"); axes[1, col].set_xlabel("step")
        axes[0, col].grid(alpha=0.3); axes[1, col].grid(alpha=0.3)
    plt.tight_layout()
    out = "analysis/align_traj_B50_vs_B60.png"
    plt.savefig(out, dpi=130); print("SAVED", out)


if __name__ == "__main__":
    main()
