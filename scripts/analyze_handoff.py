"""Test the covariate-shift hypothesis: does the front policy hand off states
that lie OUTSIDE the align+insert (stage-2) training init distribution?

Compares geometric features of:
  (A) stage-2 init = true align_done-backoff states (aligninsert_eval_states)
  (B) front-policy handoff states, recorded at several chunk counts (overshoot?)

Features: frame_mount->hole xy dist, eef z, frame_mount z.
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


def feats(env):
    sim = env.sim

    def gs(n):
        return sim.data.site_xpos[sim.model.site_name2id(n)].copy()
    hc = np.mean([sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")] for i in range(4)], axis=0)
    fm = gs("frame_mount_site")
    eef = env._get_observations(force_update=True)["robot0_eef_pos"]
    return {
        "fm_hole_xy": float(np.linalg.norm((fm - hc)[:2])),
        "eef_z": float(eef[2]),
        "fm_z": float(fm[2]),
    }


def summ(name, rows):
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print(f"\n[{name}]  n={len(rows)}")
    for k, v in a.items():
        print(f"  {k:12s} mean={v.mean():.4f}  std={v.std():.4f}  "
              f"p10={np.percentile(v,10):.4f}  p90={np.percentile(v,90):.4f}")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front_ckpt", default="logs/front_regression_2000/models/model_latest.pt")
    ap.add_argument("--front_ds", default="data/tool_hang_front_2000.hdf5")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--chunk_points", nargs="+", type=int, default=[6, 8, 10, 14])
    args = ap.parse_args()

    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.front_ds)}",
            "network=chiunet", "optimization.loss_type=regression",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device
    H = 16; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.front_ckpt, load_optimizer=False); agent.eval()
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    # (A) stage-2 init distribution = true handoff states
    es = h5py.File("data/aligninsert_eval_states.hdf5", "r")
    a_states = es["states"][:]; a_seeds = es["seeds"][:]; es.close()
    A = []
    for i in range(min(args.n, len(a_states))):
        np.random.seed(int(a_seeds[i])); env.reset()
        env.sim.set_state_from_flattened(a_states[i]); env.sim.forward()
        A.append(feats(env))
    summ("A: stage-2 TRUE init (align_done-backoff)", A)

    # (B) front-policy handoff states at several chunk counts
    pes = h5py.File("data/pick_eval_states.hdf5", "r")
    p_states = pes["states"][:]; p_seeds = pes["seeds"][:]; pes.close()
    maxc = max(args.chunk_points)
    B = {c: [] for c in args.chunk_points}
    for i in range(min(args.n, len(p_states))):
        np.random.seed(int(p_seeds[i])); env.reset()
        env.sim.set_state_from_flattened(p_states[i]); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]
        arm.update(); arm.reset_goal()
        o = env._get_observations(force_update=True)
        for _ in range(5):
            o, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        hist = [ov(o), ov(o)]
        for c in range(1, maxc + 1):
            w = np.stack(hist[-2:])[None]
            ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
            act7 = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]
            for a in act7:
                o, _, _, _ = env.step(a); hist.append(ov(o))
            if c in B:
                B[c].append(feats(env))
    for c in args.chunk_points:
        summ(f"B: front handoff @ {c} chunks ({c*AS} steps)", B[c])


if __name__ == "__main__":
    main()
