"""Break down the GRASP-phase open-loop prediction loss by position within the
grasp phase (early reach / mid approach / late pre-grasp near c1), to locate
WHERE the generalist-vs-specialist gap lives. Also reports the cosine alignment
between predicted and expert action directions per bucket.
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
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--tag", default="")
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

    def predict(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]

    buckets = ["early", "mid", "late"]
    se = {b: 0.0 for b in buckets}; cnt = {b: 0 for b in buckets}; cos = {b: 0.0 for b in buckets}
    for k in keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); c1 = int(d.attrs["c1"]); acts = np.clip(d["actions"][:], -1, 1)
        state0 = d["state0"][:]
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(state0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        for t in range(c1):
            if t >= 1 and (t + AS <= len(acts)):
                pred = predict(hist); tgt = acts[t:t+AS]
                err = float(np.mean((pred - tgt) ** 2))
                frac = t / max(c1, 1)
                b = "early" if frac < 1/3 else ("mid" if frac < 2/3 else "late")
                se[b] += err; cnt[b] += 1
                pf = pred.flatten(); tf = tgt.flatten()
                cos[b] += float(np.dot(pf, tf) / (np.linalg.norm(pf) * np.linalg.norm(tf) + 1e-9))
            o, _, _, _ = env.step(acts[t]); hist.append(ov(o))
    df.close()
    out = " ".join(f"{b}=MSE{se[b]/max(cnt[b],1):.5f},cos{cos[b]/max(cnt[b],1):.3f}(n{cnt[b]})" for b in buckets)
    print(f"GRASPBD {args.tag} {out}")


if __name__ == "__main__":
    main()
