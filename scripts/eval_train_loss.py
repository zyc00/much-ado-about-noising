"""Per-phase TRAINING-set loss (open-loop action-prediction MSE), mirroring
eval_val_loss.py but reading STORED obs+actions from a robomimic-format hdf5
(no env replay). All models are scored on the SAME full init->insertion
trajectories (training seeds) so train-loss aligns 1:1 with the val-loss table.
Error = MSE(pred_chunk, expert_chunk) in RAW action space. Phase: t<c1 grasp.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)   # for normalizer (model's training data)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--demos", required=True)     # full2ins training hdf5 (full trajectories, train seeds)
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

    def predict(win):
        w = np.stack(win)[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]

    f = h5py.File(args.demos, "r")
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:args.n]
    se = {"grasp": 0.0, "insert": 0.0}; cnt = {"grasp": 0, "insert": 0}
    for k in keys:
        d = f["data/" + k]
        obs = np.concatenate([d["obs"][ok][:] for ok in OBS_KEYS], axis=1).astype(np.float32)  # (T,53)
        acts = np.clip(d["actions"][:], -1, 1).astype(np.float32)                                # (T,7)
        g = acts[:, 6]; cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
        c1 = cl[0] if cl else len(acts)
        T = len(acts)
        for t in range(1, T):
            if (t % args.stride == 0) and (t + AS <= T):
                pred = predict([obs[t-1], obs[t]])
                err = float(np.mean((pred - acts[t:t+AS]) ** 2))
                ph = "grasp" if t < c1 else "insert"
                se[ph] += err; cnt[ph] += 1
    f.close()
    gg = se["grasp"] / max(cnt["grasp"], 1); ii = se["insert"] / max(cnt["insert"], 1)
    print(f"TRAINLOSS {args.tag} grasp_phase={gg:.5f} (n={cnt['grasp']}) "
          f"insert_phase={ii:.5f} (n={cnt['insert']})")


if __name__ == "__main__":
    main()
