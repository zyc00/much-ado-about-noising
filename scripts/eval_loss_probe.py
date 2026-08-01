"""EXP-ET5: loss/accuracy vs recovery-geometry timeline.
For one checkpoint: step-1 action prediction error (normalized-chunk MSE and raw
first-action L2) on (a) TRAIN clean demos (2k file), (b) HELD-OUT clean demos
(20kB file, demo index >= 2000 — never seen by 2k-trained models).
Operator metrics for the same ckpts come from eval_traj_probe; join by tag.
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

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
H = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--loss", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--held", default="data/tool_hang_full2ins_20kB.hdf5")
    ap.add_argument("--n_demos", type=int, default=60)
    ap.add_argument("--held_lo", type=int, default=5000)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.clean)}", "network=chiunet",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(args.ckpt, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    def pred_first(win):
        x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                an = ag.flow_map_ema.get_velocity(torch.zeros(1, device=dev),
                                                  torch.zeros((1, H, 10), device=dev), emb)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :6]

    def eval_file(path, lo, n):
        h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"
        errs = []
        for i in range(lo, lo + n):
            o = h[f"{g}/demo_{i}/obs"]
            ov = np.concatenate([np.asarray(o[k]) for k in OK], axis=1).astype(np.float32)
            acts = np.clip(np.asarray(h[f"{g}/demo_{i}/actions"]), -1, 1).astype(np.float32)
            for t in range(1, len(acts) - H, 8):
                e = pred_first([ov[t - 1], ov[t]]) - acts[t, :6]
                errs.append(np.linalg.norm(e))
        h.close()
        errs = np.array(errs)
        return errs

    tr = eval_file(args.clean, 0, args.n_demos)
    hd = eval_file(args.held, args.held_lo, args.n_demos)
    print(f"LOSSPROBE {args.tag} trainErr_p50={np.median(tr):.4f} trainErr_mean={tr.mean():.4f} "
          f"heldErr_p50={np.median(hd):.4f} heldErr_mean={hd.mean():.4f} "
          f"heldErr_p95={np.percentile(hd,95):.4f} gap={hd.mean()/max(tr.mean(),1e-9):.2f}x")


if __name__ == "__main__":
    main()
