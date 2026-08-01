"""Capture MSE-clean-20k bias on the SAME test states / SAME clean-support distance
metric as err_vectors.npz, then print the unified 4-way bias(d) table:
MSE-clean-2k, MIP-2k (from err_vectors.npz), DART-6k (from err_dart6k.npz),
MSE-clean-20k (captured here). Clean-20k uses its own local dataset normalizer."""
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
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


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


def read(path, nmax=None):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g].keys())
    if nmax: ks = ks[:nmax]
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"{g}/{k}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/dart_test_huge_full2ins.hdf5")
    ap.add_argument("--ckpt", default="logs/full_regression_20000/models/model_latest.pt")
    ap.add_argument("--ds", default="data/tool_hang_full2ins_20000.hdf5")
    ap.add_argument("--clean_n", type=int, default=40); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()

    cfg, ds, ag = load(args.ckpt, args.ds, "regression")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    clean = read(args.clean, args.clean_n); cl = np.concatenate([ov for ov, _ in clean], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6; tree = cKDTree((cl - mu) / sig)
    def dist(v): return float(tree.query((v - mu) / sig)[0])
    test = read(args.test)

    def predict(window):
        w = np.stack(window)[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    dists, e20 = [], []
    for ov, acts in test:
        T = len(acts)
        for t in range(1, T):
            if t % args.stride or t + AS > T: continue
            win = [ov[t - 1], ov[t]]; gt = acts[t:t + AS, :6]
            e20.append(predict(win)[:, :6] - gt)
            dists.append(dist(ov[t]))
    dists = np.array(dists); e20 = np.array(e20)
    print(f"captured {len(dists)} states; err shape {e20.shape}")
    np.savez("analysis/recovery/err_clean20k.npz", dist=dists, e20=e20)

    ref = np.load("analysis/recovery/err_vectors.npz")
    rd = ref["dist"]; rM = ref["eMSE"][:, 0, :]; rP = ref["eMIP"][:, 0, :]
    dk = np.load("analysis/recovery/err_dart6k.npz"); dD = dk["dist"]; eD = dk["eDART"][:, 0, :]
    e0 = e20[:, 0, :]
    print(f"\n{'bin':10} {'|MSE-2k|':>9} {'|MSE-20k|':>10} {'|MIP-2k|':>9} {'|DART-6k|':>10}")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 1e9)]:
        def b(d, e):
            m = (d >= lo) & (d < hi)
            return np.linalg.norm(e[m].mean(0)) if m.sum() >= 10 else float("nan")
        print(f"[{lo},{hi})    {b(rd,rM):>9.3f} {b(dists,e0):>10.3f} {b(rd,rP):>9.3f} {b(dD,eD):>10.3f}")
    print("\nsaved analysis/recovery/err_clean20k.npz")


if __name__ == "__main__":
    main()
