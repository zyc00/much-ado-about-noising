"""Use the trained UNCONDITIONAL action manifold (denoiser) as a MEASUREMENT (not a
projector): for held-out test states, measure how far MSE-2k's vs MIP-2k's predicted
action chunk is FROM the clean-action manifold, binned by obs distance-to-clean-support.
manifold-deviation(x) = RMS||D(x,sigma) - x||  (on-manifold -> ~0; off-manifold -> large).
Hypothesis: off obs-support, MSE actions leave the action manifold while MIP actions
stay on it. Calibration floor = deviation of TRUE clean action chunks. No env."""
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
from manifold_train_denoiser import Denoiser

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
    ap.add_argument("--denoiser", default="analysis/manifold/denoiser_uncond20k.pt")
    ap.add_argument("--clean_n", type=int, default=40); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"
    cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", args.clean, "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", args.clean, "mip")
    AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    ddev = cfg.optimization.device

    ck = torch.load(args.denoiser, weights_only=False)
    D = Denoiser(ck["dim"]).to(dev); D.load_state_dict(ck["state_dict"]); D.eval()
    SIGS = [0.05, 0.2, 0.5]

    def deviation(an):  # an (1,H,10) normalized -> RMS||D-an|| at each sigma
        x = torch.tensor(an.reshape(1, -1), device=dev, dtype=torch.float32)
        outs = []
        with torch.no_grad():
            for sg in SIGS:
                d = D(x, torch.full((1, 1), sg, device=dev)) - x
                outs.append(float((d.pow(2).mean()).sqrt()))
        return outs

    # clean-chunk calibration floor (true on-manifold action chunks, 2k-na space)
    A2 = na.normalize(ds.replay_buffer["action"][:]).astype(np.float32)
    ends = np.asarray(ds.replay_buffer.episode_ends[:]); st = np.concatenate([[0], ends[:-1]])
    floor = {s: [] for s in SIGS}; cnt = 0
    for a, b in zip(st, ends):
        seg = A2[a:b]
        for i in range(0, len(seg) - args.H + 1, 50):
            dv = deviation(seg[i:i + args.H][None])
            for j, s in enumerate(SIGS): floor[s].append(dv[j])
            cnt += 1
            if cnt > 3000: break
        if cnt > 3000: break
    print(f"clean-chunk floor (on-manifold, N={cnt}):  " + "  ".join(f"s{s}:{np.mean(floor[s]):.3f}" for s in SIGS))

    clean = read(args.clean, args.clean_n); cl = np.concatenate([ov for ov, _ in clean], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6; tree = cKDTree((cl - mu) / sig)
    def dist(v): return float(tree.query((v - mu) / sig)[0])
    test = read(args.test)

    def chunk_an(ag, win):
        w = np.stack(win)[None]
        ot = {"state": torch.tensor(no.normalize(w), device=ddev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, args.H, 10), device=ddev), obs=ot, use_ema=True)
        return an.detach().cpu().numpy()

    dists, devM, devP = [], [], []
    for ov, acts in test:
        T = len(acts)
        for t in range(1, T):
            if t % args.stride or t + AS > T: continue
            win = [ov[t - 1], ov[t]]
            devM.append(deviation(chunk_an(mse, win)))
            devP.append(deviation(chunk_an(mip, win)))
            dists.append(dist(ov[t]))
    dists = np.array(dists); devM = np.array(devM); devP = np.array(devP)
    print(f"\ncaptured {len(dists)} states.  manifold-deviation RMS||D(x,sigma)-x||")
    for si, s in enumerate(SIGS):
        print(f"\n--- sigma={s} (floor {np.mean(floor[s]):.3f}) ---  {'bin':8} {'N':>6} {'MSE dev':>9} {'MIP dev':>9} {'MSE/MIP':>8}")
        for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 1e9)]:
            m = (dists >= lo) & (dists < hi); n = int(m.sum())
            if n < 10:
                print(f"          [{lo},{hi})  {n:>6}  -- few"); continue
            dm = devM[m, si].mean(); dp = devP[m, si].mean()
            print(f"          [{lo},{hi})  {n:>6} {dm:>9.3f} {dp:>9.3f} {dm/max(dp,1e-6):>8.2f}")
    np.savez("analysis/manifold/deviation.npz", dist=dists, devM=devM, devP=devP, sigs=np.array(SIGS))
    print("\nsaved analysis/manifold/deviation.npz")


if __name__ == "__main__":
    main()
