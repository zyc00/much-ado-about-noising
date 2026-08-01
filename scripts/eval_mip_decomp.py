"""#2: WHERE does MIP's off-support pull-back come from -- the trained network function,
or the 2nd refinement step? MIP inference is 2 calls:
  step1: a1 = get_velocity(s=0, zeros, obs)      <- functionally identical to MSE regression
  step2: a2 = get_velocity(t_two_step, a1, obs)  <- refine, feeding step1 back in
Decompose the off-support bias(d) into: MSE | MIP-step1-only | MIP-full(2-step).
  - if MIP-step1 already low  -> the win is the TRAINED FUNCTION (flow-matching objective
    shapes a better-extrapolating network).
  - if MIP-step1 ~ MSE but MIP-full low -> the win is the 2nd DENOISING step (inference).
Ground truth = scripted recovery action on held-out DART states. No env."""
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
from mip.samplers import regression_sampler, mip_step1_only_sampler, mip_sampler

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
    ap.add_argument("--clean_n", type=int, default=40); ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()
    cfg_m, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", args.clean, "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", args.clean, "mip")
    dev = cfg_m.optimization.device; AS = cfg_m.task.act_steps; start = cfg_m.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    optim = cfg_m.optimization

    clean = read(args.clean, args.clean_n); cl = np.concatenate([ov for ov, _ in clean], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6; tree = cKDTree((cl - mu) / sig)
    def dist(v): return float(tree.query((v - mu) / sig)[0])
    test = read(args.test)

    def decode(an):
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][:, :6]

    def run_sampler(ag, sampler, win):
        w = np.stack(win)[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        fm = ag.flow_map_ema; enc = ag.encoder_ema
        a0 = torch.zeros((1, args.H, 10), device=dev)
        with torch.no_grad():
            with ag._inference_mode():
                an = sampler(optim, fm, enc, a0, ot)
        return decode(an)

    dists, eMSE, eS1, eFULL = [], [], [], []
    for ov, acts in test:
        T = len(acts)
        for t in range(1, T):
            if t % args.stride or t + AS > T: continue
            win = [ov[t - 1], ov[t]]; gt = acts[t:t + AS, :6]
            eMSE.append(run_sampler(mse, regression_sampler, win) - gt)
            eS1.append(run_sampler(mip, mip_step1_only_sampler, win) - gt)
            eFULL.append(run_sampler(mip, mip_sampler, win) - gt)
            dists.append(dist(ov[t]))
    dists = np.array(dists); eMSE = np.array(eMSE); eS1 = np.array(eS1); eFULL = np.array(eFULL)
    np.savez("analysis/recovery/mip_decomp.npz", dist=dists, eMSE=eMSE, eS1=eS1, eFULL=eFULL)
    print(f"captured {len(dists)} states\n")
    print(f"{'bin':10} {'N':>6} {'|MSE|':>7} {'|MIP-step1|':>11} {'|MIP-full|':>10} | {'step2 pullback':>14}")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 1e9)]:
        m = (dists >= lo) & (dists < hi); n = int(m.sum())
        if n < 10:
            print(f"[{lo},{hi})    {n:>6}  -- too few"); continue
        bM = np.linalg.norm(eMSE[m, 0].mean(0)); b1 = np.linalg.norm(eS1[m, 0].mean(0))
        bF = np.linalg.norm(eFULL[m, 0].mean(0))
        pull = b1 - bF  # how much the 2nd step reduces the bias
        print(f"[{lo},{hi})    {n:>6} {bM:>7.3f} {b1:>11.3f} {bF:>10.3f} | {pull:>+14.3f}")
    print("\n|MIP-step1| vs |MSE|: are MIP & MSE one-step functions equally bad off-support?")
    print("|MIP-full| - |MIP-step1| (step2 pullback): how much the 2nd refinement step removes.")
    print("saved analysis/recovery/mip_decomp.npz")


if __name__ == "__main__":
    main()
