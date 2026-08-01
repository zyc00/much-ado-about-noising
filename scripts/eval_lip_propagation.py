"""Quantitative Lipschitz-propagation check (note's eq.29 analog, empirical).
For each held-out test state s with clean-support 1-NN neighbor s0:
  drift(s)  = || a_hat(s) - a_hat(s0) ||   (executed 6-dim action, per policy)
  Delta_in  = || obs_window(s) - obs_window(s0) ||_F   in the network's NORMALIZED input
              space (same metric as the Jacobian probe)
Empirical gain per bin = mean drift / mean Delta_in, to be compared with the measured
Jacobian ||dA/dobs||_F from analysis/recovery/jacobian.npz. Checks:
  (1) does gain_MSE / gain_MIP track J_MSE / J_MIP (unit-free model-ratio consistency)?
  (2) fraction of pairs with drift > J_bin * Delta_in (bound-style violation rate;
      heuristic since J is measured at test states, not along the path).
Neighbor windows use the clean demo's own previous frame. No env."""
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
from mip.samplers import regression_sampler, mip_step1_only_sampler

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
        out.append(ov)
    h.close(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/dart_test_huge_full2ins.hdf5")
    ap.add_argument("--clean_n", type=int, default=40); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()
    cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", args.clean, "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", args.clean, "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; optim = cfg.optimization

    # clean cloud WITH (demo, t) bookkeeping so we can form the neighbor's own window
    clean = read(args.clean, args.clean_n)
    cl = np.concatenate(clean, 0)
    owner = np.concatenate([[ (i, t) for t in range(len(tr)) ] for i, tr in enumerate(clean)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)

    def window(tr, t):
        return [tr[max(t - 1, 0)], tr[t]]

    def predict(ag, sampler, win):
        w = no.normalize(np.stack(win)[None])
        x = torch.tensor(w, device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                an = sampler(optim, ag.flow_map_ema, ag.encoder_ema, torch.zeros((1, args.H, 10), device=dev), {"state": x})
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][:, :6][0], w[0]

    nb_cache = {}
    def neighbor_pred(idx):
        if idx not in nb_cache:
            di, t = owner[idx]
            win = window(clean[int(di)], int(t))
            aM, wn = predict(mse, regression_sampler, win)
            aP, _ = predict(mip, mip_step1_only_sampler, win)
            nb_cache[idx] = (aM, aP, wn)
        return nb_cache[idx]

    test = read(args.test)
    dists, dins, drM, drP = [], [], [], []
    for tr in test:
        T = len(tr)
        for t in range(1, T - AS):
            if t % args.stride: continue
            d, idx = tree.query((tr[t] - mu) / sig)
            aM, wS = predict(mse, regression_sampler, window(tr, t))
            aP, _ = predict(mip, mip_step1_only_sampler, window(tr, t))
            aM0, aP0, wN = neighbor_pred(int(idx))
            dists.append(float(d))
            dins.append(float(np.linalg.norm(wS - wN)))
            drM.append(float(np.linalg.norm(aM - aM0)))
            drP.append(float(np.linalg.norm(aP - aP0)))
    dists = np.array(dists); dins = np.array(dins); drM = np.array(drM); drP = np.array(drP)
    np.savez("analysis/recovery/lip_propagation.npz", dist=dists, din=dins, drM=drM, drP=drP)

    J = np.load("analysis/recovery/jacobian.npz")
    jd = J["dist"]; jM = J["jM"]; jP = J["jP"]
    print(f"captured {len(dists)} pairs.  drift = ||a(s)-a(s_NN)||, Din = normalized-input distance")
    print(f"{'bin':8} {'N':>6} {'Din':>6} | {'driftMSE':>8} {'gainMSE':>8} {'J_MSE':>6} | {'driftMIP':>8} {'gainMIP':>8} {'J_MIP':>6} | {'gain ratio':>10} {'J ratio':>8}")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)]:
        m = (dists >= lo) & (dists < hi); jm = (jd >= lo) & (jd < hi)
        if m.sum() < 20: continue
        Din = dins[m].mean()
        gM = drM[m].mean() / Din; gP = drP[m].mean() / Din
        JM = jM[jm].mean() if jm.sum() else float("nan"); JP = jP[jm].mean() if jm.sum() else float("nan")
        print(f"[{lo},{hi})  {int(m.sum()):>6} {Din:>6.2f} | {drM[m].mean():>8.3f} {gM:>8.3f} {JM:>6.2f} | "
              f"{drP[m].mean():>8.3f} {gP:>8.3f} {JP:>6.2f} | {gM/max(gP,1e-9):>10.2f} {JM/max(JP,1e-9):>8.2f}")
    # bound-style violation rate using bin-level J
    print("\nbound check drift <= J_bin * Din  (heuristic, J measured at test states):")
    for lo, hi in [(1, 2), (2, 3), (3, 4)]:
        m = (dists >= lo) & (dists < hi); jm = (jd >= lo) & (jd < hi)
        if m.sum() < 20 or jm.sum() == 0: continue
        vM = float(np.mean(drM[m] > jM[jm].mean() * dins[m]))
        vP = float(np.mean(drP[m] > jP[jm].mean() * dins[m]))
        print(f"  [{lo},{hi}): violation rate MSE {100*vM:.0f}%  MIP {100*vP:.0f}%")
    print("saved analysis/recovery/lip_propagation.npz")


if __name__ == "__main__":
    main()
