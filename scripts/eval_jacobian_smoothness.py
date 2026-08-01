"""What PROPERTY differs between the MSE and MIP-step1 1-step functions? Measure the
input sensitivity (Jacobian) of the predicted action w.r.t. the obs input, for both
(identical inference form: get_velocity(0, zeros, enc(obs))), binned by OOD distance.
  ||d action / d obs||_F  -- high gain = explosive extrapolation, low gain = smooth.
Hypothesis: off-support, MSE has a much larger Jacobian (sensitive -> escapes), MIP-step1
smaller (smooth -> controlled). Tests 'the trained property is obs-smoothness/low-gain'.
No env. Uses held-out DART test states + the clean-support distance ruler."""
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
    ap.add_argument("--clean_n", type=int, default=40); ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()
    cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", args.clean, "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", args.clean, "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; optim = cfg.optimization

    clean = read(args.clean, args.clean_n); cl = np.concatenate(clean, 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6; tree = cKDTree((cl - mu) / sig)
    def dist(v): return float(tree.query((v - mu) / sig)[0])
    test = read(args.test)

    def jac_norm(ag, sampler, win):
        # d (first executed action, 6-dim, normalized->executed) / d (obs window, normalized)
        w = no.normalize(np.stack(win)[None])           # (1,2,53)
        x = torch.tensor(w, device=dev, dtype=torch.float32, requires_grad=True)
        with ag._inference_mode():
            an = sampler(optim, ag.flow_map_ema, ag.encoder_ema, torch.zeros((1, args.H, 10), device=dev), {"state": x})
        a_exec = an[:, start, :6].reshape(-1)            # 6-dim normalized executed action (first step)
        J = torch.zeros((6, x.numel()), device=dev)
        for i in range(6):
            g, = torch.autograd.grad(a_exec[i], x, retain_graph=(i < 5))
            J[i] = g.reshape(-1)
        return float(torch.linalg.norm(J))              # Frobenius norm

    dists, jM, jP = [], [], []
    for ov in test:
        T = len(ov)
        for t in range(1, T - AS):
            if t % args.stride: continue
            win = [ov[t - 1], ov[t]]
            jM.append(jac_norm(mse, regression_sampler, win))
            jP.append(jac_norm(mip, mip_step1_only_sampler, win))
            dists.append(dist(ov[t]))
    dists = np.array(dists); jM = np.array(jM); jP = np.array(jP)
    print(f"captured {len(dists)} states.  ||d action / d obs||_F  (input-sensitivity / gain)\n")
    print(f"{'bin':10} {'N':>6} {'MSE gain':>9} {'MIP-s1 gain':>12} {'MSE/MIP':>8}")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 1e9)]:
        m = (dists >= lo) & (dists < hi); n = int(m.sum())
        if n < 5:
            print(f"[{lo},{hi})    {n:>6}  -- few"); continue
        a = jM[m].mean(); b = jP[m].mean()
        print(f"[{lo},{hi})    {n:>6} {a:>9.2f} {b:>12.2f} {a/max(b,1e-9):>8.2f}")
    np.savez("analysis/recovery/jacobian.npz", dist=dists, jM=jM, jP=jP)
    print("\nsaved analysis/recovery/jacobian.npz")


if __name__ == "__main__":
    main()
