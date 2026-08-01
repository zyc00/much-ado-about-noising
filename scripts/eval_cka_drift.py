"""CHECK 2: kernel/feature drift from random init.
Extract penultimate features phi (t=0, zeros input) for RAND (untrained, fixed torch seed),
MSE, MIP on IDENTICAL inputs:
  (a) on-support clean states,
  (b) off-support pair differences dphi = phi(s) - phi(anchor)  (band [2,4)).
Report:
  - linear CKA(rand, model) and CKA(mse, mip) on both sets;
  - principal angles between 2-d servo subspaces U_servo (from cross-cov with G_GT dz);
  - alignment of each model's U_servo with the RANDOM U_servo.
Saves U_rand (+ mu_f/anchors meta) to logs/urand_ref.npz for trajectory probes.
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
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
H = 16


def read(path, nmax=None):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"
    n = min(nmax, len(h[g])) if nmax else len(h[g])
    out = []
    for i in range(n):
        o = h[f"{g}/demo_{i}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"{g}/demo_{i}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close(); return out


def cka(X, Y):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    a = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    b = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(a / b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_puredart_full2ins_2000.hdf5")
    ap.add_argument("--mse", default="logs/full_regression_2000_s1/models/model_latest.pt")
    ap.add_argument("--mip", default="logs/full_mip_2000_s1/models/model_latest.pt")
    ap.add_argument("--train_demos", type=int, default=60); ap.add_argument("--test_n", type=int, default=400)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.clean)}", "network=chiunet",
            "optimization.loss_type=mip", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    dev = cfg.optimization.device
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    def make_agent(ckpt):
        torch.manual_seed(0)   # fixed init so RAND is reproducible across scripts
        ag = TrainingAgent(cfg)
        if ckpt is not None:
            ag.load(ckpt, load_optimizer=False)
        ag.eval()
        feats = {}
        net = None
        for name, m in ag.flow_map_ema.named_modules():
            if name.endswith("final_conv"):
                net = m
        list(net.children())[-1].register_forward_pre_hook(
            lambda mod, inp: feats.__setitem__("x", inp[0].detach()))
        def phi(win):
            x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                with ag._inference_mode():
                    emb = ag.encoder_ema({"state": x}, None)
                    ag.flow_map_ema.get_velocity(torch.zeros(1, device=dev),
                                                 torch.zeros((1, H, 10), device=dev), emb)
            return feats["x"].reshape(-1).cpu().numpy()
        return phi

    models = {"RAND": make_agent(None), "MSE": make_agent(args.mse), "MIP": make_agent(args.mip)}

    clean = read(args.clean, max(args.train_demos, 40))
    # shared on-support state list
    wins_on = []
    for i, (ov, acts) in enumerate(clean[:args.train_demos]):
        for t in range(1, len(acts) - H, 10):
            wins_on.append([ov[t - 1], ov[t]])
    # shared off-support pairs
    anchors40 = clean[:40]
    cl = np.concatenate([ov for ov, _ in anchors40], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(anchors40)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    test = read(args.test, args.test_n)
    wins_s, wins_0, DZ, GT = [], [], [], []
    for ov, acts in test:
        for t in range(1, len(acts) - H, 4):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (2.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)])
            wins_0.append([anchors40[i0][0][max(t0 - 1, 0)], anchors40[i0][0][t0]])
            wins_s.append([ov[t - 1], ov[t]])
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            GT.append(acts[t, :6] - anchors40[i0][1][min(t0, len(anchors40[i0][1]) - 1), :6])
    DZ = np.stack(DZ); GT = np.stack(GT)

    def ridge6(X, Y, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Y).T
    G_gt = ridge6(DZ, GT)
    tgt = DZ @ G_gt.T

    PHI_on, DPHI, USERVO = {}, {}, {}
    for name, phi in models.items():
        PHI_on[name] = np.stack([phi(w) for w in wins_on]).astype(np.float64)
        Fs = np.stack([phi(w) for w in wins_s]).astype(np.float64)
        F0 = np.stack([phi(w) for w in wins_0]).astype(np.float64)
        DPHI[name] = Fs - F0
        M = DPHI[name].T @ tgt
        U, sv, _ = np.linalg.svd(M, full_matrices=False)
        USERVO[name] = U[:, :2]
        print(f"CKADRIFT {name} servo_svals={np.round(sv/sv[0],3).tolist()}", flush=True)

    for a, b in [("RAND", "MSE"), ("RAND", "MIP"), ("MSE", "MIP")]:
        c_on = cka(PHI_on[a], PHI_on[b])
        c_dp = cka(DPHI[a], DPHI[b])
        # servo-subspace principal angles (2-d)
        s = np.linalg.svd(USERVO[a].T @ USERVO[b], compute_uv=False)
        ang = np.degrees(np.arccos(np.clip(s, -1, 1)))
        # servo-restricted CKA: project dphi onto a's servo subspace before comparing
        c_srv = cka(DPHI[a] @ USERVO[a], DPHI[b] @ USERVO[a])
        print(f"CKADRIFT {a}-vs-{b} CKA_on={c_on:.3f} CKA_dphi={c_dp:.3f} "
              f"CKA_dphi_in_{a}servo={c_srv:.3f} servo_angles={np.round(ang,1).tolist()}", flush=True)

    np.savez("logs/urand_ref.npz", U_rand=USERVO["RAND"], G_gt=G_gt, sig=sig)
    print("saved logs/urand_ref.npz")


if __name__ == "__main__":
    main()
