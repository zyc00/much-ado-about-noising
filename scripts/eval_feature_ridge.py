"""TASK 1: Feature-ridge operator test (two-view kernel hypothesis).
For a checkpoint, extract PENULTIMATE features (input to final 1x1 conv, 128x16=2048-d)
at a chosen view x=(t, u, s); freeze them; fit a ridge readout to the executed first
action (6-d) on clean train states; then fit the induced off-support operator G_phi on
the standard pairs protocol and compare with G_GT.
  --view t0|tau  --input zeros|tube|randn|scramble
Readouts: train-readout R2, operator R2, pred-cos vs G_GT, pos-block sym-eigs, negdef."""
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
POS = slice(44, 47)
H = 16


def read(path, nmax=None):
    # numeric demo_{i} order == replay-buffer episode order (A10/stt alignment)
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"
    n = min(nmax, len(h[g])) if nmax else len(h[g])
    out = []
    for i in range(n):
        o = h[f"{g}/demo_{i}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"{g}/demo_{i}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--loss", required=True)
    ap.add_argument("--view", required=True, choices=["t0", "tau"])
    ap.add_argument("--input", default="zeros", choices=["zeros", "tube", "randn", "scramble"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_puredart_full2ins_2000.hdf5")
    ap.add_argument("--train_demos", type=int, default=200); ap.add_argument("--test_n", type=int, default=400)
    ap.add_argument("--random_init", action="store_true",
                    help="EXP-A3: skip checkpoint loading; probe the UNTRAINED network (random-feature/architecture-prior control)")
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.clean)}", "network=chiunet",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    if not args.random_init:
        ag.load(args.ckpt, load_optimizer=False)
    ag.eval()
    dev = cfg.optimization.device; TAU = float(cfg.optimization.t_two_step)
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    A10 = ds.replay_buffer["action"][:]
    ends = np.asarray(ds.replay_buffer.episode_ends[:]); stt = np.concatenate([[0], ends[:-1]])

    # hook the input of the last 1x1 conv of final_conv
    feats = {}
    net = None
    for name, m in ag.flow_map_ema.named_modules():
        if name.endswith("final_conv"):
            net = m
    last = list(net.children())[-1]
    def hook(mod, inp):
        feats["x"] = inp[0].detach()
    last.register_forward_pre_hook(hook)

    Q = None
    if args.input == "scramble":
        q, _ = np.linalg.qr(np.random.RandomState(0).randn(10, 10))
        Q = torch.tensor(q, dtype=torch.float32, device=dev)

    def chunk_for(i, t0, rng):
        if args.input == "zeros":
            return torch.zeros((1, H, 10), device=dev)
        if args.input == "randn":
            return torch.tensor(rng.randn(1, H, 10), dtype=torch.float32, device=dev)
        seg = A10[stt[i] + t0: stt[i] + t0 + H]
        if len(seg) < H: return None
        c = torch.tensor(na.normalize(seg)[None], device=dev, dtype=torch.float32)
        if args.input == "tube":
            return c + 0.1 * torch.tensor(rng.randn(*c.shape), dtype=torch.float32, device=dev)
        return (c + 0.1 * torch.tensor(rng.randn(*c.shape), dtype=torch.float32, device=dev)) @ Q.T

    def phi(i, t0, win, rng):
        c = chunk_for(i, t0, rng)
        if c is None: return None
        x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
        tval = 0.0 if args.view == "t0" else TAU
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                ag.flow_map_ema.get_velocity(torch.full((1,), tval, device=dev), c, emb)
        return feats["x"].reshape(-1).cpu().numpy()   # (128*16,)

    rng = np.random.RandomState(0)
    clean = read(args.clean, args.train_demos)
    # ---- readout training set (clean, on-support) ----
    Xtr, Ytr = [], []
    for i in range(args.train_demos):
        ov, acts = clean[i]
        for t in range(1, len(acts) - H, 6):
            f = phi(i, t, [ov[t - 1], ov[t]], rng)
            if f is None: continue
            Xtr.append(f); Ytr.append(acts[t, :6])
    Xtr = np.stack(Xtr); Ytr = np.stack(Ytr)
    mu_f = Xtr.mean(0); Xtr = Xtr - mu_f
    mu_y = Ytr.mean(0); Yc = Ytr - mu_y   # intercept (cancels in operator differences)
    lam = 1e-3
    A = Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1])
    Wr = np.linalg.solve(A, Xtr.T @ Yc)
    r2_tr = 1 - np.sum((Xtr @ Wr - Yc) ** 2) / np.sum(Yc ** 2)

    # ---- pairs protocol ----
    anchors40 = clean[:40]
    cl = np.concatenate([ov for ov, _ in anchors40], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(anchors40)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    test = read(args.test, args.test_n)
    DZ, DR, GT = [], [], []
    nbc = {}
    for ov, acts in test:
        for t in range(1, len(acts) - H, 4):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (2.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)]); t0c = min(t0, len(anchors40[i0][1]) - H - 1)
            if int(idx) not in nbc:
                f0 = phi(i0, t0c, [anchors40[i0][0][max(t0 - 1, 0)], anchors40[i0][0][t0]], rng)
                nbc[int(idx)] = (f0 - mu_f) @ Wr
            fs = phi(i0, t0c, [ov[t - 1], ov[t]], rng)
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            DR.append((fs - mu_f) @ Wr - nbc[int(idx)])
            GT.append(acts[t, :6] - anchors40[i0][1][min(t0, len(anchors40[i0][1]) - 1), :6])
    DZ = np.stack(DZ); DR = np.stack(DR); GT = np.stack(GT)

    def ridge6(X, Y, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Y).T
    G = ridge6(DZ, DR); G_gt = ridge6(DZ, GT)
    Pa = DZ @ G.T; Pb = DZ @ G_gt.T
    pcos = float(np.mean(np.sum(Pa * Pb, 1) / (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
    P = DZ @ G.T
    r2_op = 1 - np.sum((DR - P) ** 2) / np.sum((DR - DR.mean(0)) ** 2)
    B = G[0:3, POS] * sig[POS][None, :]
    w = np.linalg.eigvalsh((B + B.T) / 2)
    print(f"FEATRIDGE {args.tag} view={args.view} input={args.input} N={len(DZ)} "
          f"trainR2={r2_tr:.3f} opR2={r2_op:.3f} cosGT={pcos:.2f} "
          f"poseig={np.round(w,3).tolist()} negdef={bool(np.all(w<0))}")


if __name__ == "__main__":
    main()
