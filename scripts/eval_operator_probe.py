"""GENERIC operator probe for any trained model. Fits the model's off-support response
operator on the standard pairs protocol ([2,4) band, 40-demo clean anchors) and compares
with G_GT fit on the same pairs. Two modes:
  --mode denoiser : response = dec(g_t(a0_chunk, w_s)) - dec(g_t(a0_chunk, w_s0)),
                    a0_chunk = anchor clean chunk (optionally transformed for scramble),
                    g_t = flow_map at t = t_two_step
  --mode policy   : response = dec(step1(w_s)) - dec(step1(w_s0))   (f(0, zeros, s))
Readouts: linearity R2, pred-cos vs G_GT, pos-block sym-eigs, top SVs.
--scramble applies the FIXED orthogonal Q (seed 0, same as denoise_scramble training) to
the anchor chunk input. Writes results to stdout; save npz beside the ckpt."""
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
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g])[:nmax] if nmax else list(h[g])
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"{g}/{k}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--loss", required=True)
    ap.add_argument("--mode", required=True, choices=["denoiser", "policy"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--t_two_step", type=float, default=None)
    ap.add_argument("--scramble", action="store_true")
    ap.add_argument("--input", default="anchor", choices=["anchor","zeros","randn"], help="denoiser-mode action input (match the variant training input)")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/dart_test_huge_full2ins.hdf5")
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    ov_ = ["task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(args.clean)}",
           "network=chiunet", f"optimization.loss_type={args.loss}",
           "optimization.auto_resume=false", "log.wandb_mode=disabled"]
    if args.t_two_step is not None:
        ov_.append(f"optimization.t_two_step={args.t_two_step}")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=ov_)
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(args.ckpt, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    TAU = float(cfg.optimization.t_two_step)
    A10 = ds.replay_buffer["action"][:]
    ends = np.asarray(ds.replay_buffer.episode_ends[:]); stt = np.concatenate([[0], ends[:-1]])

    Q = None
    if args.scramble:
        rng_q = np.random.RandomState(0)
        q, _ = np.linalg.qr(rng_q.randn(10, 10))
        Q = torch.tensor(q, dtype=torch.float32, device=dev)

    clean = read(args.clean, 40)
    cl = np.concatenate([ov for ov, _ in clean], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)

    def chunk_n(i, t0):
        seg = A10[stt[i] + t0: stt[i] + t0 + H]
        if len(seg) < H: return None
        c = torch.tensor(na.normalize(seg)[None], device=dev, dtype=torch.float32)
        return (c @ Q.T) if Q is not None else c

    def win_t(f0, f1):
        return torch.tensor(no.normalize(np.stack([f0, f1])[None]), device=dev, dtype=torch.float32)

    def respond(a_chunk, x):
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                if args.mode == "denoiser":
                    an = ag.flow_map_ema.get_velocity(torch.full((1,), TAU, device=dev), a_chunk, emb)
                else:
                    an = ag.flow_map_ema.get_velocity(torch.zeros(1, device=dev), torch.zeros((1, H, 10), device=dev), emb)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :6]

    test = read(args.test)
    DZ, DR, GT = [], [], []
    nb = {}
    for ov, acts in test:
        for t in range(1, len(acts) - H, args.stride):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (2.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)]); t0c = min(t0, len(clean[i0][1]) - H - 1)
            c0 = chunk_n(i0, t0c)
            if c0 is None: continue
            if args.input == "zeros":
                c0 = torch.zeros_like(c0)
            elif args.input == "randn":
                g_r = torch.Generator(device=c0.device).manual_seed(int(idx))
                c0 = torch.randn(c0.shape, generator=g_r, device=c0.device)
            key = (int(idx), args.input)
            if key not in nb:
                nb[key] = respond(c0, win_t(clean[i0][0][max(t0 - 1, 0)], clean[i0][0][t0]))
            y0 = nb[key]
            ys = respond(c0, win_t(ov[t - 1], ov[t]))
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            DR.append(ys - y0)
            GT.append(acts[t, :6] - clean[i0][1][min(t0, len(clean[i0][1]) - 1), :6])
    DZ = np.stack(DZ); DR = np.stack(DR); GT = np.stack(GT)

    def ridge(X, Y, lam=1e-2):
        A = X.T @ X + lam * len(X) * np.eye(X.shape[1]); return np.linalg.solve(A, X.T @ Y).T
    def r2cv(X, Y, k=5):
        rng = np.random.RandomState(0); idx = rng.permutation(len(X)); f = np.array_split(idx, k)
        sr = st = 0.0
        for i in range(k):
            te = f[i]; tr = np.concatenate([f[j] for j in range(k) if j != i])
            G = ridge(X[tr], Y[tr]); P = X[te] @ G.T
            sr += np.sum((Y[te] - P) ** 2); st += np.sum((Y[te] - Y[tr].mean(0)) ** 2)
        return 1 - sr / st
    G = ridge(DZ, DR); G_gt = ridge(DZ, GT)
    Pa = DZ @ G.T; Pb = DZ @ G_gt.T
    pcos = float(np.mean(np.sum(Pa * Pb, 1) / (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
    B = G[0:3, POS] * sig[POS][None, :]
    w = np.linalg.eigvalsh((B + B.T) / 2)
    s = np.linalg.svd(G, compute_uv=False)
    print(f"OPPROBE {args.tag} mode={args.mode} N={len(DZ)} "
          f"R2={r2cv(DZ, DR):.3f} cosGT={pcos:.2f} poseig={np.round(w,3).tolist()} "
          f"sv={np.round(s[:3],2).tolist()} negdef={bool(np.all(w<0))}")
    np.savez(os.path.join(os.path.dirname(args.ckpt), f"opprobe_{args.tag}.npz"), DZ=DZ, DR=DR, GT=GT, G=G, G_gt=G_gt)


if __name__ == "__main__":
    main()
