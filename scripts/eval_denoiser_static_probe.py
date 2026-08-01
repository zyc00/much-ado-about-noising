"""Decisive static tests on the SEPARATELY-TRAINED denoising field (denoise_only_2k:
same architecture, trained ONLY on head2 (t=0.9, a*+0.1eps -> a*), clean-2k, NO shared
regression head). Three tests:

A) Action-slot contraction (its trained job): feed f_D(0.9, a*+sigma*eps, s) at clean
   (s, a*) pairs; contraction ratio ||out - a*|| / ||sigma*eps|| per sigma. Confirms it is a
   trained negative-feedback (full-removal) operator in the action slot; radius behavior.

B) Obs-slot response (THE transmission question): on the operator-fit pairs (off-support s,
   clean anchor s0, anchor action chunk a0), measure
      dA_D = decode(f_D(0.9, a0_chunk, s)) - decode(f_D(0.9, a0_chunk, s0))
   fit  dA_D ~ G_D dz  and compare with G_GT (prediction cosine, pos-block eigs, gains).
   If G_D ~ G_GT: the corrective law lives IN the denoising objective itself, no weight
   sharing needed for the law to EXIST (sharing only transports it into step1).

C) Composition static bias: a_comp(s) = decode(f_D(0.9, chunk_MSE(s), s)); bias(d) vs GT,
   compared with MSE alone and MIP-step1. Static readout of the no-weight-sharing ablation.
No env."""
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
POS = slice(44, 47)


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
    ap.add_argument("--clean_n", type=int, default=40); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()
    cfg, ds, den = load("logs/denoise_only_2k/models/model_latest.pt", args.clean, "denoise_only")
    _, _, mse = load("logs/full_regression_2000/models/model_latest.pt", args.clean, "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", args.clean, "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; optim = cfg.optimization
    TAU = float(optim.t_two_step)

    def obs_t(win):
        return torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)

    def denoise(ag, chunk_n, win):
        x = obs_t(win)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                t = torch.full((1,), TAU, device=dev)
                out = ag.flow_map_ema.get_velocity(t, chunk_n, emb)
        return out

    def onestep(ag, sampler, win):
        x = obs_t(win)
        with torch.no_grad():
            with ag._inference_mode():
                an = sampler(optim, ag.flow_map_ema, ag.encoder_ema, torch.zeros((1, args.H, 10), device=dev), {"state": x})
        return an

    def decode(an):
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :6]

    # normalized clean action chunks via replay buffer (converted 10-dim) + episode ends
    A10 = ds.replay_buffer["action"][:]
    ends = np.asarray(ds.replay_buffer.episode_ends[:]); starts = np.concatenate([[0], ends[:-1]])
    def chunk_n(demo_i, t0):
        s0 = starts[demo_i]
        seg = A10[s0 + t0: s0 + t0 + args.H]
        if len(seg) < args.H: return None
        return torch.tensor(na.normalize(seg)[None], device=dev, dtype=torch.float32)

    clean = read(args.clean, args.clean_n)
    cl = np.concatenate([ov for ov, _ in clean], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    def window(tr, t): return [tr[max(t - 1, 0)], tr[t]]

    # ---------- A) action-slot contraction ----------
    print("== A) denoiser action-slot contraction at clean (s, a*) ==")
    rng = np.random.RandomState(0)
    for sgm in [0.05, 0.1, 0.2, 0.5]:
        ratios = []
        for _ in range(300):
            i = rng.randint(0, args.clean_n); ovi, _ = clean[i]
            t0 = rng.randint(1, len(ovi) - args.H - 1)
            c = chunk_n(i, t0)
            if c is None: continue
            eps = torch.randn_like(c)
            out = denoise(den, c + sgm * eps, window(ovi, t0))
            ratios.append(float(torch.norm(out - c) / (sgm * torch.norm(eps) + 1e-9)))
        print(f"  sigma={sgm:<5} residual/perturbation = {np.mean(ratios):.3f}  (0=full removal, 1=no correction)")

    # ---------- B) + C) pairs on the off-support band ----------
    test = read(args.test)
    DZ, GT, DD, DA_D, E_MSE, E_CMP, E_MIP = [], [], [], [], [], [], []
    nb = {}
    for ov, acts in test:
        T = len(acts)
        for t in range(1, T - AS):
            if t % args.stride: continue
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (2.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)])
            c0 = chunk_n(i0, min(t0, len(clean[i0][1]) - args.H - 1))
            if c0 is None: continue
            if int(idx) not in nb:
                nb[int(idx)] = decode(denoise(den, c0, window(clean[i0][0], t0)))
            aD0 = nb[int(idx)]
            aD = decode(denoise(den, c0, window(ov, t)))
            # composition and references at s
            anM = onestep(mse, regression_sampler, window(ov, t))
            aM = decode(anM)
            aC = decode(denoise(den, anM, window(ov, t)))
            aP = decode(onestep(mip, mip_step1_only_sampler, window(ov, t)))
            gt = acts[t, :6]
            DZ.append(z - (cl[int(idx)] - mu) / sig); DD.append(d)
            GT.append(gt - clean[i0][1][min(t0, len(clean[i0][1]) - 1), :6])
            DA_D.append(aD - aD0)
            E_MSE.append(aM - gt); E_CMP.append(aC - gt); E_MIP.append(aP - gt)
    DZ = np.stack(DZ); GT = np.stack(GT); DA_D = np.stack(DA_D); DD = np.array(DD)
    E_MSE = np.stack(E_MSE); E_CMP = np.stack(E_CMP); E_MIP = np.stack(E_MIP)
    print(f"\npairs in [2,4): {len(DZ)}")

    def ridge(X, Y, lam=1e-2):
        A = X.T @ X + lam * len(X) * np.eye(X.shape[1]); return np.linalg.solve(A, X.T @ Y).T
    G_D = ridge(DZ, DA_D); G_gt = ridge(DZ, GT)
    def pcos(Ga, Gb):
        Pa = DZ @ Ga.T; Pb = DZ @ Gb.T
        return (np.sum(Pa * Pb, 1) / (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)).mean()
    def r2(G, X, Y):
        P = X @ G.T; return 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean(0)) ** 2)
    print("== B) standalone denoiser's OBS-slot response operator G_D ==")
    print(f"  linearity R2 of dA_D ~ G_D dz : {r2(G_D, DZ, DA_D):.3f}")
    print(f"  cos(G_D dz, G_GT dz)          : {pcos(G_D, G_gt):.2f}")
    B = G_D[0:3, POS] * sig[POS][None, :]
    w = np.linalg.eigvalsh((B + B.T) / 2)
    print(f"  G_D eef-pos block sym-eigs    : {np.round(w,3)}  ({'NEGATIVE feedback' if np.all(w<0) else 'not neg-def'})")
    print(f"  G_D pos-block:\n{np.round(B,3)}")

    print("\n== C) composition (frozen MSE -> standalone denoiser) static bias, [2,4) ==")
    for nm, E in [("MSE alone", E_MSE), ("MSE->denoiser composition", E_CMP), ("MIP-step1 (shared)", E_MIP)]:
        print(f"  {nm:28}: |bias| {np.linalg.norm(E.mean(0)):.3f}   mean|err| {np.linalg.norm(E,axis=1).mean():.3f}")
    np.savez("analysis/recovery/denoiser_static.npz", DZ=DZ, GT=GT, DA_D=DA_D, d=DD,
             E_MSE=E_MSE, E_CMP=E_CMP, E_MIP=E_MIP, G_D=G_D)
    print("saved analysis/recovery/denoiser_static.npz")


if __name__ == "__main__":
    main()
