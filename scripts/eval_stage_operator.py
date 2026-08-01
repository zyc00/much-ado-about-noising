"""EXP-5: STAGE_CONDITIONED_OPERATOR.
Partition standard [2,4) pairs by the ANCHOR's episode progress (t0/T) into 4 equal-count
bins (proxy for grasp/carry/align/insert). Per bin:
  - N, G_GT ridge R2 (5-fold), G_GT top svs
  - per model: response operator cos vs bin G_GT, poseig, rotsv, gain@bad1, rot@mseamp
Models given as --models "tag:ckpt:loss,..." (policy-mode responses f(0,0,s)).
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
POS = slice(44, 47)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="tag:ckpt:loss comma-separated")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_puredart_full2ins_2000.hdf5")
    ap.add_argument("--ref", default="analysis/recovery/badmode_ref.npz")
    ap.add_argument("--test_n", type=int, default=400)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")

    clean = read(args.clean, 40)
    cl = np.concatenate([ov for ov, _ in clean], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
    lens = {i: len(ov) for i, (ov, _) in enumerate(clean)}
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    test = read(args.test, args.test_n)

    # collect pairs with progress
    P = []   # (win_prev, win_cur, anc_i, anc_t, dz, gt, progress)
    for ov, acts in test:
        for t in range(1, len(acts) - H, 4):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (2.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)])
            prog = t0 / max(lens[i0] - 1, 1)
            P.append((ov[t - 1], ov[t], i0, t0, z - (cl[int(idx)] - mu) / sig,
                      acts[t, :6] - clean[i0][1][min(t0, len(clean[i0][1]) - 1), :6], prog))
    prog = np.array([p[6] for p in P])
    qs = np.quantile(prog, [0.25, 0.5, 0.75])
    binid = np.digitize(prog, qs)
    DZ = np.stack([p[4] for p in P]); GT = np.stack([p[5] for p in P])

    def ridge6(X, Y, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Y).T

    def r2cv(X, Y, k=5):
        rng = np.random.RandomState(0); idx = rng.permutation(len(X)); f = np.array_split(idx, k)
        sr = st = 0.0
        for i in range(k):
            te = f[i]; tr = np.concatenate([f[j] for j in range(k) if j != i])
            G = ridge6(X[tr], Y[tr]); Pd = X[te] @ G.T
            sr += np.sum((Y[te] - Pd) ** 2); st += np.sum((Y[te] - Y[tr].mean(0)) ** 2)
        return 1 - sr / st

    ref = np.load(args.ref)
    def posblk(Gx):
        Bx = Gx[0:3, POS] * ref["sig"][POS][None, :]
        return (Bx + Bx.T) / 2
    wm, Vm = np.linalg.eigh(posblk(ref["G_m"]))
    bad1 = Vm[:, np.argmax(wm)]
    vrot = np.linalg.svd(ref["G_m"][3:6, :] * ref["sig"][None, :])[2][0]

    G_gt_bins = {}
    for b in range(4):
        m = binid == b
        G = ridge6(DZ[m], GT[m])
        sv = np.linalg.svd(G, compute_uv=False)
        print(f"STAGEOP GT bin={b} progq=[{prog[m].min():.2f},{prog[m].max():.2f}] N={int(m.sum())} "
              f"R2cv={r2cv(DZ[m], GT[m]):.3f} sv={np.round(sv[:3],2).tolist()}", flush=True)
        G_gt_bins[b] = G

    for spec in args.models.split(","):
        tag, ckpt, loss = spec.split(":")
        with initialize_config_dir(version_base=None, config_dir=cfgdir):
            cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
                f"+task.dataset_path={os.path.abspath(args.clean)}", "network=chiunet",
                f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
        OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
        ds = make_dataset(cfg.task)
        ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
        dev = cfg.optimization.device; startk = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
        no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

        def respond(w0, w1):
            x = torch.tensor(no.normalize(np.stack([w0, w1])[None]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                with ag._inference_mode():
                    emb = ag.encoder_ema({"state": x}, None)
                    an = ag.flow_map_ema.get_velocity(torch.zeros(1, device=dev),
                                                      torch.zeros((1, H, 10), device=dev), emb)
            return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, startk:startk + AS])[0][0, :6]

        nb = {}
        DR = []
        for (w0, w1, i0, t0, dz, gt, pg) in P:
            key = (i0, t0)
            if key not in nb:
                nb[key] = respond(clean[i0][0][max(t0 - 1, 0)], clean[i0][0][t0])
            DR.append(respond(w0, w1) - nb[key])
        DR = np.stack(DR)
        for b in range(4):
            m = binid == b
            G = ridge6(DZ[m], DR[m])
            Pa = DZ[m] @ G.T; Pb = DZ[m] @ G_gt_bins[b].T
            pcos = float(np.mean(np.sum(Pa * Pb, 1) /
                                 (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
            B = G[0:3, POS] * sig[POS][None, :]
            w = np.linalg.eigvalsh((B + B.T) / 2)
            rsv = float(np.linalg.svd(G[3:6, :] * sig[None, :], compute_uv=False)[0])
            S = posblk(G)
            gb = float(bad1 @ S @ bad1)
            ra = float(np.linalg.norm((G[3:6, :] * ref["sig"][None, :]) @ vrot))
            print(f"STAGEOP {tag} bin={b} cosGT={pcos:.2f} poseig={np.round(w,3).tolist()} "
                  f"negdef={bool(np.all(w<0))} rotsv={rsv:.3f} gain@bad1={gb:+.4f} rot@mseamp={ra:.3f}", flush=True)


if __name__ == "__main__":
    main()
