"""EXP 1 (wrong-action chart test) + EXP 2 (J_s gating) on the standalone denoiser.

H (action-conditioned local chart): J_a g ~ 0 locally, but the STATE-response operator
S(a) = J_s g(a, s) depends on the base action; correct-phase action bases yield
S(a) ~ G_GT, wrong-phase/random bases degrade the corrective operator.

EXP1: on off-support pairs, run g(a_x, s) for action variants
  a0 (anchor, correct phase) | aM (MSE proposal at s) | aWP (same demo, phase-shifted
  +80 steps) | aRD (random demo, random time) | aZ (zeros chunk, off-tube)
metrics per variant: |bias| / mean|err| vs GT; drift cos vs GT-change; fitted operator
G_D^x: pred-cos vs G_GT, pos-block sym-eigs.

EXP2: at on-support anchors, J_s(a-base) for bases {a0, a0+0.3xi, aWP, aRD}:
  ||J_a||_F (flat part), relative change ||J_s(a_x)-J_s(a0)||/||J_s(a0)||, and subspace
  angle between J_s bases (gating part).  No env."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import h5py
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from scipy.linalg import subspace_angles
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
from mip.samplers import regression_sampler

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)
H = 16


def load(ckpt, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    if ckpt: ag.load(ckpt, load_optimizer=False)
    ag.eval()
    return cfg, ds, ag


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
    cfg, ds, den = load("logs/denoise_only_2k/models/model_latest.pt", "denoise_only")
    _, _, mse = load("logs/full_regression_2000/models/model_latest.pt", "regression")
    dev = cfg.optimization.device; start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; optim = cfg.optimization
    TAU = float(optim.t_two_step)
    A10 = ds.replay_buffer["action"][:]
    ends = np.asarray(ds.replay_buffer.episode_ends[:]); st = np.concatenate([[0], ends[:-1]])
    clean = read("data/tool_hang_full2ins_2000.hdf5", 40)
    cl = np.concatenate([ov for ov, _ in clean], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    rng = np.random.RandomState(0)

    def chunk_n(i, t0):
        seg = A10[st[i] + t0: st[i] + t0 + H]
        return None if len(seg) < H else torch.tensor(na.normalize(seg)[None], device=dev, dtype=torch.float32)

    def win_t(tr, t):
        return torch.tensor(no.normalize(np.stack([tr[max(t - 1, 0)], tr[t]])[None]), device=dev, dtype=torch.float32)

    def g_chunk(a_chunk, x_obs):
        with torch.no_grad():
            with den._inference_mode():
                emb = den.encoder_ema({"state": x_obs}, None)
                return den.flow_map_ema.get_velocity(torch.full((1,), TAU, device=dev), a_chunk, emb)

    def dec(an):
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :6]

    def msechunk(x):
        with torch.no_grad():
            with mse._inference_mode():
                return regression_sampler(optim, mse.flow_map_ema, mse.encoder_ema, torch.zeros((1, H, 10), device=dev), {"state": x})

    def variants(i0, t0c, xs):
        L = len(clean[i0][1]) - H - 1
        tw = (t0c + 80) % max(L, 1)
        if abs(tw - t0c) < 40: tw = (t0c + 120) % max(L, 1)
        j = rng.randint(0, 40)
        tj = rng.randint(1, len(clean[j][1]) - H - 1)
        return {
            "a0  (anchor/correct chart)": chunk_n(i0, t0c),
            "aM  (MSE proposal)": msechunk(xs),
            "aWP (same demo, +80 phase)": chunk_n(i0, tw),
            "aRD (random demo/time)": chunk_n(j, tj),
            "aZ  (zeros, off-tube)": torch.zeros((1, H, 10), device=dev),
        }

    # ---------------- EXP 1 ----------------
    test = read("data/dart_test_huge_full2ins.hdf5")
    pairs = []
    for ov, acts in test:
        for t in range(1, len(acts) - H, 6):
            z = (ov[t] - mu) / sig; d, idx = tree.query(z)
            if 2.0 <= d < 4.0:
                pairs.append((ov, acts, t, int(idx)))
        if len(pairs) >= 700: break
    print(f"EXP1 pairs: {len(pairs)}")
    E = {}; DR = {}; DZ = []; GTd = []
    first = True
    for ov, acts, t, idx in pairs:
        i0, t0 = map(int, owner[idx]); t0c = min(t0, len(clean[i0][1]) - H - 1)
        xs = win_t(ov, t); x0 = win_t(clean[i0][0], t0)
        V = variants(i0, t0c, xs)
        if any(v is None for v in V.values()): continue
        gt = acts[t, :6]; a_anchor = clean[i0][1][min(t0, len(clean[i0][1]) - 1), :6]
        DZ.append((ov[t] - mu) / sig - (cl[idx] - mu) / sig)
        GTd.append(gt - a_anchor)
        for k, a_x in V.items():
            y_s = dec(g_chunk(a_x, xs)); y_0 = dec(g_chunk(a_x, x0))
            E.setdefault(k, []).append(y_s - gt)
            DR.setdefault(k, []).append(y_s - y_0)
        first = False
    DZ = np.stack(DZ); GTd = np.stack(GTd)

    def ridge(X, Y, lam=1e-2):
        A = X.T @ X + lam * len(X) * np.eye(X.shape[1]); return np.linalg.solve(A, X.T @ Y).T
    G_gt = ridge(DZ, GTd)
    def pcos(Ga, Gb):
        Pa = DZ @ Ga.T; Pb = DZ @ Gb.T
        return float((np.sum(Pa * Pb, 1) / (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)).mean())
    print(f"\n{'action base':30} {'|bias|':>7} {'m|err|':>7} {'driftcos':>8} {'cos(G,G_GT)':>11} {'pos-eigs':>24}")
    for k in E:
        Ek = np.stack(E[k]); Dk = np.stack(DR[k])
        c = np.sum(Dk * GTd, 1) / (np.linalg.norm(Dk, axis=1) * np.linalg.norm(GTd, axis=1) + 1e-9)
        Gk = ridge(DZ, Dk)
        B = Gk[0:3, POS] * sig[POS][None, :]
        w = np.linalg.eigvalsh((B + B.T) / 2)
        print(f"{k:30} {np.linalg.norm(Ek.mean(0)):>7.3f} {np.linalg.norm(Ek,axis=1).mean():>7.3f} "
              f"{c.mean():>8.2f} {pcos(Gk, G_gt):>11.2f} {np.round(w,3)!s:>24}")

    # ---------------- EXP 2 ----------------
    print("\nEXP2: J_s gating by action base (40 on-support anchors)")
    def jac(a_chunk, x_obs):
        a = a_chunk.clone().requires_grad_(True); x = x_obs.clone().requires_grad_(True)
        with den._inference_mode():
            emb = den.encoder_ema({"state": x}, None)
            y = den.flow_map_ema.get_velocity(torch.full((1,), TAU, device=dev), a, emb)[0, start]
            Js = torch.zeros(10, x.numel(), device=dev); Ja = torch.zeros(10, a.numel(), device=dev)
            for i in range(10):
                gx, ga = torch.autograd.grad(y[i], [x, a], retain_graph=(i < 9))
                Js[i] = gx.reshape(-1); Ja[i] = ga.reshape(-1)
        return Js.cpu().numpy(), Ja.cpu().numpy()

    anchors = []
    while len(anchors) < 40:
        i = rng.randint(0, 40); t0 = rng.randint(2, len(clean[i][0]) - H - 2)
        c = chunk_n(i, t0)
        if c is not None: anchors.append((i, t0, c))
    rows = {k: {"dJs": [], "ang": [], "Ja": []} for k in ["a0", "a0+0.3xi", "aWP", "aRD"]}
    for i, t0, c in anchors:
        x0 = win_t(clean[i][0], t0)
        Js0, Ja0 = jac(c, x0)
        U0 = np.linalg.svd(Js0, full_matrices=False)[0][:, :3]
        L = len(clean[i][1]) - H - 1
        tw = (t0 + 80) % max(L, 1)
        j = rng.randint(0, 40); tj = rng.randint(1, len(clean[j][1]) - H - 1)
        bases = {"a0": c, "a0+0.3xi": c + 0.3 * torch.randn_like(c),
                 "aWP": chunk_n(i, tw), "aRD": chunk_n(j, tj)}
        for k, a_x in bases.items():
            if a_x is None: continue
            Js, Ja = jac(a_x, x0)
            rows[k]["dJs"].append(np.linalg.norm(Js - Js0) / (np.linalg.norm(Js0) + 1e-9))
            U = np.linalg.svd(Js, full_matrices=False)[0][:, :3]
            rows[k]["ang"].append(np.degrees(subspace_angles(U0, U)).mean())
            rows[k]["Ja"].append(np.linalg.norm(Ja))
    print(f"{'base':10} {'||dJ_s||/||J_s||':>15} {'subspace ang(deg)':>17} {'||J_a||_F':>9}")
    for k, v in rows.items():
        print(f"{k:10} {np.mean(v['dJs']):>15.2f} {np.mean(v['ang']):>17.1f} {np.mean(v['Ja']):>9.3f}")


if __name__ == "__main__":
    main()
