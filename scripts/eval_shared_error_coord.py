"""Test the SHARED JOINT ERROR-COORDINATE hypothesis on the standalone denoiser:
model r(a,s) = H(P ds + Q da)  with HQ ~ -I (trained), and corrective obs-response iff
P couples to the same bottleneck (HP ~ -K).

T1 column-space sharing: principal angles between col(J_s) and col(J_a) of
   y = g(a0, s) (first executed step, 10-dim normalized) at anchors. Shared bottleneck
   -> angles small and both Jacobians low-rank (~3). Control: random-init same net.
T2 cancellation: for off-support pairs, solve da from J_a da = -(g(a0,s)-g(a0,s0)) and
   check the NONLINEAR residual ||g(a0+da, s) - g(a0, s0)|| / ||g(a0,s) - g(a0,s0)||.
   Shared bottleneck -> deep cancellation (<<1). Control: random-init net.
T3 pos-block proportionality: B_s = dy_pos/ds_eefpos (both window frames shifted) vs
   B_a = dy_pos/da1_pos. Feature story (phi = pose + scale*a) predicts B_s ~ c*B_a with
   high matrix-regression R^2. Report c, R^2, and sign structure.
All static, no env."""
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

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
EEFPOS = slice(44, 47)


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
    H = 16
    cfg, ds, den = load("logs/denoise_only_2k/models/model_latest.pt", "denoise_only")
    _, _, rnd = load(None, "denoise_only")   # random-init control (same arch)
    dev = cfg.optimization.device; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    TAU = float(cfg.optimization.t_two_step)
    A10 = ds.replay_buffer["action"][:]
    ends = np.asarray(ds.replay_buffer.episode_ends[:]); st = np.concatenate([[0], ends[:-1]])
    clean = read("data/tool_hang_full2ins_2000.hdf5", 40)
    cl = np.concatenate([ov for ov, _ in clean], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)

    def chunk_n(i, t0):
        seg = A10[st[i] + t0: st[i] + t0 + H]
        return None if len(seg) < H else torch.tensor(na.normalize(seg)[None], device=dev, dtype=torch.float32)

    def win_t(tr, t):
        return torch.tensor(no.normalize(np.stack([tr[max(t - 1, 0)], tr[t]])[None]), device=dev, dtype=torch.float32)

    def g_fn(ag, a_chunk, x_obs):
        emb = ag.encoder_ema({"state": x_obs}, None)
        out = ag.flow_map_ema.get_velocity(torch.full((1,), TAU, device=dev), a_chunk, emb)
        return out[0, start]   # first executed step, 10-dim normalized

    def jacobians(ag, a_chunk, x_obs):
        a = a_chunk.clone().requires_grad_(True); x = x_obs.clone().requires_grad_(True)
        with ag._inference_mode():
            y = g_fn(ag, a, x)
            Js = torch.zeros(10, x.numel(), device=dev); Ja = torch.zeros(10, a.numel(), device=dev)
            for i in range(10):
                gx, ga = torch.autograd.grad(y[i], [x, a], retain_graph=(i < 9))
                Js[i] = gx.reshape(-1); Ja[i] = ga.reshape(-1)
        return Js.cpu().numpy(), Ja.cpu().numpy(), y.detach().cpu().numpy()

    # anchors for T1/T3
    rng = np.random.RandomState(0)
    anchors = []
    while len(anchors) < 60:
        i = rng.randint(0, 40); t0 = rng.randint(2, len(clean[i][0]) - H - 2)
        c = chunk_n(i, t0)
        if c is not None: anchors.append((i, t0, c))

    def rank90(J):
        s = np.linalg.svd(J, compute_uv=False)
        return int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), 0.9) + 1)

    print("== T1: column-space sharing of J_s and J_a (principal angles, deg) ==")
    for name, ag in [("denoiser", den), ("random-init", rnd)]:
        angs, rs, ra = [], [], []
        for i, t0, c in anchors:
            Js, Ja, _ = jacobians(ag, c, win_t(clean[i][0], t0))
            r = 3
            Us = np.linalg.svd(Js, full_matrices=False)[0][:, :r]
            Ua = np.linalg.svd(Ja, full_matrices=False)[0][:, :r]
            angs.append(np.degrees(subspace_angles(Us, Ua)).mean())
            rs.append(rank90(Js)); ra.append(rank90(Ja))
        print(f"  {name:12}: mean principal angle {np.mean(angs):5.1f}±{np.std(angs):.1f} deg | rank90 J_s {np.mean(rs):.1f}  J_a {np.mean(ra):.1f}")

    print("\n== T3: pos-block proportionality  B_s ~ c * B_a ==")
    for name, ag in [("denoiser", den), ("random-init", rnd)]:
        cs, r2s = [], []
        for i, t0, c in anchors[:40]:
            Js, Ja, _ = jacobians(ag, c, win_t(clean[i][0], t0))
            # B_s: dy_pos / d eef_pos (sum both frames), de-normalized obs input scale
            Jsr = Js.reshape(10, 2, 53)
            B_s = (Jsr[0:3, 0, EEFPOS] + Jsr[0:3, 1, EEFPOS])
            Jar = Ja.reshape(10, H, 10)
            B_a = Jar[0:3, start, 0:3]   # response of exec-step pos to its own pos input dims
            x = B_a.reshape(-1); y = B_s.reshape(-1)
            cfit = float(x @ y / (x @ x + 1e-12))
            resid = y - cfit * x
            r2 = 1 - resid @ resid / (y @ y + 1e-12)
            cs.append(cfit); r2s.append(r2)
        print(f"  {name:12}: c = {np.mean(cs):+.3f}±{np.std(cs):.3f}   matrix-proportionality R2 = {np.mean(r2s):.2f}±{np.std(r2s):.2f}")

    print("\n== T2: cancellation of the obs-response through the action slot ==")
    test = read("data/dart_test_huge_full2ins.hdf5")
    pairs = []
    for ov, acts in test:
        for t in range(1, len(acts) - H, 8):
            z = (ov[t] - mu) / sig; d, idx = tree.query(z)
            if 2.0 <= d < 4.0:
                pairs.append((ov, t, int(idx)))
            if len(pairs) >= 150: break
        if len(pairs) >= 150: break
    for name, ag in [("denoiser", den), ("random-init", rnd)]:
        ratios = []
        for ov, t, idx in pairs:
            i0, t0 = map(int, owner[idx]); t0c = min(t0, len(clean[i0][1]) - H - 1)
            c0 = chunk_n(i0, t0c)
            if c0 is None: continue
            xs = win_t(ov, t); x0 = win_t(clean[i0][0], t0)
            with torch.no_grad():
                with ag._inference_mode():
                    y_s = g_fn(ag, c0, xs); y_0 = g_fn(ag, c0, x0)
            resp = (y_s - y_0)
            base = float(torch.norm(resp))
            if base < 1e-4: continue
            # solve J_a da = -resp (2 Gauss-Newton steps), da over full chunk
            a_cur = c0.clone()
            for _ in range(2):
                Js, Ja, _ = jacobians(ag, a_cur, xs)
                with torch.no_grad():
                    with ag._inference_mode():
                        y_cur = g_fn(ag, a_cur, xs)
                r = (y_cur - y_0).cpu().numpy()
                da, *_ = np.linalg.lstsq(Ja + 1e-3 * np.eye(10, Ja.shape[1])[:10], -r, rcond=None)
                a_cur = a_cur + torch.tensor(da.reshape(1, H, 10), device=dev, dtype=torch.float32)
            with torch.no_grad():
                with ag._inference_mode():
                    y_fin = g_fn(ag, a_cur, xs)
            ratios.append(float(torch.norm(y_fin - y_0)) / base)
        print(f"  {name:12}: cancellation residual ratio median {np.median(ratios):.2f}  p25/p75 {np.percentile(ratios,25):.2f}/{np.percentile(ratios,75):.2f}  (<<1 = shared bottleneck)")


if __name__ == "__main__":
    main()
