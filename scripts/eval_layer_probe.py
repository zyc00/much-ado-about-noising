"""EXP2: layerwise feature-ridge probe. Where along the network does the
servo-compatible state-response appear?
Hooks, per forward at view (t, zeros, s):
  enc   : obs-encoder output (encoder_ema)
  cond  : global_cond_encoder output
  down0/1/2 : output of last module of each down stage
  mid   : mids[-1] output
  up0/1 : output of last module of each up stage
  pen   : input of the final 1x1 conv (penultimate; = existing FEATRIDGE probe)
Per layer: ridge readout (intercept, lam=1e-3) from frozen features to the first executed
6-d action on clean states; induced operator on the standard pairs protocol ([2,4) band).
Prints: LAYERPROBE tag layer dim trainR2 opR2 cosGT poseig negdef
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
    ap.add_argument("--ckpt", required=True); ap.add_argument("--loss", required=True)
    ap.add_argument("--view", required=True, choices=["t0", "tau"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_puredart_full2ins_2000.hdf5")
    ap.add_argument("--train_demos", type=int, default=200); ap.add_argument("--test_n", type=int, default=400)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.clean)}", "network=chiunet",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(args.ckpt, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; TAU = float(cfg.optimization.t_two_step)
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    net = ag.flow_map_ema
    core = None
    for nm, m in net.named_modules():
        if nm.endswith("final_conv"):
            core = m
    # locate the actual UNet holding downs/ups/mids
    unet = None
    for nm, m in net.named_modules():
        if hasattr(m, "downs") and hasattr(m, "ups") and hasattr(m, "mids"):
            unet = m
            break
    feats = {}
    def mk(name):
        def hk(mod, inp, out=None):
            x = out if out is not None else inp[0]
            if isinstance(x, tuple): x = x[0]
            feats[name] = x.detach()
        return hk
    unet.global_cond_encoder.register_forward_hook(mk("cond"))
    for i, stage in enumerate(unet.downs):
        list(stage.children())[-1].register_forward_hook(mk(f"down{i}"))
    unet.mids[-1].register_forward_hook(mk("mid"))
    for i, stage in enumerate(unet.ups):
        list(stage.children())[-1].register_forward_hook(mk(f"up{i}"))
    list(core.children())[-1].register_forward_pre_hook(
        lambda mod, inp: feats.__setitem__("pen", inp[0].detach()))

    LAYERS = ["enc", "cond"] + [f"down{i}" for i in range(len(unet.downs))] + ["mid"] + \
             [f"up{i}" for i in range(len(unet.ups))] + ["pen"]

    def phi_all(win):
        x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
        tval = 0.0 if args.view == "t0" else TAU
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                net.get_velocity(torch.full((1,), tval, device=dev),
                                 torch.zeros((1, H, 10), device=dev), emb)
        out = {"enc": emb.reshape(-1).cpu().numpy()}
        for k in LAYERS:
            if k == "enc": continue
            out[k] = feats[k].reshape(-1).float().cpu().numpy()
        return out

    rngstate = np.random.RandomState(0)
    clean = read(args.clean, args.train_demos)
    Xtr = {k: [] for k in LAYERS}; Ytr = []
    for i, (ov, acts) in enumerate(clean):
        for t in range(1, len(acts) - H, 6):
            fa = phi_all([ov[t - 1], ov[t]])
            for k in LAYERS: Xtr[k].append(fa[k])
            Ytr.append(acts[t, :6])
    Ytr = np.stack(Ytr).astype(np.float64)

    anchors40 = clean[:40]
    cl = np.concatenate([ov for ov, _ in anchors40], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(anchors40)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    test = read(args.test, args.test_n)
    Fs = {k: [] for k in LAYERS}; F0 = {k: [] for k in LAYERS}
    DZ, GT = [], []
    f0cache = {}
    for ov, acts in test:
        for t in range(1, len(acts) - H, 4):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (2.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)])
            if int(idx) not in f0cache:
                f0cache[int(idx)] = phi_all([anchors40[i0][0][max(t0 - 1, 0)], anchors40[i0][0][t0]])
            fa0 = f0cache[int(idx)]
            fas = phi_all([ov[t - 1], ov[t]])
            for k in LAYERS:
                F0[k].append(fa0[k]); Fs[k].append(fas[k])
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            GT.append(acts[t, :6] - anchors40[i0][1][min(t0, len(anchors40[i0][1]) - 1), :6])
    DZ = np.stack(DZ); GT = np.stack(GT)

    def ridge6(X, Y, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Y).T
    G_gt = ridge6(DZ, GT)
    for k in LAYERS:
        X = np.stack(Xtr[k]).astype(np.float64)
        mu_f = X.mean(0); Xc = X - mu_f
        mu_y = Ytr.mean(0); Yc = Ytr - mu_y
        lam = 1e-3
        W = np.linalg.solve(Xc.T @ Xc + lam * len(Xc) * np.eye(Xc.shape[1]), Xc.T @ Yc)
        r2tr = 1 - np.sum((Xc @ W - Yc) ** 2) / np.sum(Yc ** 2)
        Ps = (np.stack(Fs[k]) - mu_f) @ W; P0 = (np.stack(F0[k]) - mu_f) @ W
        DR = Ps - P0
        G = ridge6(DZ, DR)
        Pa = DZ @ G.T; Pb = DZ @ G_gt.T
        pcos = float(np.mean(np.sum(Pa * Pb, 1) /
                             (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
        r2op = 1 - np.sum((DR - Pa) ** 2) / np.sum((DR - DR.mean(0)) ** 2)
        B = G[0:3, POS] * sig[POS][None, :]
        w = np.linalg.eigvalsh((B + B.T) / 2)
        print(f"LAYERPROBE {args.tag} layer={k} dim={X.shape[1]} trainR2={r2tr:.3f} "
              f"opR2={r2op:.3f} cosGT={pcos:.2f} poseig={np.round(w,3).tolist()} "
              f"negdef={bool(np.all(w<0))}", flush=True)


if __name__ == "__main__":
    main()
