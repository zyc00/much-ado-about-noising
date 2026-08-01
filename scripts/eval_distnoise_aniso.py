"""Inject ANISOTROPIC + temporally-correlated + biased noise into MIP, matched to
the residual error structure (Sigma_MSE - Sigma_MIP, mu_MSE - mu_MIP) per distance
bin (from err_vectors.npz). Tests whether aligning the FULL error structure (not
just RMSE) aligns the phenomenon (escape/SR) to MSE -- instead of the isotropic
white version that overshot (MIP->2.5%)."""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import robosuite
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS = "data/tool_hang_full2ins_2000.hdf5"
BINS = [(0, 1), (1, 2), (2, 4), (4, 1e9)]


def load(ckpt, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(DS)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    return cfg, ds, ag


def psd_chol(S):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0, None)
    return V @ np.diag(np.sqrt(w))   # L with L L^T = PSD(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--configs", default="0:0:0,1:0:1,1:0.9:1,1:0.9:0")  # scale:rho:meanon
    args = ap.parse_args()

    # error stats per bin from captured vectors
    E = np.load("analysis/recovery/err_vectors.npz")
    dist_e = E["dist"]; eM = E["eMSE"]; eP = E["eMIP"]  # (N,AS,6)
    MU, COV = {}, {}
    for b in BINS:
        m = (dist_e >= b[0]) & (dist_e < b[1])
        if m.sum() < 5:
            MU[b] = np.zeros(6); COV[b] = np.zeros((6, 6)); continue
        aM = eM[m].reshape(-1, 6); aP = eP[m].reshape(-1, 6)
        mu_res = aM.mean(0) - aP.mean(0)
        S_res = np.cov(aM, rowvar=False) - np.cov(aP, rowvar=False)
        MU[b] = mu_res; COV[b] = S_res
    def binof(d):
        for b in BINS:
            if b[0] <= d < b[1]: return b
        return BINS[-1]

    cfg, ds, mip = load("logs/full_mip_2000/models/model_latest.pt", "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def pred(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = mip.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    man = []
    for k in man_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); acts = np.clip(d["actions"][:], -1, 1); s0 = d["state0"][:]
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        man.append(ov(env._get_observations(force_update=True)))
        for t in range(len(acts)):
            o, _, _, _ = env.step(acts[t]); man.append(ov(o))
    man = np.array(man); mu, sig = man.mean(0), man.std(0) + 1e-6
    tree = cKDTree((man - mu) / sig)
    def ood(v): return float(tree.query((v - mu) / sig)[0])
    Lcache = {b: psd_chol(COV[b]) for b in BINS}

    def run(scale, rho, meanon):
        rng = np.random.RandomState(0); succ = 0; esc = 0; tot = 0
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
            steps = 0; asm = False; cur = ood(hist[-1]); eps = np.zeros(6)
            while steps < args.max_steps and not asm:
                for a in pred(hist):
                    b = binof(cur); L = Lcache[b]
                    white = L @ rng.randn(6)
                    eps = rho * eps + np.sqrt(max(1 - rho * rho, 0)) * white   # AR(1) colored
                    inj = np.sqrt(max(scale, 0)) * eps + (meanon * MU[b])      # cov scale on noise, + bias
                    ae = a.copy(); ae[:6] = np.clip(ae[:6] + inj, -1, 1)
                    o, _, _, _ = env.step(ae); steps += 1; v = ov(o); hist.append(v)
                    cur = ood(v); tot += 1; esc += int(cur > args.tau)
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            succ += int(asm)
        return 100 * succ / len(eval_keys), 100 * esc / max(tot, 1)

    print("ANISO-NOISE into MIP (residual Sigma_MSE-Sigma_MIP, bias mu_res, AR(1) rho)")
    print("  scale rho mean | MIP SR% | escape%")
    for c in args.configs.split(","):
        sc, rho, mo = c.split(":"); sc = float(sc); rho = float(rho); mo = float(mo)
        sr, e = run(sc, rho, mo)
        print(f"  {sc:<4} {rho:<4} {mo:<4} | {sr:5.1f}  | {e:5.1f}", flush=True)
    print("  (pure MIP ~92.5/12 ; pure MSE ~67.5/34 ; isotropic-white scale1 gave 2.5/67)")
    df.close()


if __name__ == "__main__":
    main()
