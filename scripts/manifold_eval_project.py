"""Closed-loop eval of  MSE-clean-2k  +  manifold-projection.
At each step MSE-2k predicts a normalized H=16 x 10 action chunk; we PROJECT it onto
the unconditional 20k-action manifold (annealed denoising with the trained denoiser),
then unnormalize + decode + execute act_steps. Tests whether a separable manifold
projector lifts MSE-2k toward MIP-2k. Sweeps the projection schedule strength.
Baselines (same 40 seeds, same settle): pure MSE-2k ~68%, pure MIP-2k ~92%."""
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
from manifold_train_denoiser import Denoiser

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS_DECODE = "data/tool_hang_full2ins_2000.hdf5"


def build_cfg(loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(DS_DECODE)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--denoiser", default="analysis/manifold/denoiser_uncond20k.pt")
    ap.add_argument("--scheds", default="off|0.05|0.1,0.05|0.3,0.15,0.05|0.5,0.3,0.15,0.05|1.0,0.5,0.25,0.1")
    ap.add_argument("--policy", default="mse", choices=["mse", "mip"])
    ap.add_argument("--alpha", type=float, default=1.0, help="partial projection step: a += alpha*(D(a)-a)")
    args = ap.parse_args()
    dev = "cuda"

    base_cfg = build_cfg("regression")
    ds = make_dataset(base_cfg.task)
    AS = base_cfg.task.act_steps; start = base_cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    if args.policy == "mse":
        mse = TrainingAgent(build_cfg("regression")); mse.load("logs/full_regression_2000/models/model_latest.pt", load_optimizer=False); mse.eval()
    else:
        mse = TrainingAgent(build_cfg("mip")); mse.load("logs/full_mip_2000/models/model_latest.pt", load_optimizer=False); mse.eval()

    ck = torch.load(args.denoiser, weights_only=False)
    D = Denoiser(ck["dim"]).to(dev); D.load_state_dict(ck["state_dict"]); D.eval()
    Hm = ck["H"]
    print(f"denoiser dim={ck['dim']} H={Hm} smin={ck['smin']} smax={ck['smax']}")

    def project(an, sched):  # an: (1,H,10) normalized -> projected (1,H,10)
        if sched is None:
            return an
        x = torch.tensor(an.reshape(1, -1), device=dev, dtype=torch.float32)
        with torch.no_grad():
            for sg in sched:
                step = D(x, torch.full((1, 1), sg, device=dev))
                x = x + args.alpha * (step - x)
        return x.cpu().numpy().reshape(an.shape)

    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def predict(hist, sched):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=mse.config.optimization.device, dtype=torch.float32)}
        with torch.no_grad():
            an = mse.sample(act_0=torch.randn((1, args.H, 10), device=mse.config.optimization.device), obs=ot, use_ema=True)
        an = an.detach().cpu().numpy()
        an = project(an, sched)
        return ds.undo_transform_action(na.unnormalize(an)[:, start:start + AS])[0]

    def rollout(sched):
        succ = 0
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; asm = False
            while steps < args.max_steps and not asm:
                for a in predict(hist, sched):
                    o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            succ += int(asm)
        return succ

    n = len(eval_keys)
    print(f"\nMSE-2k + manifold-projection  (n={n})  [ref: pure MSE-2k~68%, MIP-2k~92%]")
    for tok in args.scheds.split("|"):
        sched = None if tok == "off" else [float(x) for x in tok.split(",")]
        sr = rollout(sched)
        label = "pure MSE (no projection)" if sched is None else f"proj sched {sched}"
        print(f"  {label:40s} : {sr:2d}/{n} = {100*sr/n:.0f}%", flush=True)
    df.close()


if __name__ == "__main__":
    main()
