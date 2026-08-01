"""Can the three on-manifold theories be made to separate the objectives?

Three things a state-averaged mean error hides, each a candidate for the
missing quantity:

P1 TAIL. Closed-loop failure is driven by the WORST chunks, not the average
   one. Report p50/p90/p95/p99/max of the per-state insertion-phase error.

P2 BIAS vs NOISE. A chunk is executed 8 steps open loop, so what displaces
   the robot is the SUM of the per-step errors. Decompose each chunk's
   position error into the systematic part (the mean over the 8 steps, which
   accumulates) and the zero-mean part (which partly cancels):
     bias_mm  = || mean_t e_t ||        * 50 mm
     noise_mm = mean_t || e_t - mean ||  * 50 mm
   Compounding theory says bias, not total error, should track SR.

P3 MARGIN-WEIGHTED SOKOLIC. Their bound is margin / ||J||_2 and our margin
   is state dependent (tight at insertion, loose in transit). Report ||J||_2
   restricted to insertion-phase states, and the ratio error/||J||_2.

Env: OR_ARMS, OR_N, OR_NJ. Prints ORES / OPAIR lines.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

ARMS = [tuple(a.split(":")) for a in os.environ["OR_ARMS"].split(",")]
NST = int(os.environ.get("OR_N", "4000"))
NJ = int(os.environ.get("OR_NJ", "40"))
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
MM = 50.0

h = h5py.File(HELD, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2600]
rng = np.random.RandomState(0)
W, Y, Z = [], [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"]).astype(np.float32)
    P = np.asarray(o["robot0_eef_pos"]).astype(np.float32)
    if len(S) < 20:
        continue
    for i in range(1, len(S) - 9, 3):
        W.append(np.stack([S[i - 1], S[i]]))
        Y.append(A[i:i + 8])
        Z.append(P[i, 2])
h.close()
W, Y, Z = np.stack(W), np.stack(Y), np.array(Z)
sel = rng.choice(len(W), min(NST, len(W)), replace=False)
W, Y, Z = W[sel], Y[sel], Z[sel]
INS = Z < 0.86
ins_idx = np.where(INS)[0]
print(f"states {len(W)} insertion {int(INS.sum())}", flush=True)

STORE = {}
for name, loss, ck, dset in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false",
            "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    H = int(cfg.task.horizon)
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device
    out = []
    for i in range(0, len(W), 256):
        xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 256]]),
                          device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), H, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": xb})
        out.append(np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, 1:9])))
    P = np.concatenate(out)
    E = (P - Y)[:, :, 0:3]                       # (N, 8, 3) position error
    tot = np.linalg.norm(E, axis=2).mean(1) * MM
    bias = np.linalg.norm(E.mean(1), axis=1) * MM
    noise = np.linalg.norm(E - E.mean(1, keepdims=True), axis=2).mean(1) * MM
    STORE[name] = dict(tot=tot, bias=bias, noise=noise)
    q = lambda v, p: float(np.percentile(v, p))
    print(f"ORES {name} insert_tot_mm p50 {q(tot[INS],50):.3f} p90 "
          f"{q(tot[INS],90):.3f} p95 {q(tot[INS],95):.3f} p99 "
          f"{q(tot[INS],99):.3f} max {tot[INS].max():.2f} mean "
          f"{tot[INS].mean():.3f}", flush=True)
    print(f"ORES {name} insert_BIAS_mm mean {bias[INS].mean():.3f} p90 "
          f"{q(bias[INS],90):.3f} p99 {q(bias[INS],99):.3f} | NOISE_mm mean "
          f"{noise[INS].mean():.3f} p90 {q(noise[INS],90):.3f} | "
          f"bias/noise {bias[INS].mean()/max(noise[INS].mean(),1e-9):.2f}",
          flush=True)
    # P3: spectral norm restricted to insertion-phase states
    s2 = []
    for i in rng.choice(ins_idx, min(NJ, len(ins_idx)), replace=False):
        x = torch.tensor(no.normalize(W[i]).reshape(-1), device=dev,
                         dtype=torch.float32)

        def f(inp):
            a0 = torch.zeros((1, H, 10), device=dev)
            return sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                           a0, {"state": inp.reshape(1, 2, 53)}).reshape(-1)

        J = torch.autograd.functional.jacobian(f, x, vectorize=True)
        s2.append(float(torch.linalg.svdvals(
            J.reshape(-1, x.numel()).double())[0]))
    print(f"ORES {name} insert_specnorm mean {np.mean(s2):.4f} med "
          f"{np.median(s2):.4f} | err/spec {tot[INS].mean()/np.mean(s2):.4f}",
          flush=True)

base = ARMS[0][0]
for name in STORE:
    if name == base:
        continue
    for key in ("tot", "bias", "noise"):
        a, b = STORE[base][key][INS], STORE[name][key][INS]
        d = b - a
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"OPAIR {name}-vs-{base} insert {key} {d.mean():+.4f}mm "
              f"+-{se:.4f} (t={d.mean()/se:+.1f})", flush=True)
print("ORES done", flush=True)
