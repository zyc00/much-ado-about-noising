"""Final theory probes.

T-A (Sokolic, where the margin is tight): the bound is margin/||J||_2 and our
margin is smallest at the insertion phase, so the theory's prediction is about
||J||_2 THERE. High-N measurement of the insertion-phase gain distribution,
spectral and Frobenius, mean / median / p90, plus the same at transit for
contrast.

T-C (Rahaman, off the manifold): the on-manifold spectral test found every arm
spectrally complete. The pre-registered falsifier is whether the fine
structure survives OFF the manifold. Along captured off-support rollout states
(no GT available) we measure the output's high-band power FRACTION and total
power, and compare with the same policy's on-manifold values: an arm that
keeps its spectral shape off-support retains fine structure, one whose
spectrum flattens or explodes does not.

Env: FT_ARMS, FT_N, FT_TAG. Prints FTH lines.
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

ARMS = [tuple(a.split(":")) for a in os.environ["FT_ARMS"].split(",")]
NJ = int(os.environ.get("FT_N", "120"))
TAG = os.environ.get("FT_TAG", "l2mp200v2")
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

# ---- held-out states, split by phase --------------------------------------
h = h5py.File(HELD, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2400]
W, Z = [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    P = np.asarray(o["robot0_eef_pos"]).astype(np.float32)
    if len(S) < 20:
        continue
    for i in range(1, len(S) - 9, 4):
        W.append(np.stack([S[i - 1], S[i]]))
        Z.append(P[i, 2])
h.close()
W, Z = np.stack(W), np.array(Z)
rng = np.random.RandomState(0)
ins = np.where(Z < 0.86)[0]
tra = np.where((Z >= 0.86) & (Z < 1.02))[0]
INS = W[rng.choice(ins, min(NJ, len(ins)), replace=False)]
TRA = W[rng.choice(tra, min(NJ, len(tra)), replace=False)]

# ---- off-support rollout states -------------------------------------------
z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
RW = np.concatenate([z[f"W{sd}"] for sd in seeds])
RD = np.concatenate([z[f"D{sd}"] for sd in seeds])
ON = RW[rng.choice(np.where(RD < 2)[0], 600, replace=False)]
OFF = RW[rng.choice(np.where(RD > 10)[0], 600, replace=False)]
print(f"insert {len(INS)} transit {len(TRA)} | on {len(ON)} off {len(OFF)}",
      flush=True)


def hifrac(P):
    """fraction of output power above f=0.2 along the state sequence"""
    X = np.fft.rfft(P - P.mean(0), axis=0)
    p = (np.abs(X) ** 2).sum(1)
    f = np.fft.rfftfreq(len(P))
    return float(p[f >= 0.2].sum() / (p.sum() + 1e-30)), float(p.sum())


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

    def gains(WS):
        s2, sf = [], []
        for w in WS:
            x = torch.tensor(no.normalize(w).reshape(-1), device=dev,
                             dtype=torch.float32)

            def f(inp):
                a0 = torch.zeros((1, H, 10), device=dev)
                return sampler(cfg.optimization, ag.flow_map_ema,
                               ag.encoder_ema, a0,
                               {"state": inp.reshape(1, 2, 53)}).reshape(-1)

            J = torch.autograd.functional.jacobian(f, x, vectorize=True)
            s = torch.linalg.svdvals(J.reshape(-1, x.numel()).double())
            s2.append(float(s[0]))
            sf.append(float((s ** 2).sum().sqrt()))
        return np.array(s2), np.array(sf)

    for lab, WS in (("insert", INS), ("transit", TRA)):
        a, b = gains(WS)
        print(f"FTH {name} gain_{lab} spec_mean {a.mean():.3f} med "
              f"{np.median(a):.3f} p90 {np.percentile(a,90):.3f} | fro_mean "
              f"{b.mean():.3f} med {np.median(b):.3f}", flush=True)

    def outs(WS):
        o = []
        for i in range(0, len(WS), 256):
            xb = torch.tensor(np.stack([no.normalize(w) for w in WS[i:i + 256]]),
                              device=dev, dtype=torch.float32)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
            o.append(np.asarray(ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, 1:2]))[:, 0, 0:3])
        return np.concatenate(o)

    hon, pon = hifrac(outs(ON))
    hof, pof = hifrac(outs(OFF))
    print(f"FTH {name} spectrum on hi_frac {hon:.4f} pow {pon:.2f} | off "
          f"hi_frac {hof:.4f} pow {pof:.2f} | shape_change "
          f"{hof/max(hon,1e-9):.3f} pow_ratio {pof/max(pon,1e-9):.2f}",
          flush=True)
print("FTH done", flush=True)
