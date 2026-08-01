"""Does off-support Jacobian RANK matter because it determines whether the
policy can EXPRESS the corrective action?

Mechanism under test: the deployed map's response to a state displacement is
confined to the column space of J (rank r out of the action space). If the
action that would return the state toward the demonstration support lies
outside that span, the policy cannot produce it no matter what the state
says. Then recovery capability should scale with how much of the corrective
direction lies inside the span -- and rank is the quantity that governs it.

At off-support rollout states (d > 2 and d > 10) we measure, per arm:
  PR, and effective ranks k50/k90 of the deployed Jacobian (function space)
  cos_corr : cosine between the CORRECTIVE action direction (the action that
             would move the end effector toward the nearest support state)
             and its projection onto the top-k column space of J
  capt_k   : fraction of the corrective direction captured by the top-k
             left singular vectors, k = 1, 2, 4, 8
  align_out: cosine between the policy's ACTUAL output and the corrective
             direction (does it use the capability it has?)
Env: RM_ARMS, RM_N, RM_TAG. Prints RMECH lines.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

ARMS = [tuple(a.split(":")) for a in os.environ["RM_ARMS"].split(",")]
NJ = int(os.environ.get("RM_N", "80"))
TAG = os.environ.get("RM_TAG", "l2mp200v2")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
RW = np.concatenate([z[f"W{sd}"] for sd in seeds])
RD = np.concatenate([z[f"D{sd}"] for sd in seeds])
rng = np.random.RandomState(0)
SETS = {}
for lab, m in (("annulus", (RD >= 2) & (RD < 10)), ("far", RD >= 10)):
    idx = np.where(m)[0]
    SETS[lab] = RW[rng.choice(idx, min(NJ, len(idx)), replace=False)]
print(f"annulus {len(SETS['annulus'])} far {len(SETS['far'])}", flush=True)


def pr(ev):
    ev = np.clip(np.asarray(ev, dtype=np.float64), 0, None)
    return float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30))


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
    # support set (raw windows) for the corrective direction
    rb = ds.replay_buffer
    ends = rb.episode_ends[:]
    oe = rb["obs"]
    try:
        S_all = oe["state"][:]
    except (TypeError, IndexError, KeyError):
        S_all = oe[:]
    A_all = rb["action"][:]
    starts = np.concatenate([[0], ends[:-1]])
    SW, SA = [], []
    for e in range(len(ends)):
        s0, e0 = int(starts[e]), int(ends[e])
        if e0 - s0 < H + 3:
            continue
        S, A = S_all[s0:e0], A_all[s0:e0]
        for i in range(1, e0 - s0 - H, 3):
            SW.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
            SA.append(na.normalize(A[i:i + H]).reshape(-1))
    SW, SA = np.stack(SW), np.stack(SA)
    stree = cKDTree(SW)

    for lab, WS in SETS.items():
        prs, k50, k90, capt = [], [], [], {1: [], 2: [], 4: [], 8: []}
        align = []
        for w in WS:
            xn = no.normalize(w).reshape(-1)
            x = torch.tensor(xn, device=dev, dtype=torch.float32)

            def f(inp):
                a0 = torch.zeros((1, H, 10), device=dev)
                return sampler(cfg.optimization, ag.flow_map_ema,
                               ag.encoder_ema, a0,
                               {"state": inp.reshape(1, 2, 53)}).reshape(-1)

            J = torch.autograd.functional.jacobian(f, x, vectorize=True)
            J = J.reshape(-1, x.numel()).double()
            U, S_, _ = torch.linalg.svd(J, full_matrices=False)
            s2 = (S_ ** 2)
            prs.append(pr(s2.cpu().numpy()))
            c = torch.cumsum(s2, 0) / s2.sum()
            k50.append(int((c < 0.50).sum()) + 1)
            k90.append(int((c < 0.90).sum()) + 1)
            # corrective direction: nearest support state's action chunk minus
            # this policy's own output (what it WOULD have to change to act
            # like the demonstration at the nearest supported state)
            _, nn = stree.query(xn)
            with torch.no_grad():
                out = f(x)
            corr = torch.tensor(SA[nn], device=dev, dtype=torch.float64) - out.double()
            cn = corr / (corr.norm() + 1e-30)
            for k in capt:
                Uk = U[:, :k]
                capt[k].append(float((Uk.T @ cn).norm()))
            on = out.double() / (out.double().norm() + 1e-30)
            align.append(float(torch.dot(on, cn)))
        cs = " ".join(f"k{k}:{np.mean(v):.3f}" for k, v in capt.items())
        print(f"RMECH {name} {lab} n={len(WS)} PR {np.mean(prs):.3f}"
              f"+-{np.std(prs)/np.sqrt(len(prs)):.3f} k50 {np.mean(k50):.2f} "
              f"k90 {np.mean(k90):.2f} | captured[{cs}] | align_out "
              f"{np.mean(align):+.3f}", flush=True)
print("RMECH done", flush=True)
