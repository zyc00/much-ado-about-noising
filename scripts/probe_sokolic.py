"""Sokolic et al. (1605.08254): generalization is bounded by the Jacobian
SPECTRAL norm in the neighbourhood of the training samples. Measured here on
the DEPLOYED map at TRAINING states, both norms, matched states/normalizer,
against each arm's measured generalization gap.
Env: SK_ARMS, SK_N. Prints SOKO lines."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

NJ = int(os.environ.get("SK_N", "48"))
ARMS = [tuple(a.split(":")) for a in os.environ["SK_ARMS"].split(",")]

def load(loss, dset):
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
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

for name, loss, ck, dset in ARMS:
    cfg, ds, ag = load(loss, dset)
    ag.load(ck, load_optimizer=False); ag.eval()
    sampler = get_sampler(loss)
    no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device
    H = int(cfg.task.horizon)
    rb = ds.replay_buffer; ends = rb.episode_ends[:]
    oe = rb["obs"]
    try: S_all = oe["state"][:]
    except (TypeError, IndexError, KeyError): S_all = oe[:]
    starts = np.concatenate([[0], ends[:-1]]); Wl = []
    for e in range(len(ends)):
        s0, e0 = int(starts[e]), int(ends[e])
        if e0 - s0 < H + 3: continue
        S = S_all[s0:e0]
        for i in range(1, e0 - s0 - H, 5):
            Wl.append(no.normalize(np.stack([S[i-1], S[i]])).reshape(-1))
    WN = np.stack(Wl); nwin = len(WN)
    sel = np.random.RandomState(0).choice(nwin, NJ, replace=False)
    s2s, sfs = [], []
    for i in sel:
        x = torch.tensor(WN[i], device=dev, dtype=torch.float32)
        def f(inp):
            a0 = torch.zeros((1, H, 10), device=dev)
            return sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                           a0, {"state": inp.reshape(1, 2, 53)}).reshape(-1)
        J = torch.autograd.functional.jacobian(f, x, vectorize=True)
        s = torch.linalg.svdvals(J.reshape(-1, x.numel()).double())
        s2s.append(float(s[0])); sfs.append(float((s**2).sum().sqrt()))
    print(f"SOKO {name} n_win {nwin} specnorm_mean {np.mean(s2s):.4f} "
          f"med {np.median(s2s):.4f} fro_mean {np.mean(sfs):.4f} "
          f"spec/sqrtN {np.mean(s2s)/np.sqrt(nwin):.6f}", flush=True)
print("SOKO done", flush=True)
