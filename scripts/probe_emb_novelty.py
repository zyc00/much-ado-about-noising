"""Does embedding rank determine whether OFF-MANIFOLD-NESS IS VISIBLE to the
decoder? — the mechanistic link between the embedding-side and Jacobian-side
measurements.

Hypothesis (the null-space-swallow mechanism): a low-rank embedding maps
off-support states INTO the populated on-support embedding region (the huge
null space swallows the displacement), so the decoder sees a normal-looking
code and emits full-power actions computed from the wrong chart; a
higher-rank embedding makes off-support states EMBEDDING OUTLIERS, and the
decoder (which never saw that region) attenuates. Predictions:
  P1 embedding novelty ratio (far/on) is SMALLEST for L2-200 and orders ~SR
  P2 per-state embedding novelty correlates with per-state output
     attenuation (the r=-0.844 off/on power result becomes a per-state law)
All arms are probed at the SAME physical states (the l2mp200v2 capture), so
input-space novelty is identical across arms by construction — any
difference is the encoder's.

Per arm: fit = 400 on-support states; probe sets = 200 held-out on-support,
300 annulus (2<=d<10), 600 far (d>=10). novelty(x) = mean distance to the
k=5 nearest fit embeddings, normalized by the median self-novelty of the
held-out on-support probes (scale-free). Output power = mean |pos action|
of the predicted chunk. Prints ENOV lines.
Env: EN_ARMS "name:loss:ckpt:dataset".
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

ARMS = [tuple(a.split(":")) for a in os.environ["EN_ARMS"].split(",")]
TAG = os.environ.get("EN_TAG", "l2mp200v2")

z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
RW = np.concatenate([z[f"W{sd}"] for sd in seeds])
RD = np.concatenate([z[f"D{sd}"] for sd in seeds])
rng = np.random.RandomState(0)
on_idx = np.where(RD < 2)[0]
on_sel = rng.choice(on_idx, 600, replace=False)
FIT, ONP = RW[on_sel[:400]], RW[on_sel[400:]]
ANN = RW[rng.choice(np.where((RD >= 2) & (RD < 10))[0], 300, replace=False)]
FAR = RW[rng.choice(np.where(RD >= 10)[0], 600, replace=False)]
print(f"fit 400 | on-probe {len(ONP)} annulus {len(ANN)} far {len(FAR)}",
      flush=True)


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

    def embed(WS):
        out = []
        for i in range(0, len(WS), 256):
            xb = torch.tensor(np.stack([no.normalize(w) for w in WS[i:i + 256]]),
                              device=dev, dtype=torch.float32)
            with torch.no_grad():
                E = ag.encoder_ema({"state": xb}, None)
            out.append(E.reshape(len(xb), -1).cpu().numpy())
        return np.concatenate(out)

    def power(WS):
        out = []
        for i in range(0, len(WS), 256):
            xb = torch.tensor(np.stack([no.normalize(w) for w in WS[i:i + 256]]),
                              device=dev, dtype=torch.float32)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
            pe = np.asarray(ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, 1:9]))
            out.append(np.abs(pe[:, :, 0:3]).mean(axis=(1, 2)))
        return np.concatenate(out)

    EF = embed(FIT)
    tree = cKDTree(EF)

    def nov(WS):
        E = embed(WS)
        d, _ = tree.query(E, k=5)
        return d.mean(1)

    n_on, n_ann, n_far = nov(ONP), nov(ANN), nov(FAR)
    ref = np.median(n_on)
    p_on, p_ann, p_far = power(ONP), power(ANN), power(FAR)
    # per-state law: novelty vs attenuation on the union of off states
    nn_ = np.concatenate([n_ann, n_far]) / ref
    pp_ = np.concatenate([p_ann, p_far]) / np.median(p_on)
    m = (nn_ > 0) & (pp_ > 0)
    r = float(np.corrcoef(np.log(nn_[m]), np.log(pp_[m]))[0, 1])
    print(f"ENOV {name} nov_on 1.00 nov_ann {np.median(n_ann)/ref:.2f} "
          f"nov_far {np.median(n_far)/ref:.2f} (p90 "
          f"{np.percentile(n_far,90)/ref:.2f}) | pow_on 1.00 pow_far "
          f"{np.median(p_far)/np.median(p_on):.2f} | r(log nov, log pow) "
          f"{r:+.3f}", flush=True)
print("ENOV done", flush=True)
