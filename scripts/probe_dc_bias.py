"""THE MEDIATOR TEST for the structural (+15) component: decompose each
arm's held-out prediction error into its CHUNK-COHERENT (DC) and INCOHERENT
components, in the same units as the ACT_SBIAS dose-response.

Per held-out state (demos 2000-2600 of the 20k file, same protocol as
probe_insupport_precision): e_t = pred_t - gt_t over the executed window
(chunk steps 1..8), position dims, UNNORMALIZED action units.
  dc(x)   = || mean_t e_t ||           (the component ACT_SBIAS injects)
  inc(x)  = mean_t || e_t - mean e ||  (the component ACT_NOISE injects)
Pre-registered predictions: dc medians order L2 > NONOISE > MIP and track
SR; inc does not; the L2-MIP dc DIFFERENCE is of the order of the sbias
doses that move SR (0.004-0.008 action units). Prints DCB lines.
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

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("NONOISE", "mip_nonoise", "logs/mp200_nonoise_s1000/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt"),
    ("HG", "regression_hetero_gauss", "logs/mp200_hg_s1000/models/snap_300000.pt"),
]
SR = {"L2": 63, "NONOISE": 79, "MIP": 94, "HT": 75, "HG": 87,
      "CND2B": 82, "CBAL50": 70, "ALIGNRS": 67}
if os.environ.get("DC_ARMS"):
    ARMS = [tuple(a.split(":")) for a in os.environ["DC_ARMS"].split(",")]
NS = int(os.environ.get("DC_N", "4000"))
AS_LO, AS_HI = 1, 9      # executed window (harness convention)

h = h5py.File(HELD, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2600]
rng = np.random.RandomState(0)
W, GA, PH = [], [], []
per = max(1, NS // len(names))
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    L = len(S)
    if L < 20:
        continue
    idx = rng.choice(np.arange(1, L - 10), min(per, L - 11), replace=False)
    for i in idx:
        W.append(np.stack([S[i - 1], S[i]]))
        GA.append(A[i - 1 + AS_LO:i - 1 + AS_HI])   # executed 8 raw actions
        PH.append(i / L)
h.close()
W, GA, PH = np.stack(W), np.stack(GA), np.asarray(PH)
print(f"DCB held-out states {len(W)}", flush=True)

rows = {}
for name, loss, ck in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
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

    DC, INC = [], []
    for i in range(0, len(W), 256):
        xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 256]]),
                          device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), H, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema,
                         ag.encoder_ema, a0, {"state": xb})
        pe = np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, AS_LO:AS_HI]))
        e = pe[:, :, 0:3] - GA[i:i + 256, :, 0:3]     # raw action units, pos
        m = e.mean(axis=1)                            # (B, 3) chunk mean
        DC.append(np.linalg.norm(m, axis=1))
        INC.append(np.linalg.norm(e - m[:, None, :], axis=2).mean(axis=1))
    DC, INC = np.concatenate(DC), np.concatenate(INC)
    rows[name] = (DC, INC)
    late = PH >= 0.5
    print(f"DCB {name} SR={SR[name]} dc_p50 {np.median(DC):.5f} "
          f"dc_p90 {np.percentile(DC,90):.5f} | inc_p50 {np.median(INC):.5f} "
          f"| dc_p50 early {np.median(DC[~late]):.5f} late "
          f"{np.median(DC[late]):.5f}", flush=True)

# paired deltas vs L2 and rank correlations
from scipy.stats import spearmanr, wilcoxon
dcl, incl = rows["L2"]
for name in ("NONOISE", "MIP", "HT", "HG"):
    dc, inc = rows[name]
    w = wilcoxon(dcl, dc)
    print(f"DCB pair L2-{name}: d_dc_p50 {np.median(dcl - dc):+.5f} "
          f"wilcoxon_p {w.pvalue:.2e} | d_inc_p50 "
          f"{np.median(incl - inc):+.5f}", flush=True)
srv = [SR[n] for n in rows]
dcm = [np.median(rows[n][0]) for n in rows]
incm = [np.median(rows[n][1]) for n in rows]
print(f"DCB spearman(SR, dc_p50) {spearmanr(srv, dcm).statistic:+.2f} | "
      f"spearman(SR, inc_p50) {spearmanr(srv, incm).statistic:+.2f}",
      flush=True)
print("DCB done", flush=True)
