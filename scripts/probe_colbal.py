"""Column balance of the encoder Jacobian: per-input-dim gains g_d = ||J[:,d]||^2.
Reports PR over columns (effective # of contributing input dims), gain share of top-5
columns, and the top-5 dim names. Env: none."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

NAMES53 = (
    [f"base_rel{i}" for i in "xyz"] + [f"base_rq{i}" for i in range(4)] + [f"base_pos{i}" for i in "xyz"] + [f"base_q{i}" for i in range(4)] +
    [f"frame_rel{i}" for i in "xyz"] + [f"frame_rq{i}" for i in range(4)] + [f"frame_pos{i}" for i in "xyz"] + [f"frame_q{i}" for i in range(4)] +
    [f"tool_rel{i}" for i in "xyz"] + [f"tool_rq{i}" for i in range(4)] + [f"tool_pos{i}" for i in "xyz"] + [f"tool_q{i}" for i in range(4)] +
    ["flag_asm", "flag_tool"] + [f"eef_pos{i}" for i in "xyz"] + [f"eef_q{i}" for i in range(4)] + ["grip0", "grip1"]
)
NAMES = [f"f0.{n}" for n in NAMES53] + [f"f1.{n}" for n in NAMES53]

def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device
Q = np.load("scripts/jac_queries.npz")
# QSET=full: pooled per-phase demo states across the whole trajectory
if os.environ.get("QSET") == "full":
    import h5py as _h5
    _f = _h5.File("data/tool_hang_full2ins_2000.hdf5", "r")
    _rng = np.random.RandomState(7)
    _st = []
    for _di in _rng.choice(200, 25, replace=False):
        _d = _f[f"data/demo_{_di}"]
        _a = np.clip(np.asarray(_d["actions"]), -1, 1); _g = _a[:, 6]
        _cl = [t for t in range(1, len(_g)) if _g[t-1] < 0 and _g[t] >= 0]
        if not _cl: continue
        _c1 = _cl[0]; _T = len(_g)
        _ov = np.concatenate([np.asarray(_d["obs"][q]) for q in ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]], axis=1).astype(np.float32)
        for _lo, _hi in [(_c1-40, _c1-15), (_c1-12, _c1-1), (_c1+20, _c1+60), (_c1+72, _T-2)]:
            _t = _rng.randint(max(1, _lo), min(_hi, _T-1))
            _st.append(np.stack([_ov[_t-1], _ov[_t]]))
    _f.close()
    Q = {"canon": np.stack(_st), "dep": np.stack(_st[:2])}

import os as _o
if _o.environ.get("ONE_CKPT"):
    MODELS = [(_o.environ.get("ONE_TAG", "one"), _o.environ.get("ONE_LOSS", "regression"), _o.environ["ONE_CKPT"])]
else:
    MODELS = [
        ("init", "regression", None),
        ("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
        ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt"),
        ("MIPs2", "mip", "logs/full_mip_2000_s2/models/model_latest.pt"),
        ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt"),
    ]
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    if ck is not None:
        if not os.path.exists(ck):
            print(f"COLBAL {name}: missing {ck}"); continue
        ag.load(ck, load_optimizer=False)
    ag.eval(); enc = ag.encoder_ema
    for qname in ["canon", "dep"]:
        prs, top5s, glists = [], [], []
        for w in Q[qname]:
            x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
            def f(inp): return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
            J = torch.autograd.functional.jacobian(f, x, vectorize=True).squeeze(1).detach().cpu().numpy()
            g = np.linalg.norm(J, axis=0) ** 2
            prs.append(float((g.sum() ** 2) / ((g ** 2).sum() + 1e-12)))
            order = np.argsort(g)[::-1]
            top5s.append(float(g[order[:5]].sum() / (g.sum() + 1e-12)))
            glists.append(g / (g.sum() + 1e-12))
        gm = np.mean(glists, axis=0)
        gsum = gm[:53] + gm[53:]
        print(f"COLVEC {name} {qname}: " + ",".join(f"{v:.5f}" for v in gsum), flush=True)
        top = np.argsort(gm)[::-1][:5]
        names = " ".join(f"{NAMES[i]}({gm[i]*100:.0f}%)" for i in top)
        print(f"COLBAL {name} {qname}: colPR p50={np.median(prs):.1f}/106  top5share p50={np.median(top5s):.0%}  top dims: {names}", flush=True)
        g53 = gm[:53] + gm[53:]
        print(f"GAINVEC {name} {qname}: " + ",".join(f"{v:.5f}" for v in g53), flush=True)
print("COLBAL-DONE")
