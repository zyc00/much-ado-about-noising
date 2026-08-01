"""FD position-block Jacobian at canonical vs deployed-failure states. Env: CKPT, LOSS, TAG."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); ag.load(os.environ["CKPT"], load_optimizer=False); ag.eval()
dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
POS = [44,45,46]; RELP = [0,1,2, 14,15,16, 28,29,30]; DELTA = 0.002
def shift(w, axis, sgn):
    w = w.copy()
    for fr in range(2):
        w[fr, POS[axis]] += sgn * DELTA
        for k in range(3): w[fr, RELP[3*k + axis]] += sgn * DELTA
    return w
def first_act(wins):
    ot = {"state": torch.tensor(no.normalize(wins), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((len(wins), 16, 10), device=dev), obs=ot, use_ema=True)
    return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[:, 0, :3]
Q = np.load("scripts/jac_queries.npz")
for qname in ["canon", "dep"]:
    eigs, fro = [], []
    for w in Q[qname]:
        wins = np.stack([shift(w, ax, sg) for ax in range(3) for sg in (+1, -1)])
        A = first_act(wins)
        J = np.zeros((3, 3))
        for ax in range(3): J[:, ax] = (A[2*ax] - A[2*ax+1]) / (2 * DELTA)
        S = (J + J.T) / 2
        eigs.append(np.linalg.eigvals(S).real.max()); fro.append(np.linalg.norm(J))
    print(f"JAC {os.environ.get('TAG','')} {qname}: ‖J‖_F p50/p90={np.median(fro):.1f}/{np.percentile(fro,90):.1f}  max-sym-eig p50/p90={np.median(eigs):+.1f}/{np.percentile(eigs,90):+.1f}")
print("DONE")
