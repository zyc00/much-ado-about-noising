"""V^R_aux->main = -<grad_theta L_aux, grad_theta R> : does an aux-only update push
the MAIN slice's off-support signed recovery R upward? Compare V_main (main-only).
R(theta) = mean over synthetic off-support probes of -n^T [f(s_off,0,0)-f(s_base,0,0)]
(differential inward component, raw pos units). CKPTS env = comma list; DS dataset."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DS = os.environ.get("DS", "data/tool_hang_full2ins_wpmatch_2000.hdf5")
OFF_CM = float(os.environ.get("OFF_CM", "3.0"))
NB = int(os.environ.get("NB", "12"))
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(DS), "network=chiunet",
        "optimization.loss_type=mip", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
ds = make_dataset(cfg.task)
dev = cfg.optimization.device
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
tts = cfg.optimization.t_two_step
# tube + affines
hf = h5py.File(DS, "r")
keys = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
grid = np.linspace(0, 1, 120); tube = []
for k in keys[:300]:
    p = np.asarray(hf[f"data/{k}/obs/robot0_eef_pos"]); t = np.linspace(0, 1, len(p))
    tube.append(np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], 1))
tube = np.stack(tube)
Cn = tube.mean(0); tgn = np.gradient(Cn, axis=0); tgn /= (np.linalg.norm(tgn, axis=1, keepdims=True) + 1e-12)
hf.close()
z = np.zeros((1, 53), dtype=np.float32)
B0 = no.unnormalize(z)[0]; A0 = np.zeros(53, dtype=np.float32)
for i in range(44, 47):
    e = z.copy(); e[0, i] = 1.0
    A0[i] = no.unnormalize(e)[0, i] - B0[i]
Ao = torch.tensor(A0[44:47], device=dev); Bo = torch.tensor(B0[44:47], device=dev)
z10 = np.zeros((1, 16, 10), dtype=np.float32)
Ba = na.unnormalize(z10)[0, 0]; Aa = np.zeros(10, dtype=np.float32)
for i in range(3):
    e = z10.copy(); e[0, :, i] = 1.0
    Aa[i] = na.unnormalize(e)[0, 0, i] - Ba[i]
AaT = torch.tensor(Aa[:3], device=dev)

from torch.utils.data import DataLoader
torch.manual_seed(0)
dl = DataLoader(ds, batch_size=192, shuffle=True, num_workers=2)
batches = []
for i, b in enumerate(dl):
    batches.append(b)
    if i >= NB - 1: break

def flat_grad(scalar, params):
    gs = torch.autograd.grad(scalar, params, retain_graph=False, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])

CK = torch.tensor(Cn, device=dev, dtype=torch.float32)
TGK = torch.tensor(tgn, device=dev, dtype=torch.float32)

for ck in os.environ["CKPTS"].split(","):
    ag = TrainingAgent(cfg); ag.load(ck, load_optimizer=False)
    net, enc = ag.flow_map, ag.encoder
    params = [p for p in list(net.parameters()) + list(enc.parameters()) if p.requires_grad]
    gA = gM = gR = None
    for b in batches:
        obs = b["obs"]["state"].to(dev).float()[:, :cfg.task.obs_steps]
        a = b["action"].to(dev).float()
        B = obs.shape[0]
        zt = torch.zeros(B, device=dev)
        # L_main
        emb = enc(obs, None)
        y0 = net.get_velocity(zt, torch.zeros_like(a), emb)
        Lm = ((y0 - a) / tts).pow(2).sum(-1).mean()
        g = flat_grad(Lm, params); gM = g if gM is None else gM + g
        # L_aux
        emb = enc(obs, None)
        noise = torch.randn_like(a)
        y1 = net.get_velocity(zt + tts, a + (1 - tts) * noise, emb)
        La = ((y1 - a) / (1 - tts)).pow(2).sum(-1).mean()
        g = flat_grad(La, params); gA = g if gA is None else gA + g
        # R: synthetic off-support differential inward (main slice)
        with torch.no_grad():
            eef = obs[:, -1, 44:47] * Ao + Bo
            diff = eef[:, None, :] - CK[None]
            kidx = torch.argmin(diff.norm(dim=2), dim=1)
            dv = eef - CK[kidx]; tg = TGK[kidx]
            dvp = dv - (dv * tg).sum(1, keepdim=True) * tg
            nhat = dvp / (dvp.norm(dim=1, keepdim=True) + 1e-9)
            disp_raw = (OFF_CM / 100.0) * nhat                 # raw meters
            disp_n = disp_raw / Ao                              # normalized obs delta
        obs_off = obs.clone()
        obs_off[:, :, 44:47] = obs_off[:, :, 44:47] + disp_n[:, None, :]
        emb_b = enc(obs, None); emb_o = enc(obs_off, None)
        y0b = net.get_velocity(zt, torch.zeros_like(a), emb_b)
        y0o = net.get_velocity(zt, torch.zeros_like(a), emb_o)
        da = (y0o[:, 1, :3] - y0b[:, 1, :3]) * AaT
        Rv = (-(da * nhat).sum(1)).mean()
        g = flat_grad(Rv, params); gR = g if gR is None else gR + g
    gA /= NB; gM /= NB; gR /= NB
    vA = -float(torch.dot(gA, gR)); vM = -float(torch.dot(gM, gR))
    cA = vA / (gA.norm() * gR.norm() + 1e-12).item(); cM = vM / (gM.norm() * gR.norm() + 1e-12).item()
    print(f"VAUX {os.path.basename(os.path.dirname(os.path.dirname(ck)))}/{os.path.basename(ck)}: "
          f"V_aux={vA:+.3e} V_main={vM:+.3e} ratio={vA/(abs(vM)+1e-12):+.2f} "
          f"cos_aux={-cA:+.3f} cos_main={-cM:+.3f} |gA|={gA.norm():.2e} |gM|={gM.norm():.2e} |gR|={gR.norm():.2e}", flush=True)
print("VAUX-DONE")
