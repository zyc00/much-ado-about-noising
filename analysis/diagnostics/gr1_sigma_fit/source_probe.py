"""Dump per-element residuals and per-sample learned sigma of a trained HT/HG
GR00T checkpoint on GR1 training data, for Q-Q / calibration plots.

Same model/data setup as resid_fit_gr1.py (eval mode, bf16 autocast, first 4
GR1 datasets, fixed seed, no shuffle -> the same samples for every checkpoint).

Usage: EMB=gr1 CKPT=<ckpt dir> SUF=<tag> N_BATCHES=150 python resid_dump_gr1.py
"""
import glob
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from gr00t.configs.base_config import get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model import MODEL_REGISTRY

EMB = os.environ.get("EMB", "gr1")
N_BATCHES = int(os.environ.get("N_BATCHES", "150"))
BATCH = 8
SUF = os.environ.get("SUF", "")
torch.manual_seed(0)

if EMB == "gr1":
    tag = EmbodimentTag.resolve("ROBOCASA_GR1_TABLETOP").value
    paths = sorted(glob.glob("/mnt/pfs/yuchen/groot/lerobot/LeRobot/*/"))[:int(os.environ.get("N_DS", "4"))]
    sbias = -1.0884732364972949
    state_dropout = None
    OUT = "/mnt/pfs/yuchen/groot/gate_out_gr1"
else:
    tag = EmbodimentTag.resolve("SIMPLER_ENV_WIDOWX").value
    paths = ["/mnt/pfs/yuchen/groot/bridge_orig_lerobot/"]
    sbias = -0.45346351477341934
    state_dropout = 0.8
    OUT = "/mnt/pfs/yuchen/groot/gate_out_widowx"

CKPT = os.environ.get("CKPT", "/mnt/pfs/yuchen/groot/model")
print(f"EMB={EMB} CKPT={CKPT} SUF={SUF} sbias={sbias} datasets={[os.path.basename(p.rstrip('/')) for p in paths]}", flush=True)

config = get_default_config().load_dict(
    {"data": {"download_cache": False,
              "datasets": [{"dataset_paths": paths, "mix_ratio": 1.0, "embodiment_tag": tag}]}}
)
config.load_config_path = None
config.model.loss_type = "hetero_t"
config.model.ht_sbias = float(os.environ.get("SBIAS", str(sbias)))
config.model.ht_df = 2.0
config.model.load_bf16 = False
config.model.reproject_vision = False
config.model.model_name = "nvidia/Cosmos-Reason2-2B"
config.model.backbone_trainable_params_fp32 = True
config.model.use_relative_action = True
if state_dropout is not None:
    config.model.state_dropout_prob = state_dropout
config.training.start_from_checkpoint = CKPT
config.training.num_gpus = 1
config.training.global_batch_size = BATCH
config.training.output_dir = OUT
config.training.use_wandb = False
config.validate()

(Path(OUT) / "experiment_cfg").mkdir(parents=True, exist_ok=True)
pipeline = MODEL_REGISTRY.get(type(config.model))(config, Path(OUT) / "experiment_cfg")
pipeline.setup()
model = pipeline.return_model().cuda().eval()
train_dataset, _ = pipeline.return_dataset()
collator = pipeline.return_collator()

# EFFECTIVE sbias: the value the model's own config carries (what the loss site uses)
mcfg = getattr(model, "config", None)
sb_model = getattr(mcfg, "ht_sbias", None)
sb_head = getattr(getattr(model.action_head, "config", None), "ht_sbias", None)
print(f"[effective] config.model.ht_sbias={config.model.ht_sbias} model.config.ht_sbias={sb_model} "
      f"action_head.config.ht_sbias={sb_head} hg_steps={getattr(mcfg, 'ht_hg_steps', None)}", flush=True)
SB_USED = float(sb_head if sb_head is not None else (sb_model if sb_model is not None else config.model.ht_sbias))
print(f"[effective] sbias used for sigma = {SB_USED}", flush=True)

captured = {}
model.action_head.action_decoder.register_forward_hook(
    lambda m, i, o: captured.__setitem__("pred", o.detach())
)
assert hasattr(model.action_head, "sigma_decoder"), "no sigma_decoder in this checkpoint"
model.action_head.sigma_decoder.register_forward_hook(
    lambda m, i, o: captured.__setitem__("s_raw", o.detach())
)
loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH,
                                     collate_fn=collator, num_workers=4)


def to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, Mapping):
        return type(x)({k: to_cuda(v) for k, v in x.items()})
    return x


R, ACT, SP, SIG, M, D = [], [], [], [], [], []
LOSS_EVAL, ALOSS_EVAL, LOSS_TRAIN, ALOSS_TRAIN, M_TRAIN, SIG_TRAIN = [], [], [], [], [], []
t_valid = a_valid = None
with torch.no_grad():
    for bi, batch in enumerate(loader):
        if bi >= N_BATCHES:
            break
        batch = to_cuda(batch)
        inner = batch["inputs"] if "inputs" in batch else batch
        # ---- eval-mode forward (as before) ----
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        LOSS_EVAL.append(float(out["loss"])); ALOSS_EVAL.append(out["action_loss"].float().cpu().numpy())
        act = inner["action"].float()
        mask = inner["action_mask"].float()
        pred = captured["pred"][:, -act.shape[1]:].float()
        sr = captured["s_raw"][:, -act.shape[1]:].float()
        B, T, A = act.shape
        mk = mask.reshape(B, T, A)
        if t_valid is None:
            t_valid = torch.nonzero(mk.sum(dim=(0, 2)) > 0).flatten()
            a_valid = torch.nonzero(mk.sum(dim=(0, 1)) > 0).flatten()
            print(f"real extent: T={len(t_valid)} A={len(a_valid)} (padded {T}x{A})", flush=True)
        sp = torch.nn.functional.softplus(sr + SB_USED) * mask
        d = mask.reshape(B, -1).sum(1).clamp_min(1.0)
        sigma = sp.reshape(B, -1).sum(1) / d + 1e-3
        r = (pred - act) * mask
        m = (r ** 2).reshape(B, -1).sum(1) / d
        sub = lambda x: x[:, t_valid][:, :, a_valid].cpu().numpy().astype(np.float32)
        R.append(sub(r)); ACT.append(sub(act)); SP.append(sub(sp))
        SIG.append(sigma.cpu().numpy()); M.append(m.cpu().numpy()); D.append(d.cpu().numpy())
        # cross-check: per-sample NLL recomputed from (m, sigma) must equal the model's action_loss
        if bi == 0:
            nll_hat = 0.5 * m * d / sigma ** 2 + d * torch.log(sigma)        # Gaussian form
            df = float(getattr(model.action_head.config, "ht_df", 2.0))
            nll_t = 0.5 * (df + 1.0) * torch.log1p(m / (df * sigma ** 2)) * d + d * torch.log(sigma)
            print(f"[xcheck b0] model action_loss[:4]={out['action_loss'].float()[:4].cpu().numpy()} "
                  f"gauss_hat={nll_hat[:4].cpu().numpy()} t_hat(df={df})={nll_t[:4].cpu().numpy()} "
                  f"model loss={float(out['loss']):.4f} d={float(d[0])}", flush=True)
        # ---- train-mode forward on the SAME batch (dropout/state-dropout active), no grad ----
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_t = model(**batch)
        LOSS_TRAIN.append(float(out_t["loss"])); ALOSS_TRAIN.append(out_t["action_loss"].float().cpu().numpy())
        pred_t = captured["pred"][:, -act.shape[1]:].float()
        sr_t = captured["s_raw"][:, -act.shape[1]:].float()
        sp_t = torch.nn.functional.softplus(sr_t + SB_USED) * mask
        sigma_t = sp_t.reshape(B, -1).sum(1) / d + 1e-3
        m_t = (((pred_t - act) * mask) ** 2).reshape(B, -1).sum(1) / d
        M_TRAIN.append(m_t.cpu().numpy()); SIG_TRAIN.append(sigma_t.cpu().numpy())
        model.eval()
        if bi % 10 == 0:
            print(f"batch {bi} n={sum(len(x) for x in SIG)} EVAL: loss={LOSS_EVAL[-1]:.4f} sigma_med={float(sigma.median()):.4f} "
                  f"rms_med={float(m.median() ** 0.5):.4f} | TRAIN-MODE: loss={LOSS_TRAIN[-1]:.4f} "
                  f"sigma_med={float(sigma_t.median()):.4f} rms_med={float(m_t.median() ** 0.5):.4f}", flush=True)

print(f"[summary] eval-mode mean loss {np.mean(LOSS_EVAL):.4f} | train-mode mean loss {np.mean(LOSS_TRAIN):.4f}", flush=True)
out = f"/mnt/pfs/yuchen/groot/resid_dump3_{EMB}{SUF}.npz"
np.savez(out, r=np.concatenate(R), act=np.concatenate(ACT), sp=np.concatenate(SP),
         sigma=np.concatenate(SIG), m=np.concatenate(M), d=np.concatenate(D),
         t_valid=t_valid.cpu().numpy(), a_valid=a_valid.cpu().numpy(),
         sbias=np.array(SB_USED), ckpt=np.array(CKPT), loss_eval=np.array(LOSS_EVAL), loss_train=np.array(LOSS_TRAIN),
         aloss_eval=np.concatenate(ALOSS_EVAL), aloss_train=np.concatenate(ALOSS_TRAIN), m_train=np.concatenate(M_TRAIN), sigma_train=np.concatenate(SIG_TRAIN))
print(f"RESID_DUMP3_SAVED {out} n={sum(len(x) for x in SIG)}", flush=True)
