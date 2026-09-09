"""Training-set fitting dump on GR1, v2: ALL 24 datasets (so the recomputed normalization statistics equal the training run's),
batches cached once and reused for every model (identical samples), identities captured in-process (num_workers=0),
and the training-path forward loss recorded per sample as a normalization sanity check."""
import os, glob, json, numpy as np, torch
from pathlib import Path
from collections.abc import Mapping
from gr00t.configs.base_config import get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model import MODEL_REGISTRY
import gr00t.data.dataset.sharded_single_step_dataset as sss
PER_SHARD = int(os.environ.get("PER_SHARD", "25")); N_SHARDS = int(os.environ.get("N_SHARDS", "16")); N = PER_SHARD * N_SHARDS; K_FLOW = int(os.environ.get("K_FLOW", "4"))
MODELS = os.environ.get("MODELS", "mse:/mnt/pfs/yuchen/groot/ft_mse/checkpoint-60000:mse:2.0:0,ht464:/mnt/pfs/yuchen/groot/ft_ht3/checkpoint-60000:hetero_t:2.0:0,ht928:/mnt/pfs/yuchen/groot/ft_gr1c2/checkpoint-60000:hetero_t:928.0:1,l1:/mnt/pfs/yuchen/groot/ft_l1/checkpoint-60000:l1:2.0:0,flow:/mnt/pfs/yuchen/groot/ft_flow/checkpoint-60000:flow:2.0:0")
OUT = os.environ.get("OUT", "/mnt/pfs/yuchen/groot/fit_dump_gr1_v3.npz")
tag = EmbodimentTag.resolve("ROBOCASA_GR1_TABLETOP").value
paths = sorted(glob.glob("/mnt/pfs/yuchen/groot/lerobot/LeRobot/gr1_unified.*/"))
print(f"{len(paths)} datasets", flush=True)
LAST = {"ident": None}
class TaggedList(list):
    def __init__(self, items, ids, dpath): super().__init__(items); self.ids = ids; self.dpath = dpath
    def __getitem__(self, i): LAST["ident"] = (self.dpath, int(self.ids[i][0]), int(self.ids[i][1])); return super().__getitem__(i)
_orig = sss.ShardedSingleStepDataset.get_shard
def get_shard(self, idx):
    items = _orig(self, idx); ids = [(ep, int(s)) for ep, steps in self.sharded_episodes[idx] for s in steps]
    return TaggedList(items, ids, os.path.basename(str(self.episode_loader.dataset_path).rstrip("/")))
sss.ShardedSingleStepDataset.get_shard = get_shard
def to_dev(x, dev):
    if torch.is_tensor(x): return x.to(dev)
    if isinstance(x, Mapping): return type(x)({k: to_dev(v, dev) for k, v in x.items()})
    return x
def make_config(loss_type, df, mvt, ckpt, od):
    config = get_default_config().load_dict({"data": {"download_cache": False, "datasets": [{"dataset_paths": paths, "mix_ratio": 1.0, "embodiment_tag": tag}]}})
    config.load_config_path = None; config.model.loss_type = loss_type
    if loss_type == "hetero_t": config.model.ht_sbias = -1.088; config.model.ht_df = df; config.model.ht_mvt = bool(mvt)
    config.model.load_bf16 = False; config.model.reproject_vision = False; config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True; config.model.use_relative_action = True
    config.training.start_from_checkpoint = ckpt; config.training.num_gpus = 1; config.training.global_batch_size = 1
    config.training.output_dir = od; config.training.use_wandb = False; config.validate()
    (Path(od) / "experiment_cfg").mkdir(parents=True, exist_ok=True); return config
specs = [s.split(":") for s in MODELS.split(",")]
# ---- dataset once (from the first model's pipeline) ----
name0, ckpt0, lt0, df0, mvt0 = specs[0]
torch.manual_seed(0); np.random.seed(0)
cfg0 = make_config(lt0, float(df0), int(mvt0), ckpt0, f"/mnt/pfs/yuchen/groot/fitdump3_{name0}")
pipe0 = MODEL_REGISTRY.get(type(cfg0.model))(cfg0, Path(cfg0.training.output_dir) / "experiment_cfg"); pipe0.setup()
ds, _ = pipe0.return_dataset(); collator = pipe0.return_collator(); proc = pipe0.processor
try:
    npar = proc.state_action_processor.norm_params[tag]["state"]["right_arm"]; print("processor state norm right_arm min", np.round(np.array(npar["min"]), 3), "max", np.round(np.array(npar["max"]), 3), flush=True)
except Exception as e: print("norm_params print failed", e, flush=True)
print("caching batches...", flush=True)
it = iter(torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collator, num_workers=0))
batches, idents = [], []
SHARD_LEN = 1024
for sh in range(N_SHARDS):
    for j in range(SHARD_LEN):
        b = next(it)
        if j < PER_SHARD:
            inner = b["inputs"] if "inputs" in b else b
            batches.append(to_dev(inner, "cpu")); idents.append(LAST["ident"] or ("?", -1, -1))
    print(f"shard {sh}: cached {len(batches)} items, last ident={idents[-1]}", flush=True)
mask0 = batches[0]["action_mask"].reshape(-1).numpy() > 0; print("mask sum", int(mask0.sum()), "datasets seen", sorted(set(i[0] for i in idents)), flush=True)
out = {"ident_ds": np.array([i[0] for i in idents]), "ident_ep": np.array([i[1] for i in idents]), "ident_step": np.array([i[2] for i in idents])}
out["gt"] = np.stack([b["action"].float().reshape(-1).numpy()[mask0] for b in batches])
sm = batches[0]["state"].reshape(-1).numpy(); out["state"] = np.stack([b["state"].float().reshape(-1).numpy() for b in batches])
# ---- models ----
for j, (name, ckpt, lt, df, mvt) in enumerate(specs):
    if j == 0: model = pipe0.return_model()
    else:
        cfg = make_config(lt, float(df), int(mvt), ckpt, f"/mnt/pfs/yuchen/groot/fitdump3_{name}")
        pipe = MODEL_REGISTRY.get(type(cfg.model))(cfg, Path(cfg.training.output_dir) / "experiment_cfg"); model = pipe._create_model()
    model = model.cuda().eval()
    preds, losses = [], []
    with torch.no_grad():
        for k, b in enumerate(batches):
            inner = to_dev(b, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(inner)
            losses.append(float(o["loss"]) if isinstance(o, Mapping) and "loss" in o else float(o.loss))
            inf = {kk: v for kk, v in inner.items() if kk != "action"}; draws = []
            for r in range(K_FLOW if lt == "flow" else 1):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model.get_action(inf)
                a = o["action_pred"] if isinstance(o, Mapping) and "action_pred" in o else (o[0] if isinstance(o, tuple) else o)
                if isinstance(a, Mapping): a = list(a.values())[0]
                draws.append(a.float().reshape(-1).cpu().numpy()[: mask0.size][mask0].copy())
            preds.append(np.stack(draws))
            if k % 100 == 0: print(f"{name}: {k}/{N} loss={np.mean(losses):.4f} rms={np.sqrt(((draws[0]-out['gt'][k])**2).mean()):.4f}", flush=True)
    out[f"{name}_pred"] = np.array(preds); out[f"{name}_loss"] = np.array(losses)
    print(f"{name} done: mean forward loss {np.mean(losses):.4f}, residual rms {np.sqrt(((out[f'{name}_pred'][:,0]-out['gt'])**2).mean()):.4f}", flush=True)
    np.savez_compressed(OUT, **out)
    model.cpu(); del model; torch.cuda.empty_cache()
print("FITDUMP3_DONE", flush=True)
