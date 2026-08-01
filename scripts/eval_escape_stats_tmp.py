"""P1: decompose WHERE the standalone denoiser's correction comes from (a-slot vs s-slot).
Variants at each off-support pair (s off-support, s0 = clean anchor, a0 = anchor clean
chunk, aM = frozen MSE proposal at s; GT = recovery action at s):
  V1 = g(aM, s)    full composition            (expected bias ~0.14)
  V2 = g(aM, s0)   action-slot only (obs anchored to the clean neighbor)
  V3 = g(a0, s)    obs-slot only   (action anchored clean)
  V4 = g(a0, s0)   both anchored   (baseline)
If V2 ~ V1: correction is a-slot geometry (obs only locates the phase)  -> F1
If V3 moves toward GT strongly: genuine obs-response                    -> F2
If V1 << V2,V3 individually: interaction / gating                       -> F3"""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, h5py, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
from mip.samplers import regression_sampler
OK = ["object","robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos"]

def load(ckpt, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path="+os.path.abspath("data/tool_hang_full2ins_2000.hdf5"),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false","log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    return cfg, ds, ag

def read(path, nmax=None):
    h=h5py.File(path,"r"); g="data" if "data" in h else "demos"; ks=list(h[g])[:nmax] if nmax else list(h[g])
    out=[]
    for k in ks:
        o=h[f"{g}/{k}/obs"]
        ov=np.concatenate([np.asarray(o[key]) for key in OK],axis=1).astype(np.float32)
        a=np.clip(np.asarray(h[f"{g}/{k}/actions"]),-1,1).astype(np.float32)
        out.append((ov,a))
    h.close(); return out

cfg, ds, den = load("logs/denoise_only_2k/models/model_latest.pt","denoise_only")
_,_, mse = load("logs/full_regression_2000/models/model_latest.pt","regression")
dev=cfg.optimization.device; AS=cfg.task.act_steps; start=cfg.task.obs_steps-1; H=16
no=ds.normalizer["obs"]["state"]; na=ds.normalizer["action"]; optim=cfg.optimization
TAU=float(optim.t_two_step)
A10=ds.replay_buffer["action"][:]; ends=np.asarray(ds.replay_buffer.episode_ends[:]); st=np.concatenate([[0],ends[:-1]])
clean=read("data/tool_hang_full2ins_2000.hdf5",40)
cl=np.concatenate([ov for ov,_ in clean],0)
owner=np.concatenate([[(i,t) for t in range(len(ov))] for i,(ov,_) in enumerate(clean)],0)
mu,sig=cl.mean(0),cl.std(0)+1e-6; tree=cKDTree((cl-mu)/sig)
def win(tr,t): return [tr[max(t-1,0)],tr[t]]
def ot(w): return torch.tensor(no.normalize(np.stack(w)[None]),device=dev,dtype=torch.float32)
def chunk_n(i,t0):
    seg=A10[st[i]+t0: st[i]+t0+H]
    return None if len(seg)<H else torch.tensor(na.normalize(seg)[None],device=dev,dtype=torch.float32)
def g(an,w):
    x=ot(w)
    with torch.no_grad():
        with den._inference_mode():
            emb=den.encoder_ema({"state":x},None)
            return den.flow_map_ema.get_velocity(torch.full((1,),TAU,device=dev),an,emb)
def msechunk(w):
    x=ot(w)
    with torch.no_grad():
        with mse._inference_mode():
            return regression_sampler(optim,mse.flow_map_ema,mse.encoder_ema,torch.zeros((1,H,10),device=dev),{"state":x})
def dec(an): return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:,start:start+AS])[0][0,:6]

test=read("data/dart_test_huge_full2ins.hdf5")
E={k:[] for k in ["MSE","V1","V2","V3","V4"]}
n=0
for ov,acts in test:
    T=len(acts)
    for t in range(1,T-AS):
        if t%4: continue
        z=(ov[t]-mu)/sig; d,idx=tree.query(z)
        if not (2.0<=d<4.0): continue
        i0,t0=map(int,owner[int(idx)]); t0c=min(t0,len(clean[i0][1])-H-1)
        a0=chunk_n(i0,t0c)
        if a0 is None: continue
        aM=msechunk(win(ov,t)); gt=acts[t,:6]
        E["MSE"].append(dec(aM)-gt)
        E["V1"].append(dec(g(aM,win(ov,t)))-gt)
        E["V2"].append(dec(g(aM,win(clean[i0][0],t0)))-gt)
        E["V3"].append(dec(g(a0,win(ov,t)))-gt)
        E["V4"].append(dec(g(a0,win(clean[i0][0],t0)))-gt)
        n+=1
print(f"pairs: {n}")
print(f"{'variant':44} {'|bias|':>7} {'mean|err|':>9}")
LBL={"MSE":"MSE proposal (no denoiser)","V1":"g(aM, s)   full composition","V2":"g(aM, s0)  action-slot only (obs=anchor)","V3":"g(a0, s)   obs-slot only (action=anchor)","V4":"g(a0, s0)  both anchored"}
for k in ["MSE","V1","V2","V3","V4"]:
    X=np.stack(E[k]); print(f"{LBL[k]:44} {np.linalg.norm(X.mean(0)):>7.3f} {np.linalg.norm(X,axis=1).mean():>9.3f}")
np.savez("analysis/recovery/denoiser_decomp.npz", **{k:np.stack(v) for k,v in E.items()})
print("saved analysis/recovery/denoiser_decomp.npz")
