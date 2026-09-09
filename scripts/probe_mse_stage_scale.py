#!/usr/bin/env python3
"""Frozen late-MSE diagnostic; see analysis/paper/mse_scale/PROTOCOL.md.

Runs in the corresponding stack's existing environment. No training and no
changes to checkpoints/datasets. All artifacts are written to --out.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


SEED = 20260904


def device(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: device(v) for k, v in x.items()}
    return x


def sample_steps(length, first, last, per_bin, rng):
    """Actual episode progress; no terminal padding in the executed chunk."""
    result = []
    for b in range(10):
        lo = max(first, int(np.ceil(b * length / 10)))
        hi = min(last + 1, int(np.ceil((b + 1) * length / 10)))
        if lo < hi:
            for t in sorted(rng.choice(np.arange(lo, hi), min(per_bin, hi-lo), replace=False)):
                result.append((int(t), b))
    return result


class Dump:
    def __init__(self, path, meta):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.rows = []
        self.arrays = {}

    def add(self, rows, **arrays):
        self.rows.extend(rows)
        for k, v in arrays.items():
            self.arrays.setdefault(k, []).append(np.asarray(v))

    def save(self):
        if not self.rows:
            return
        temporary=self.path.with_name(self.path.stem+'.writing.npz')
        np.savez_compressed(temporary, **{k: np.concatenate(v) for k,v in self.arrays.items()},
                            task=np.array([r['task'] for r in self.rows]),
                            episode=np.array([r['episode'] for r in self.rows]),
                            step=np.array([r['step'] for r in self.rows]),
                            length=np.array([r['length'] for r in self.rows]),
                            stage=np.array([r['stage'] for r in self.rows]),
                            split=np.array([r['split'] for r in self.rows]),
                            metadata=np.array(json.dumps(self.meta)))
        temporary.replace(self.path)
        self.path.with_suffix('.json').write_text(json.dumps({**self.meta, 'n':len(self.rows)}, indent=2))


def run_pi05(args):
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies import make_pre_post_processors

    root = Path('/mnt/pfs/yuchen/pi05')
    ckpts = [root/'run_mse/checkpoints'/s/'pretrained_model' for s in ('030000','025000')]
    data_root = root/'libero_lerobot'
    models = [PI05Policy.from_pretrained(str(c)).cuda().eval() for c in ckpts]
    for model in models:
        assert model.config.loss_type == 'mse'
        assert not model.config.use_visual_memory and not model.config.use_proprioceptive_memory
    pre, _ = make_pre_post_processors(models[0].config, pretrained_path=str(ckpts[0]))
    T = models[0].config.chunk_size
    fps = json.loads((data_root/'meta/info.json').read_text())['fps']
    ds = LeRobotDataset(repo_id='HuggingFaceVLA/libero', root=data_root,
                       delta_timestamps={'action':[t/fps for t in range(T)]})
    eps = pd.concat([pd.read_parquet(p) for p in sorted((data_root/'meta/episodes').rglob('*.parquet'))])
    eps['task_name'] = eps['tasks'].apply(lambda v: str(v[0]))
    if args.max_tasks:
        names = sorted(eps.task_name.unique())[:args.max_tasks]
        eps = eps[eps.task_name.isin(names)]
    dump = Dump(Path(args.out)/'pi05.npz', dict(stack='pi05', checkpoints=list(map(str,ckpts)),
        checkpoint_steps=[30000,25000], dataset=str(data_root), policy_split='train',
        seed=SEED, executed_start=0, executed_horizon=10, continuous_channels=list(range(6)),
        normalization='saved checkpoint action normalizer', task_count=eps.task_name.nunique()))
    rng = np.random.default_rng(SEED)
    t0=time.time()
    for task, group in eps.groupby('task_name', sort=True):
        chosen = rng.permutation(len(group))[:args.episodes]
        for rank, pos in enumerate(chosen):
            ep = group.iloc[pos]
            n = int(ep['length'])
            points = sample_steps(n,0,n-10,1,rng)
            rows = [dict(task=task,episode=int(ep.episode_index),step=t,length=n,stage=b,split=rank%2) for t,b in points]
            for j in range(0,len(points),args.batch):
                rr=rows[j:j+args.batch]
                raw = torch.utils.data.default_collate([ds[int(ep.dataset_from_index)+r['step']] for r in rr])
                assert all(int(e)==int(ep.episode_index) for e in raw['episode_index'])
                batch=device(pre(raw))
                gt=batch['action'].float().cpu().numpy()
                mask=np.arange(T)[None,:] < np.array([n-r['step'] for r in rr])[:,None]
                preds=[]
                with torch.inference_mode():
                    for model in models:
                        pred=model.predict_action_chunk(batch).float()
                        preds.append(pred.cpu().numpy())
                assert gt.shape == preds[0].shape and gt.shape[-1]==7
                if not dump.rows:
                    with torch.inference_mode():
                        repeat=models[0].predict_action_chunk(batch).float().cpu().numpy()
                        loss, detail=models[0].forward(batch)
                    dump.meta['repeat_max_abs']=float(np.max(np.abs(repeat-preds[0])))
                    dump.meta['first_batch_forward_loss']=float(loss)
                    dump.meta['first_batch_inference_mse_including_gripper_padding']=float(np.mean((gt-preds[0])**2))
                    print('SANITY',dump.meta,flush=True)
                dump.add(rr,gt=gt,pred_final=preds[0],pred_late=preds[1],valid_time=mask,
                         state=raw['observation.state'].numpy())
            if rank%4==0:
                print('PROGRESS pi05',task,rank,'n',len(dump.rows),'seconds',round(time.time()-t0),flush=True)
        dump.save()
    dump.save()
    print('DONE pi05',len(dump.rows),flush=True)


def run_gr1(args):
    from transformers import AutoModel, AutoProcessor
    import gr00t.model  # registers model/processor classes
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing

    root=Path('/mnt/pfs/yuchen/groot')
    ckpts=[root/'ft_mse'/f'checkpoint-{s}' for s in (60000,58000)]
    tag=EmbodimentTag.resolve('ROBOCASA_GR1_TABLETOP')
    # Resolve the cached processor locally: this Transformers version otherwise
    # calls model_info while patching a tokenizer, even in local-files-only mode.
    backbone_cache=Path('/mnt/pfs/yuchen/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561')
    original_build_processor=processing.build_processor
    processing.build_processor=lambda name, kwargs: original_build_processor(str(backbone_cache),kwargs)
    proc=AutoProcessor.from_pretrained(str(ckpts[0]),local_files_only=True,
        model_name=str(backbone_cache),transformers_loading_kwargs={'local_files_only':True})
    proc.eval()
    assert proc.state_action_processor.use_relative_action, 'Must preserve relative action training targets'
    modalities=proc.modality_configs[tag.value]
    print('MODALITIES',modalities,flush=True)
    models=[]
    for c in ckpts:
        model, info=AutoModel.from_pretrained(str(c),local_files_only=True,output_loading_info=True,
             transformers_loading_kwargs={'local_files_only':True})
        assert not info.get('missing_keys') and not info.get('unexpected_keys'), info
        assert model.config.loss_type=='mse'
        models.append(model.cuda().eval())
    paths=sorted((root/'lerobot/LeRobot').glob('gr1_unified.*'))
    if args.max_tasks:
        paths=paths[:args.max_tasks]
    dump=Dump(Path(args.out)/'gr1.npz',dict(stack='gr1',checkpoints=list(map(str,ckpts)),
        checkpoint_steps=[60000,58000],dataset_paths=list(map(str,paths)),policy_split='train',
        seed=SEED,executed_start=0,executed_horizon=8,continuous_channels=list(range(29)),
        normalization='saved checkpoint statistics.json; no dataset statistics recomputation',
        task_count=len(paths),joint_order=list(modalities['action'].modality_keys)))
    rng=np.random.default_rng(SEED)
    t0=time.time()
    for path in paths:
        loader=LeRobotEpisodeLoader(path,modalities)
        eligible=np.flatnonzero(np.asarray(loader.episode_lengths)>=80)
        chosen=rng.permutation(eligible)[:args.episodes]
        for rank,ep in enumerate(chosen):
            episode=loader[int(ep)]
            n=len(episode)
            first=max(0,-min(d for c in modalities.values() for d in c.delta_indices))
            last=n-1-max(d for c in modalities.values() for d in c.delta_indices)
            points=sample_steps(n,first,last,1,rng)
            rows=[dict(task=path.name,episode=int(ep),step=t,length=n,stage=b,split=rank%2) for t,b in points]
            for j in range(0,len(points),args.batch):
                rr=rows[j:j+args.batch]
                features=[]
                for row in rr:
                    data=extract_step_data(episode,row['step'],modalities,tag,allow_padding=False)
                    features.append(proc([{'type':MessageType.EPISODE_STEP.value,'content':data}]))
                batch=device(proc.collator(features))
                inner=batch['inputs'] if 'inputs' in batch else batch
                gt_full=inner['action'].float()
                action_mask=inner['action_mask'].bool()
                tv=torch.nonzero(action_mask[0].any(dim=1)).flatten()
                av=torch.nonzero(action_mask[0].any(dim=0)).flatten()
                assert len(tv)==8 and len(av)==29,(len(tv),len(av))
                gt=gt_full[:,tv][:,:,av].cpu().numpy()
                preds=[]
                inf={k:v for k,v in inner.items() if k!='action'}
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for model in models:
                        result=model.get_action(inf)
                        pred=result['action_pred']
                        preds.append(pred[:,tv][:,:,av].float().cpu().numpy())
                if not dump.rows:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        repeat=models[0].get_action(inf)['action_pred'][:,tv][:,:,av].float().cpu().numpy()
                        result=models[0](inner)
                    dump.meta['repeat_max_abs']=float(np.max(np.abs(repeat-preds[0])))
                    dump.meta['first_batch_forward_loss']=float(result['loss'])
                    dump.meta['first_batch_inference_mse']=float(np.mean((gt-preds[0])**2))
                    print('SANITY',dump.meta,flush=True)
                dump.add(rr,gt=gt,pred_final=preds[0],pred_late=preds[1],
                         valid_time=np.ones((len(rr),8),bool),state=inner['state'].float().cpu().numpy().reshape(len(rr),-1))
            del episode
            if rank%4==0:
                print('PROGRESS gr1',path.name,rank,'n',len(dump.rows),'seconds',round(time.time()-t0),flush=True)
        dump.save()
    dump.save()
    print('DONE gr1',len(dump.rows),flush=True)


def run_robomimic(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from mip.agent import TrainingAgent
    from mip.datasets.robomimic_dataset import make_dataset

    root=Path('/mnt/pfs/yuchen/code/much-ado-about-noising')
    specs=[('tool_hang','tool_hang_ph_state_delta_legacy',root/'data/tool_hang_human_lowdim_up.hdf5',
            root/'logs/snap_hmse_chi/models',('snap_299999.pt','snap_287999.pt'),(299999,287999)),
           ('transport','transport_ph_state_delta_legacy',Path('/mnt/pfs/yuchen/data/mip/robomimic/transport/ph/low_dim.hdf5'),
            root/'logs/trph_l2_s1000/models',('snap_300000.pt','snap_280000.pt'),(300000,280000))]
    for name,task,data,ckdir,files,steps in specs:
        with initialize_config_dir(version_base=None,config_dir=str(root/'examples/configs')):
            cfg=compose(config_name='main',overrides=[f'task={task}',f'+task.dataset_path={data}',
                 'network=chiunet','optimization.loss_type=regression','optimization.auto_resume=false',
                 'optimization.use_compile=false','optimization.use_cudagraphs=false','log.wandb_mode=disabled'])
        OmegaConf.set_struct(cfg,False)
        cfg.task.horizon=int(2**np.ceil(np.log2(cfg.task.horizon)))
        ds=make_dataset(cfg.task)
        cfg.task.obs_dim=ds[0]['obs']['state'].shape[-1]
        models=[]
        for f in files:
            model=TrainingAgent(cfg)
            model.load(str(ckdir/f),load_optimizer=False)
            model.eval()
            models.append(model)
        T=cfg.task.horizon
        A=cfg.task.act_dim
        keep=[i for i in range(A) if i%10!=9]
        dump=Dump(Path(args.out)/(name+'.npz'),dict(stack='robomimic',task=name,
            checkpoints=[str(ckdir/f) for f in files],checkpoint_steps=list(steps),dataset=str(data),
            policy_split='train',seed=SEED,executed_start=cfg.task.obs_steps-1,executed_horizon=8,
            continuous_channels=keep,normalization='original full-dataset normalizer',
            task_count=1,config=OmegaConf.to_container(cfg,resolve=True)))
        starts=np.r_[0,ds.replay_buffer.episode_ends[:-1]]
        ends=ds.replay_buffer.episode_ends[:]
        indices=ds.sampler.indices
        # Map the current observation's absolute replay-buffer index to its sampler row.
        current=indices[:,0]-indices[:,2]+cfg.task.obs_steps-1
        lookup={int(c):i for i,c in enumerate(current)}
        rng=np.random.default_rng(SEED)
        ep_order=rng.permutation(len(ends))
        if args.max_tasks:
            ep_order=ep_order[:args.episodes]
        samples=[]
        for rank,ep in enumerate(ep_order):
            start,end=int(starts[ep]),int(ends[ep]); n=end-start
            for t,b in sample_steps(n,cfg.task.obs_steps-1,n-8,2,rng):
                idx=lookup.get(start+t)
                if idx is not None:
                    samples.append((idx,dict(task=name,episode=int(ep),step=t,length=n,stage=b,split=rank%2)))
        for j in range(0,len(samples),64):
            items=samples[j:j+64]
            rr=[r for _,r in items]
            raw=torch.utils.data.default_collate([ds[i] for i,_ in items])
            gt=raw['action'].float().numpy()
            obs={'state':raw['obs']['state'][:,:cfg.task.obs_steps].cuda().float()}
            preds=[]
            with torch.inference_mode():
                for model in models:
                    p=model.sample(torch.zeros((len(rr),T,A),device='cuda'),obs,use_ema=True)
                    preds.append(p.float().cpu().numpy())
            mask=np.stack([np.arange(T)>=int(indices[i,2]) for i,_ in items])
            mask &= np.stack([np.arange(T)<int(indices[i,3]) for i,_ in items])
            if not dump.rows:
                with torch.inference_mode():
                    repeat=models[0].sample(torch.zeros((len(rr),T,A),device='cuda'),obs,use_ema=True).cpu().numpy()
                dump.meta['repeat_max_abs']=float(np.max(np.abs(repeat-preds[0])))
                print('SANITY',name,gt.shape,dump.meta['repeat_max_abs'],flush=True)
            dump.add(rr,gt=gt,pred_final=preds[0],pred_late=preds[1],valid_time=mask,
                     state=obs['state'].cpu().numpy().reshape(len(rr),-1))
            if j%512==0:
                print('PROGRESS',name,len(dump.rows),'/',len(samples),flush=True)
        dump.save()
        print('DONE',name,len(dump.rows),flush=True)
        del models,model,ds
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stack',choices=['gr1','pi05','robomimic'])
    parser.add_argument('--out',required=True)
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--max-tasks',type=int,default=0,help='nonzero only for smoke tests')
    args=parser.parse_args()
    torch.manual_seed(SEED);np.random.seed(SEED)
    torch.set_num_threads(4)
    globals()['run_'+args.stack](args)


if __name__=='__main__':
    main()
