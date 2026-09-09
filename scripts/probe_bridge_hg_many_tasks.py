"""Extend the three-task pure-HG Bridge probe to frequency-selected groups."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path('/mnt/pfs/yuchen/groot')
SEED=20260908


def prepare(output, count):
    groups=defaultdict(list)
    for ordinal,line in enumerate((ROOT/'bridge_orig_lerobot/meta/episodes.jsonl').open()):
        row=json.loads(line)
        name=(row.get('tasks') or [''])[0].strip().lower()
        if row['length']>=24 and name:
            groups[name].append(dict(ordinal=ordinal,episode=int(row['episode_index']),length=row['length']))
    names=sorted(groups,key=lambda name:(-len(groups[name]),name))[:count]
    assert len(names)==count and all(len(groups[name])>=24 for name in names)
    old=Path('/mnt/pfs/yuchen/widowx_general_scale_20260906/raw/hg_rank0.npz')
    with np.load(old) as src:
        old_meta=json.loads(str(src['metadata']))
        original={k:src[k].copy() for k in src.files if k!='metadata'}
    assert old_meta['task_names']==names[:3]
    checkpoint=ROOT/'ft_wxpurehg/checkpoint-9000'
    config=json.loads((checkpoint/'config.json').read_text())
    assert config['ht_hg_steps']>9000 and config['hg_mode_now']
    assert str(checkpoint)==old_meta['checkpoint']
    assert config['ht_sbias']==old_meta['ht_sbias']
    selected=[]
    for tid,name in enumerate(names):
        candidates=groups[name]
        if tid<3:
            wanted=set(original['episode'][original['task_id']==tid].tolist())
            chosen=[ep for ep in candidates if ep['episode'] in wanted]
            assert len(chosen)==48
        else:
            choices=np.random.default_rng(SEED+tid).choice(len(candidates),24,replace=False)
            chosen=[candidates[int(i)] for i in sorted(choices)]
        selected.append(chosen)
    meta=dict(dataset='Bridge / WidowX',objective='hg',checkpoint=str(checkpoint),
        task_names=names,eligible_episode_counts=[len(groups[name]) for name in names],
        task_selection='40 most frequent lowercased, stripped exact instruction groups among episodes of length >=24; ties alphabetical.',
        episodes=selected,seed=SEED,steps_per_episode=12,
        episodes_per_task='48 in the original three groups (preserved); 24 in each additional group.',
        continuous_channels=list(range(6)),gripper_excluded_in_residual=True,
        sigma='masked mean softplus(s_raw + checkpoint ht_sbias) + 1e-3 over all seven valid loss channels',
        ht_sbias=config['ht_sbias'],hg_steps=config['ht_hg_steps'],
        policy_split='Training-demonstration diagnostic; calibration/density holdout is not policy holdout.',
        original_probe=str(old),original_probe_sha256=hashlib.sha256(old.read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    protocol=output/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())==meta
    else:
        protocol.write_text(json.dumps(meta,indent=2)+'\n')
    for tid in range(3):
        dest=output/f'hg_task{tid:02d}.npz'
        if not dest.exists():
            mask=original['task_id']==tid
            np.savez_compressed(dest,**{k:v[mask] for k,v in original.items()},metadata=json.dumps(meta))
    return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',type=int,default=40)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--batch',type=int,default=4)
    args=p.parse_args()
    if args.prepare_only:
        meta=prepare(args.output,args.tasks)
        print('PREPARED',len(meta['task_names']),'groups',sum(map(len,meta['episodes'])),'episodes',flush=True)
        return
    meta=json.loads((args.output/'protocol.json').read_text())
    assert len(meta['task_names'])==args.tasks
    import torch
    from probe_widowx_general_scale import load_model,map_tensors
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType
    torch.set_num_threads(2)
    model,proc=load_model(Path(meta['checkpoint']),'hetero_t')
    assert model.action_head.config.ht_sbias==meta['ht_sbias']
    tag=EmbodimentTag.resolve('SIMPLER_ENV_WIDOWX')
    modalities=proc.modality_configs[tag.value]
    loader=LeRobotEpisodeLoader(ROOT/'bridge_orig_lerobot',modalities)
    captured={}
    model.action_head.action_decoder.register_forward_hook(lambda m,i,o:captured.__setitem__('pred',o.detach()))
    model.action_head.sigma_decoder.register_forward_hook(lambda m,i,o:captured.__setitem__('raw',o.detach()))
    checked=False
    start=time.monotonic()
    for tid in range(3+args.rank,args.tasks,args.world_size):
        dest=args.output/f'hg_task{tid:02d}.npz'
        if dest.exists():
            print('EXISTS',dest,flush=True)
            continue
        rows=[]
        for ei,ep in enumerate(meta['episodes'][tid]):
            episode=loader[ep['ordinal']]
            steps=np.unique(np.rint(np.linspace(0,ep['length']-8,12)).astype(int))
            for offset in range(0,len(steps),args.batch):
                active=steps[offset:offset+args.batch]
                feats=[]
                for step in active:
                    item=extract_step_data(episode,int(step),modalities,tag,allow_padding=False)
                    feats.append(proc([{'type':MessageType.EPISODE_STEP.value,'content':item}]))
                batch=proc.collator(feats)
                inner=batch.get('inputs',batch)
                target=inner['action'].float().cpu()
                mask=inner['action_mask'].bool().cpu()
                tv=torch.nonzero(mask[0].any(1)).flatten()
                av_all=torch.nonzero(mask[0].any(0)).flatten()
                assert len(tv)==8 and len(av_all)==7
                assert torch.equal(mask,mask[:1].expand_as(mask))
                av=av_all[:6]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    model(map_tensors(inner,lambda t:t.cuda()))
                pred=captured['pred'][:,-target.shape[1]:].float().cpu()
                raw=captured['raw'][:,-target.shape[1]:].float().cpu()
                sp=torch.nn.functional.softplus(raw+meta['ht_sbias'])
                sigma=(sp*mask).sum((1,2))/mask.sum((1,2))+.001
                assert torch.isfinite(sigma).all() and torch.isfinite(pred).all()
                if not checked:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        inferred=model.get_action({k:v for k,v in inner.items() if k!='action'})['action_pred'].float().cpu()
                    diff=float((inferred[:,tv][:,:,av]-pred[:,tv][:,:,av]).abs().max())
                    assert diff<.03,diff
                    (args.output/f'inference_check_rank{args.rank}.json').write_text(json.dumps(dict(max_difference=diff)))
                    print('INFERENCE_PATH_MAX_DIFF',diff,flush=True)
                    checked=True
                for k,step in enumerate(active):
                    sub=lambda v:v[k,tv][:,av].numpy()
                    a,mu=sub(target),sub(pred)
                    rows.append(dict(task_id=tid,episode=ep['episode'],step=int(step),
                        length=ep['length'],progress=step/(ep['length']-1),sigma=float(sigma[k]),
                        target=a,prediction=mu,residual=mu-a))
        tmp=dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**{k:np.asarray([r[k] for r in rows]) for k in rows[0]},metadata=json.dumps(meta))
        tmp.replace(dest)
        print('SAVED',dest,'states',len(rows),'elapsed',round(time.monotonic()-start),flush=True)
    print('BRIDGE_HG_WORKER_DONE',args.rank,flush=True)


if __name__=='__main__':
    main()
