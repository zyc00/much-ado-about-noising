"""Frozen training-demonstration audit, not historical training gradients.

Uniformly select 20 episodes per drawer instruction; evaluate every valid
chunk start without augmentation/dropout. Save labels/predictions and exact
shared-sigma likelihood weights. No rollout pseudo-labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from probe_widowx_general_scale import load_model, map_tensors
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import MessageType


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path('/mnt/pfs/yuchen/groot/fractal_lerobot')
    ckpt=Path('/mnt/pfs/yuchen/groot/ft_fr_nu224/checkpoint-20000')
    names=['close bottom drawer','close middle drawer','close top drawer']
    pools={n:[] for n in names}
    for ordinal,line in enumerate((root/'meta/episodes.jsonl').open()):
        row=json.loads(line); name=(row.get('tasks') or [''])[0].strip().lower()
        if name in pools and row['length']>=16:
            pools[name].append(dict(row,ordinal=ordinal))
    rng=np.random.default_rng(20260909)
    selected=[]
    for task,name in enumerate(names):
        for j in sorted(rng.choice(len(pools[name]),20,replace=False)):
            selected.append(dict(pools[name][j],task_id=task))
    model,proc=load_model(ckpt,'hetero_t');cfg=model.action_head.config
    assert cfg.ht_mvt and cfg.ht_df==224 and not cfg.ht_gripper_bce
    assert getattr(cfg,'ht_beta',0)==0 and getattr(cfg,'ht_sigma_clamp',-1)<0
    tag=EmbodimentTag.resolve('SIMPLER_ENV_GOOGLE');mods=proc.modality_configs[tag.value]
    assert mods['action'].modality_keys==['x','y','z','roll','pitch','yaw','gripper']
    loader=LeRobotEpisodeLoader(root,mods)
    cap={}
    model.action_head.action_decoder.register_forward_hook(lambda m,i,o:cap.__setitem__('pred',o.detach()))
    model.action_head.sigma_decoder.register_forward_hook(lambda m,i,o:cap.__setitem__('raw',o.detach()))
    meta=dict(checkpoint=str(ckpt),seed=20260909,names=names,episodes=selected,
              nu=float(cfg.ht_df),sbias=float(cfg.ht_sbias),definition='all valid starts; no progress filtering',
              scope='Frozen checkpoint on training demonstrations in eval mode; not historical optimizer gradients')
    (args.output/'protocol.json').write_text(json.dumps(meta,indent=2))
    rows=[]; thumbs=[]; max_loss_diff=0.
    for uid,ep in enumerate(selected):
        episode=loader[ep['ordinal']]
        starts=list(range(ep['length']-7))
        for offset in range(0,len(starts),8):
            steps=starts[offset:offset+8];features=[];raw=[];images=[]
            for step in steps:
                item=extract_step_data(episode,step,mods,tag,allow_padding=False)
                raw.append(np.concatenate([item.actions[k] for k in mods['action'].modality_keys],axis=-1))
                frame=next(iter(item.images.values()))[-1]
                images.append(np.asarray(Image.fromarray(np.asarray(frame)).resize((120,96))))
                features.append(proc([{'type':MessageType.EPISODE_STEP.value,'content':item}]))
            batch=proc.collator(features);inner=batch.get('inputs',batch)
            y=inner['action'].float().cpu();mask=inner['action_mask'].float().cpu()
            assert torch.all(mask.sum((1,2))==56)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(map_tensors(inner,lambda t:t.cuda()))
            pred=cap['pred'][:,-y.shape[1]:].float().cpu()
            sp=torch.nn.functional.softplus(cap['raw'][:,-y.shape[1]:].float().cpu()+cfg.ht_sbias)
            sigma=(sp*mask).sum((1,2))/56+.001
            S=((pred-y).square()*mask).sum((1,2))
            gate=(cfg.ht_df+56)/(cfg.ht_df+S/sigma.square())
            weight=gate/sigma.square()
            expected=((cfg.ht_df+56)/2*torch.log1p(S/(cfg.ht_df*sigma.square()))+56*torch.log(sigma)).mean()/56
            err=abs(float(result['loss'])-float(expected));max_loss_diff=max(max_loss_diff,err)
            assert err<2e-5,(float(result['loss']),float(expected))
            for b,step in enumerate(steps):
                rows.append(dict(episode=ep['episode_index'],episode_uid=uid,task_id=ep['task_id'],step=step,
                                 progress=step/max(1,ep['length']-1),sigma=float(sigma[b]),S=float(S[b]),
                                 gate=float(gate[b]),weight=float(weight[b]),
                                 target=y[b,:8,:7].numpy(),prediction=pred[b,:8,:7].numpy(),raw_target=raw[b]))
                thumbs.append(images[b])
        if (uid+1)%5==0: print(f'PUSH_AUDIT {uid+1}/60 episodes {len(rows)} states',flush=True)
    np.savez_compressed(args.output/'probe.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
    np.savez_compressed(args.output/'frames.npz',images=np.stack(thumbs))
    (args.output/'done.json').write_text(json.dumps(dict(states=len(rows),episodes=len(selected),max_loss_formula_difference=max_loss_diff)))
    print('PUSH_AUDIT_COMPLETE',len(rows),max_loss_diff,flush=True)

if __name__=='__main__':main()
