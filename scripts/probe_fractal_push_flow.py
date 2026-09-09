"""Matched Flow predictions for manually inspected episodes; same labels and states."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from probe_widowx_general_scale import load_model,map_tensors
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import MessageType

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
z=np.load(args.root/'probe.npz');meta=json.loads((args.root/'protocol.json').read_text())
intervals=json.loads((args.root/'visual_push_intervals.json').read_text())['windows']
uids=sorted(map(int,intervals));selected=np.flatnonzero(np.isin(z['episode_uid'],uids))
ck=Path('/mnt/pfs/yuchen/groot/ft_fr_flow/checkpoint-20000')
model,proc=load_model(ck,'flow');tag=EmbodimentTag.resolve('SIMPLER_ENV_GOOGLE');mods=proc.modality_configs[tag.value]
loader=LeRobotEpisodeLoader(Path('/mnt/pfs/yuchen/groot/fractal_lerobot'),mods)
torch.manual_seed(20260910);out=[];last=None;max_label_difference=0.
for offset in range(0,len(selected),8):
    ix=selected[offset:offset+8];features=[]
    for i in ix:
        uid=int(z['episode_uid'][i]);ep=meta['episodes'][uid]
        if uid!=last: episode=loader[ep['ordinal']];last=uid
        item=extract_step_data(episode,int(z['step'][i]),mods,tag,allow_padding=False)
        features.append(proc([{'type':MessageType.EPISODE_STEP.value,'content':item}]))
    batch=proc.collator(features);inner=batch.get('inputs',batch)
    y=inner['action'][:,:8,:7].float().numpy();diff=float(np.max(np.abs(y-z['target'][ix])))
    assert diff<1e-6,diff;max_label_difference=max(max_label_difference,diff)
    inf=map_tensors({k:v for k,v in inner.items() if k!='action'},lambda t:t.cuda())
    samples=[]
    for rep in range(16):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model.get_action(inf)['action_pred'][:,:8,:7].float().cpu().numpy()
        samples.append(pred)
    os.environ['GROOT_FLOW_NOISE_SCALE']='0'
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        zero=model.get_action(inf)['action_pred'][:,:8,:7].float().cpu().numpy()
    os.environ['GROOT_FLOW_NOISE_SCALE']='1'
    arr=np.stack(samples,axis=1)
    for b,i in enumerate(ix):out.append(dict(index=int(i),samples=arr[b],zero=zero[b]))
    if offset%64==0:print(f'MATCHED_FLOW {offset}/{len(selected)}',flush=True)
np.savez_compressed(args.root/'matched_flow.npz',**{k:np.asarray([r[k] for r in out]) for k in out[0]})
(args.root/'matched_flow_done.json').write_text(json.dumps(dict(states=len(out),samples_per_state=16,
       max_label_difference=max_label_difference,checkpoint=str(ck),seed=20260910)))
print('MATCHED_FLOW_COMPLETE',len(out),flush=True)
import subprocess,sys
subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_fractal_push_flow.py')),str(args.root)],check=True)
