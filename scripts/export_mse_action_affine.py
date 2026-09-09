#!/usr/bin/env python3
"""Export original action-coordinate inverse normalization, without model loading."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stack');ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();out={}
    if args.stack=='robomimic':
        from omegaconf import OmegaConf
        from mip.datasets.robomimic_dataset import make_dataset
        for name in ('tool_hang','transport'):
            z=np.load(args.out/(name+'.npz'))
            meta=json.loads(str(z['metadata']))
            ds=make_dataset(OmegaConf.create(meta['config']['task']))
            a=ds.normalizer['action'];A=meta['config']['task']['act_dim']
            offset=a.unnormalize(np.zeros((1,A),np.float32))[0]
            slope=a.unnormalize(np.ones((1,A),np.float32))[0]-offset
            out[name]=dict(offset=offset.tolist(),slope=slope.tolist(),
                magnitude_channels=[0,1,2] if A==10 else [0,1,2,10,11,12],
                magnitude_description='predicted translation command RMS in original controller units')
    elif args.stack=='pi05':
        from safetensors.numpy import load_file
        stats=load_file('/mnt/pfs/yuchen/pi05/run_mse/checkpoints/030000/pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors')
        lo=stats['action.q01'];hi=stats['action.q99']
        out['pi05']=dict(offset=((hi+lo)/2).tolist(),slope=((hi-lo)/2).tolist(),
            magnitude_channels=[0,1,2],magnitude_description='predicted translation command RMS in original controller units')
    elif args.stack=='gr1':
        from gr00t.data.state_action.state_action_processor import StateActionProcessor
        from gr00t.data.embodiment_tags import EmbodimentTag
        ck=Path('/mnt/pfs/yuchen/groot/ft_mse/checkpoint-60000')
        kwargs=json.loads((ck/'processor_config.json').read_text())['processor_kwargs']
        proc=StateActionProcessor(modality_configs=kwargs['modality_configs'],
            statistics=json.loads((ck/'statistics.json').read_text()),
            use_relative_action=kwargs['use_relative_action'],
            use_percentiles=kwargs['use_percentiles'])
        tag=EmbodimentTag.resolve('ROBOCASA_GR1_TABLETOP').value
        order=['left_arm','right_arm','left_hand','right_hand','waist']
        # Query the action-only unnormalizer, preserving its exact treatment of
        # relative-action quantiles. Supply zero states so relative joints stay offsets.
        params=proc.norm_params[tag]['action']
        zero={k:np.zeros((1,8,int(params[k]['dim'])),np.float32) for k in order}
        one={k:np.ones_like(v) for k,v in zero.items()}
        state={k:np.zeros((1,1,int(params[k]['dim'])),np.float32) for k in order}
        a0=proc.unapply_action(zero,embodiment_tag=tag,state=state)
        a1=proc.unapply_action(one,embodiment_tag=tag,state=state)
        offset=np.concatenate([a0[k][0,0] for k in order])
        slope=np.concatenate([a1[k][0,0]-a0[k][0,0] for k in order])
        out['gr1']=dict(offset=offset.tolist(),slope=slope.tolist(),
            magnitude_channels=list(range(26)),
            magnitude_description='predicted arm and hand joint-offset RMS in radians; absolute waist excluded from magnitude only')
    else:raise ValueError(args.stack)
    path=args.out/f'action_affine_{args.stack}.json'
    path.write_text(json.dumps(out,indent=2))
    print('ACTION_AFFINE',json.dumps(out),flush=True)


if __name__=='__main__':main()
