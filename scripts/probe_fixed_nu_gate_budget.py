"""Separate the joint-t gate from inverse-variance weighting on frozen policies."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(root):
    result={}
    for ds in ('fractal','bridge','gr1'):
        directory=root/ds
        meta=json.loads((directory/'protocol.json').read_text());nu=meta['ht_df']
        with np.load(directory/'probe.npz') as z:
            r=z['prediction'].astype(float)-z['target'].astype(float)
            sigma=z['sigma'].astype(float)
        s=np.sum(r*r,axis=(1,2));d=int(np.prod(r.shape[1:]));q=s/sigma**2
        gate=(nu+d)/(nu+q)
        # These are output-space norms with the common 1/d factor omitted.
        # Gaussian uses the SAME predictions and sigma, not another HG policy.
        gaussian=np.sqrt(s)/sigma**2
        ht=gate*gaussian
        top=q>=np.quantile(q,.99)
        result[ds]=dict(source=str(directory/'probe.npz'),checkpoint=meta['checkpoint'],nu=nu,d=d,n=len(q),
            scope='Frequency-selected training-demonstration probe, not policy holdout or rollout distribution.',
            gate_below_half=float((gate<.5).mean()),gate_below_quarter=float((gate<.25).mean()),
            gate_quantiles=np.quantile(gate,[.01,.1,.5,.9,.99]).tolist(),
            negative_radial_curvature_fraction={str(v):float((q>v).mean()) for v in (nu,14,7)},
            sigma_quantiles=np.quantile(sigma,[.1,.5,.9]).tolist(),
            output_gradient_scope='Sum of per-example prediction-gradient norms, NOT parameter gradients.',
            ht_to_same_sigma_gaussian_norm_sum=float(ht.sum()/gaussian.sum()),
            top_q_one_percent_gaussian_norm_share=float(gaussian[top].sum()/gaussian.sum()),
            top_q_one_percent_ht_norm_share=float(ht[top].sum()/ht.sum()))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/mnt/pfs/yuchen/ht_sigma_context_20260908'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=run(args.root)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
