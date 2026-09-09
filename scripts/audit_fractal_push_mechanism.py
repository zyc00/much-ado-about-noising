"""Separate local scale weighting, coordinate errors, directional bias and nu effects.
All counterfactuals freeze predictions/sigma: these are not retraining results.
"""
import json,sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]);z=np.load(root/'probe.npz')
a=json.loads((root/'visual_push_intervals.json').read_text())
ck=Path('/mnt/pfs/yuchen/groot/ft_fr_nu224/checkpoint-20000')
stats=json.loads((ck/'statistics.json').read_text());tag=next(k for k in stats if 'google' in k.lower())
cfg=json.loads((ck/'processor_config.json').read_text())['processor_kwargs']
assert cfg['use_percentiles']
keys=['x','y','z','roll','pitch','yaw','gripper']
lo=np.array([stats[tag]['action'][k]['q01'][0] for k in keys]);hi=np.array([stats[tag]['action'][k]['q99'][0] for k in keys])
y=z['target'];p=z['prediction'];r=p-y;raw=z['raw_target'];sg=z['sigma'];S=z['S'];w=z['weight'];uid=z['episode_uid'];t=z['step']
# Decoder physical outputs are clipped using the original percentile scales.
py=(np.clip(y,-1,1)+1)/2*(hi-lo)+lo
pp=(np.clip(p,-1,1)+1)/2*(hi-lo)+lo

def desc(ix):
    rr=r[ix];e=(rr**2).sum((0,1));mask=raw[ix,0,0]>0
    target=py[ix,0,:3];pred=pp[ix,0,:3]
    norm=np.linalg.norm(target,axis=1);unit=target/np.maximum(norm[:,None],1e-8)
    dot=(pred*unit).sum(1)
    # Positive dL/d log(sigma) => a free scale would move downward under GD.
    siggrad=1-w[ix]*S[ix]/56
    return dict(n=int(ix.sum()),sigma_median=float(np.median(sg[ix])),
                residual_rms=float(np.sqrt(np.mean(S[ix]/56))),
                predicted_sigma_rms=float(np.sqrt(np.mean(sg[ix]**2))),
                standardized_energy_mean=float(np.mean(S[ix]/(56*sg[ix]**2))),
                sigma_gradient_log_mean=float(siggrad.mean()),
                sigma_gradient_log_positive_fraction=float((siggrad>0).mean()),
                coordinate_error_energy_shares=dict(zip(keys,(e/e.sum()).tolist())),
                gripper_error_by_horizon=(rr[:,:,6]**2).sum(0).tolist(),
                first_action_gripper_share_of_total_error=float((rr[:,0,6]**2).sum()/e.sum()),
                last_four_gripper_share_of_total_error=float((rr[:,4:,6]**2).sum()/e.sum()),
                fixed_sigma_remove_gripper_weight_ratio=float(np.median(
                    ((224+48)/(224*sg[ix]**2+(rr[:,:,:6]**2).sum((1,2))))/w[ix])),
                raw_first_x_mean=float(raw[ix,0,0].mean()),
                first_x_label_clipped_fraction=float(np.mean(np.abs(y[ix,0,0])>=1)),
                first_forward_target_mean=float(target[:,0].mean()),
                first_forward_prediction_mean=float(pred[:,0].mean()),
                first_forward_bias_mean=float((pred[:,0]-target[:,0]).mean()),
                first_forward_mse=float(np.mean(rr[:,0,0]**2)),
                forward_positive_n=int(mask.sum()),
                forward_positive_prediction_target_ratio=float(pred[mask,0].sum()/max(target[mask,0].sum(),1e-8)),
                projected_prediction_target_ratio=float(dot.sum()/max(norm.sum(),1e-8)),
                first_forward_underprediction_fraction=float((rr[:,0,0]<0).mean()),
                abs_forward_error_mean=float(np.abs(rr[:,0,0]).mean()),
                signed_forward_error_mean=float(rr[:,0,0].mean()),
                weighted_signed_forward_error_mean=float((w[ix]*rr[:,0,0]).mean()),
                weights_by_nu={str(nu):dict(weight_median=float(np.median((nu+56)/(nu*sg[ix]**2+S[ix]))),
                   median_ratio_to_224=float(np.median(((nu+56)/(nu*sg[ix]**2+S[ix]))/w[ix]))) for nu in [1024,224,56,14,7]})

rows=[];sens={}
for k,(start,end) in a['windows'].items():
    ep=int(k);inside=(uid==ep)&(t>=start)&(t<=end);before=(uid==ep)&(t<start)
    rows.append(dict(uid=ep,episode=int(z['episode'][inside][0]),before=desc(before),push=desc(inside)))
    ratios=[]
    for da in [-2,-1,0,1,2]:
        for db in [-2,-1,0,1,2]:
            m=(uid==ep)&(t>=start+da)&(t<=end+db);b=(uid==ep)&(t<start+da)
            if min(m.sum(),b.sum())<3:continue
            ratios.append(float(np.median(w[m])/np.median(w[b])))
    sens[k]=dict(min_ratio=min(ratios),max_ratio=max(ratios),fraction_below_one=float(np.mean(np.array(ratios)<1)),variants=len(ratios))
result=dict(rows=rows,boundary_sensitivity=sens,
            physical_normalization=dict(q01=lo.tolist(),q99=hi.tolist()),
            caveats=['Visual intervals are approximate, labels are normalized/clipped as in training.',
                     'Counterfactual nu weights keep predictions and sigma fixed; not training outcomes.',
                     'Output gradients are not shared-parameter gradients or historical training updates.'])
(root/'mechanism_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(example_19034=next(x for x in rows if x['episode']==19034),boundary_sensitivity=sens),indent=2))
