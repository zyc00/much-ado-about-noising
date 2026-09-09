"""Predeclared action-defined groups; no late-episode or sigma-based selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    z=np.load(args.root/'probe.npz');meta=json.loads((args.root/'protocol.json').read_text())
    y=z['target'];pred=z['prediction'];raw=z['raw_target'];r=pred-y
    sigma=z['sigma'];w=z['weight'];gate=z['gate'];uid=z['episode_uid'];task=z['task_id']
    # Raw action x is robot-forward; this is an ACTION proxy, not a contact annotation.
    trans=raw[:,:,:3].mean(1);amp=np.linalg.norm(trans,axis=1)
    forward=(trans[:,0]>0)&(trans[:,0]>=np.linalg.norm(trans[:,1:],axis=1))
    first_forward=(raw[:,0,0]>0)&(raw[:,0,0]>=np.linalg.norm(raw[:,0,1:3],axis=1))
    strong=np.zeros(len(uid),bool);low=strong.copy();high=strong.copy()
    for ep in np.unique(uid):
        ix=uid==ep;lo,hi=np.quantile(amp[ix],[.25,.75])
        low[ix]=amp[ix]<=lo;high[ix]=amp[ix]>=hi
    strong=forward&high
    # Output-space gradient per sample, ignoring common minibatch averaging.
    output_grad_norm=w*np.sqrt(z['S'])/56
    energy_gripper=np.sum(r[:,:,6]**2,axis=1)
    def stat(m):
        return dict(n=int(m.sum()),episodes=int(len(np.unique(uid[m]))),
                    sigma_median=float(np.median(sigma[m])),gate_median=float(np.median(gate[m])),
                    weight_median=float(np.median(w[m])),rms_median=float(np.median(np.sqrt(z['S'][m]/56))),
                    gradient_norm_median=float(np.median(output_grad_norm[m])),
                    action_mean_translation_norm_median=float(np.median(amp[m])),
                    first_x_residual_mean=float(np.mean(r[m,0,0])),
                    first_x_underprediction_fraction=float(np.mean(r[m,0,0]<0)),
                    gate_below_half_fraction=float(np.mean(gate[m]<.5)),
                    gripper_error_energy_share=float(energy_gripper[m].sum()/z['S'][m].sum()))
    groups=dict(all=np.ones(len(uid),bool),forward=forward,nonforward=~forward,
                strong_forward=strong,other=~strong,amplitude_q1=low,amplitude_q4=high,
                first_forward=first_forward)
    rng=np.random.default_rng(9183)
    def paired(a,b):
        ratios={k:[] for k in ['sigma','gate','weight','gradient_norm']}
        values=dict(sigma=sigma,gate=gate,weight=w,gradient_norm=output_grad_norm)
        for ep in np.unique(uid):
            ia=a&(uid==ep);ib=b&(uid==ep)
            if min(ia.sum(),ib.sum())<3: continue
            for k,v in values.items():ratios[k].append(float(np.median(v[ia])/np.median(v[ib])))
        out={}
        for k,vals in ratios.items():
            vals=np.array(vals)
            if not len(vals):out[k]=None;continue
            boots=np.median(vals[rng.integers(0,len(vals),size=(3000,len(vals)))],axis=1)
            out[k]=dict(episodes=len(vals),median_ratio=float(np.median(vals)),
                        ci95=np.quantile(boots,[.025,.975]).tolist())
        return out
    result=dict(definitions=dict(forward='positive robot-forward mean x >= lateral mean-vector norm, over 8-step raw label chunk',
                                strong='forward AND top quartile of mean-translation magnitude within the same episode',
                                weight='(nu+d)/(nu*sigma^2 + sum_squared_normalized_residual); output gradient = weight * residual / d',
                                caveat='Action-defined proxy, not confirmed contact. Frozen eval-mode checkpoint, not historical training gradients.'),
                overall={k:stat(m) for k,m in groups.items()},
                paired_forward_vs_nonforward=paired(forward,~forward),
                paired_strong_vs_other=paired(strong,~strong),
                paired_q4_vs_q1=paired(high,low),
                by_task={meta['names'][int(t)]:dict(forward=stat(forward&(task==t)),nonforward=stat((~forward)&(task==t)),
                           strong=stat(strong&(task==t)),other=stat((~strong)&(task==t))) for t in np.unique(task)})
    (args.root/'analysis.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(args.root/'derived.npz',forward=forward,strong=strong,amp=amp,
                        sigma=sigma,gate=gate,weight=w,gradient_norm=output_grad_norm,
                        episode_uid=uid,task=task,step=z['step'])
    # Visual sanity check: first two qualifying episodes per task, peak forward
    # label within each episode; selection does not use sigma or residual.
    frames=np.load(args.root/'frames.npz')['images'];chosen=[]
    for t in np.unique(task):
        eps=[e for e in np.unique(uid[task==t]) if np.sum(strong&(uid==e))>=3][:2]
        for ep in eps:
            ix=np.flatnonzero(strong&(uid==ep));chosen.append(ix[np.argmax(trans[ix,0])])
    canvas=Image.new('RGB',(1000,380),color='white');draw=ImageDraw.Draw(canvas)
    for j,ix in enumerate(chosen):
        x=(j%3)*330;y0=(j//3)*190
        canvas.paste(Image.fromarray(frames[ix]).resize((180,144)),(x,y0+40))
        draw.text((x,y0),f"{meta['names'][int(task[ix])]} ep {z['episode'][ix]} step {z['step'][ix]}\nsigma {sigma[ix]:.3f} gate {gate[ix]:.3f} W {w[ix]:.1f}",fill='black')
    canvas.save(args.root/'label_selected_push_examples.jpg',quality=90)
    print(json.dumps({k:v for k,v in result.items() if k not in ['by_task','definitions']},indent=2),flush=True)


if __name__=='__main__':main()
