#!/usr/bin/env python3
"""Cross-episode, within-task kNN scale prediction using observations only.

Exploratory follow-up to the pre-specified coarse progress-bin diagnostic.
Neither query actions nor query residuals enter the scale predictor or bins.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from analyze_mse_stage_scale import metric_arrays, NAMES, COLORS, load_npz


def run(path):
    z=load_npz(path)
    r=metric_arrays(z,'pred_final')
    energy=np.mean(r*r,axis=1)
    states=z['state'].reshape(len(r),-1).astype(np.float64)
    predicted=np.full(len(r),np.nan);bins=np.full(len(r),-1)
    normalized=np.full(len(r),np.nan)
    rows=[]
    for task in sorted(set(z['task'])):
        for split in (0,1):
            cal=np.flatnonzero((z['task']==task)&(z['split']==split))
            query=np.flatnonzero((z['task']==task)&(z['split']!=split))
            X=states[cal];Q=states[query]
            std=np.std(X,axis=0)
            active=std>max(1e-6,np.max(std)*1e-4)
            assert active.any()
            scale=np.maximum(std[active],np.median(std[active])*.05)
            distance=cdist(Q[:,active]/scale,X[:,active]/scale,metric='sqeuclidean')
            k=max(5,min(30,int(np.ceil(np.sqrt(len(cal))))))
            k=min(k,len(cal))
            near=np.argpartition(distance,k-1,axis=1)[:,:k]
            pred=energy[cal][near].mean(axis=1)
            predicted[query]=pred
            # Rank query *predicted* scale, not query residuals. Within each task
            # and fold, thus between-task differences cannot produce the effect.
            order=np.argsort(pred,kind='stable')
            for b,inds in enumerate(np.array_split(order,3)):
                bins[query[inds]]=b
            baseline=energy[cal].mean()
            normalized[query]=energy[query]/baseline
            values=[float(np.sqrt(np.mean(energy[query][bins[query]==b])/baseline)) for b in range(3)]
            rows.append(dict(task=task,split=split,k=k,calibration_n=len(cal),query_n=len(query),
                low_mid_high=values,high_low_ratio=values[2]/values[0],
                rho=float(spearmanr(pred,energy[query]).statistic)))
    tasks=sorted(set(z['task']))
    task_values=[]
    for task in tasks:
        mask=z['task']==task
        task_values.append([np.sqrt(np.mean(normalized[mask&(bins==b)])) for b in range(3)])
    task_values=np.array(task_values)
    # Episode-cluster bootstrap of fixed cross-fitted predictions (conditional
    # on fitted neighborhood estimates; does not include refitting uncertainty).
    rng=np.random.default_rng(20260904)
    boot=[]
    by_task=[]
    for task in tasks:
        eps=sorted(set(z['episode'][z['task']==task]))
        sums=np.zeros((len(eps),3));counts=np.zeros_like(sums)
        for i,ep in enumerate(eps):
            for b in range(3):
                mask=(z['task']==task)&(z['episode']==ep)&(bins==b)
                sums[i,b]=np.sum(normalized[mask]);counts[i,b]=mask.sum()
        by_task.append((sums,counts))
    for _ in range(500):
        curves=[]
        for sums,counts in by_task:
            ii=rng.integers(len(sums),size=len(sums))
            curves.append(np.sqrt(sums[ii].sum(0)/np.maximum(counts[ii].sum(0),1)))
        boot.append(np.mean(curves,axis=0))
    boot=np.array(boot)
    curve=np.mean(task_values,axis=0)
    out=dict(method='within-task cross-episode kNN residual-energy estimate from proprioception',
        status='exploratory follow-up; no query residuals used for scale estimation or grouping',
        n=len(r),tasks=len(tasks),task_names=tasks,records=rows,task_curves=task_values.tolist(),
        curve=curve.tolist(),ci=np.quantile(boot,[.025,.975],axis=0).tolist(),
        high_low_ratio=float(curve[2]/curve[0]),
        high_low_ratio_ci=np.quantile(boot[:,2]/boot[:,0],[.025,.975]).tolist(),
        median_rho=float(np.nanmedian([v['rho'] for v in rows])))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    results={}
    for key in NAMES:
        path=args.root/(key+'.npz')
        if path.exists():
            results[key]=run(path)
            print(key,{k:results[key][k] for k in ('high_low_ratio','high_low_ratio_ci','median_rho')},flush=True)
    (args.root/'local_scale_summary.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix',
        'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
    fig,axes=plt.subplots(1,len(results),figsize=(5.5,2.05),sharey=True,squeeze=False)
    for ax,(key,s) in zip(axes[0],results.items()):
        curve=np.array(s['curve']);ci=np.array(s['ci'])
        ax.axhline(1,color='0.45',ls='--',lw=.6)
        ax.errorbar([0,1,2],curve,yerr=np.maximum(0,np.array([curve-ci[0],ci[1]-curve])),
            fmt='o-',color=COLORS[key],lw=1.5,ms=4,capsize=2,elinewidth=.8)
        ax.set_title(NAMES[key],pad=5)
        ax.set_xticks([0,1,2],['Low','Mid','High']);ax.set_xlim(-.35,2.35)
        ax.set_xlabel('Scale group',labelpad=3)
        ax.tick_params(length=2.5,pad=2)
        ax.text(.05,.95,f"{s['high_low_ratio']:.2f}×",transform=ax.transAxes,va='top',fontsize=8)
    axes[0,0].set_ylabel('Measured residual RMS\n(relative)',labelpad=3)
    top=max(np.max(s['ci'][1]) for s in results.values())
    bottom=min(np.min(s['ci'][0]) for s in results.values())
    axes[0,0].set_ylim(max(0,bottom-.15),top+.28)
    fig.subplots_adjust(left=.11,right=.98,bottom=.25,top=.77,wspace=.2)
    fig.savefig(args.root/'mse_local_scale.pdf')
    fig.savefig(args.root/'mse_local_scale.png',dpi=240)


if __name__=='__main__':main()
