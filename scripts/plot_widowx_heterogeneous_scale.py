#!/usr/bin/env python3
"""General MSE, progress, matched Flow, and episode-cross-fitted scale evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

SEED=20260906
COLORS=['#0072B2','#D55E00','#009E73']
LABELS=['Sweep into pile','Open drawer','Close drawer']


def load_matched(root):
    merged, metadata={},{}
    for mode in ('mse','flow'):
        paths=sorted(root.glob(f'{mode}_rank*.npz'))
        if len(paths)!=4: raise ValueError(f'Expected four {mode} parts, found {len(paths)}')
        parts={}
        for path in paths:
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']))
                assert meta['objective']==mode
                assert 'specialist' not in meta['checkpoint'] and '_q' not in meta['checkpoint']
                metadata[mode]=meta
                for key in z.files:
                    if key!='metadata': parts.setdefault(key,[]).append(z[key])
        data={k:np.concatenate(v) for k,v in parts.items()}
        order=np.lexsort((data['step'],data['episode'],data['task_id']))
        merged[mode]={k:v[order] for k,v in data.items()}
    mse,flow=merged['mse'],merged['flow']
    for key in ('task_id','episode','step','length'): np.testing.assert_array_equal(mse[key],flow[key])
    np.testing.assert_allclose(mse['target'],flow['target'],atol=1e-6)
    assert mse['target'].shape[1:]==(8,6)
    return mse,flow,metadata


def relative(x,task):
    result=np.empty(len(x),dtype=float)
    for tid in np.unique(task):
        keep=task==tid
        result[keep]=x[keep]/np.exp(np.log(np.maximum(x[keep],1e-12)).mean())
    return result


def rho_within(x,y,task):
    rx,ry=np.empty(len(x)),np.empty(len(y))
    for tid in np.unique(task):
        keep=task==tid
        for src,dst in ((x,rx),(y,ry)):
            r=rankdata(src[keep]); dst[keep]=(r-r.mean())/max(r.std(),1e-12)
    return float(np.mean(rx*ry))


def correlation(x,y,task,episode):
    groups=[[np.flatnonzero((task==tid)&(episode==ep)) for ep in np.unique(episode[task==tid])]
            for tid in np.unique(task)]
    rng=np.random.default_rng(SEED)
    boot=[]
    for _ in range(1500):
        idx=np.concatenate([g[i] for g in groups for i in rng.integers(len(g),size=len(g))])
        boot.append(rho_within(x[idx],y[idx],task[idx]))
    return dict(rho=rho_within(x,y,task),ci95=np.quantile(boot,[.025,.975]).tolist(),
        per_task=[float(spearmanr(x[task==t],y[task==t]).statistic) for t in np.unique(task)])


def likelihood(mse):
    """Scale fitting uses frozen policy inputs/features/predictions, never labels at test time."""
    energy=np.mean(mse['residual'].astype(float)**2,axis=(1,2))
    task,episode=mse['task_id'],mse['episode']
    pred=mse['prediction'].astype(float); state=mse['state'].astype(float)
    state=state[:,np.std(state,axis=0)>1e-10]
    descriptors=np.column_stack([state,pred.mean(axis=1),pred.std(axis=1),
        np.sqrt(np.mean(pred**2,axis=(1,2))),np.eye(3)[task]])
    folds=np.empty(len(energy),dtype=int)
    rng=np.random.default_rng(SEED)
    for tid in np.unique(task):
        for i,ep in enumerate(rng.permutation(np.unique(episode[task==tid]))): folds[episode==ep]=i%5
    scores=np.full((len(energy),2),np.nan); variances=np.full_like(scores,np.nan)
    for fold in range(5):
        train,test=folds!=fold,folds==fold
        base=np.asarray([energy[train&(task==t)].mean() for t in task])
        pca=PCA(n_components=20,svd_solver='randomized',random_state=SEED)
        pca.fit(mse['embedding'][train])
        features=np.column_stack([descriptors,pca.transform(mse['embedding'])])
        model=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=32,
            max_features=1.0,n_jobs=4,random_state=SEED+fold)
        model.fit(features[train],energy[train]/base[train])
        variances[test,0]=base[test]
        variances[test,1]=base[test]*np.maximum(model.predict(features[test]),1e-6)
        for j in range(2):
            v=variances[test,j]; scores[test,j]=-.5*(np.log(2*np.pi*v)+energy[test]/v)
    assert np.isfinite(scores).all()
    episode_scores=np.asarray([scores[episode==ep].mean(axis=0) for ep in np.unique(episode)])
    rng=np.random.default_rng(SEED+1)
    boot=episode_scores[rng.integers(len(episode_scores),size=(10000,len(episode_scores)))].mean(axis=1)
    means=episode_scores.mean(axis=0)
    return dict(means=means.tolist(),ci95=np.quantile(boot,[.025,.975],axis=0).tolist(),
        gain=float(means[1]-means[0]),gain_ci95=np.quantile(boot[:,1]-boot[:,0],[.025,.975]).tolist(),
        per_task_gain=[float(np.mean(scores[task==t,1]-scores[task==t,0])) for t in range(3)],
        features='proprioception, frozen MSE predicted-action descriptors, training-fold PCA of frozen MSE observation features',
        baseline='one fixed variance per task',validation='five-fold disjoint episodes for scale fitting',
        estimator='ExtraTrees, 300 trees, leaf size 32, direct residual energy regression',
        scores=scores,variances=variances,folds=folds)


def progress_curves(mse):
    task,episode=mse['task_id'],mse['episode']
    energy=np.mean(mse['residual'].astype(float)**2,axis=(1,2))
    stage=np.minimum(9,(mse['progress']*10).astype(int)); curves=[]
    for tid in range(3):
        eps=np.unique(episode[task==tid]); matrix=np.full((len(eps),10),np.nan)
        for i,ep in enumerate(eps):
            for b in range(10):
                keep=(episode==ep)&(stage==b)
                if keep.any(): matrix[i,b]=energy[keep].mean()
        n=np.sum(np.isfinite(matrix),axis=0); bins=np.flatnonzero(n>=8); matrix=matrix[:,bins]
        rng=np.random.default_rng(SEED+tid)
        boot=np.sqrt(np.nanmean(matrix[rng.integers(len(eps),size=(3000,len(eps)))],axis=1))
        curves.append(dict(x=(bins+.5)*10,mean=np.sqrt(np.nanmean(matrix,axis=0)),
            ci95=np.nanquantile(boot,[.025,.975],axis=0),episodes_per_bin=n[bins]))
    return curves


def draw(mse,flow,metadata,output):
    task,episode=mse['task_id'],mse['episode']
    label=np.sqrt(np.mean(mse['target'].astype(float)**2,axis=(1,2)))
    error=np.sqrt(np.mean(mse['residual'].astype(float)**2,axis=(1,2)))
    samples=flow['samples'].astype(float)
    spread=np.sqrt(np.mean((samples-samples.mean(axis=1,keepdims=True))**2,axis=(1,2,3)))
    np.testing.assert_allclose(spread,flow['spread'],rtol=1e-6)
    stats_a=correlation(label,error,task,episode);stats_c=correlation(label,spread,task,episode)
    ll=likelihood(mse);curves=progress_curves(mse)
    mpl.rcParams.update({'font.family':'serif','font.serif':['Nimbus Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,
        'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,4,figsize=(5.5,2.08));xr=relative(label,task)
    for ax,values,stats,title,ylabel in [
        (axs[0],error,stats_a,'a  General MSE','Residual RMS (relative)'),
        (axs[2],spread,stats_c,'c  Matched Flow','Sample spread (relative)')]:
        yr=relative(values,task)
        for tid in range(3):
            keep=task==tid
            ax.scatter(xr[keep],yr[keep],s=3.5,color=COLORS[tid],marker=['o','s','^'][tid],
                alpha=.16,linewidths=0,rasterized=True)
        ax.set_xscale('log');ax.set_yscale('log')
        ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(.12,6);ax.set_ylim(.12,6)
        ax.set_xticks([.25,1,4],['0.25','1','4']);ax.set_yticks([.25,1,4],['0.25','1','4'])
        ax.set_xlabel('Label RMS (relative)');ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',fontweight='bold',pad=5)
        ax.text(.05,.96,rf"$\rho={stats['rho']:.2f}$",transform=ax.transAxes,va='top',fontsize=8,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    ax=axs[1]
    for tid,curve in enumerate(curves):
        ax.fill_between(curve['x'],curve['ci95'][0],curve['ci95'][1],color=COLORS[tid],alpha=.1,linewidth=0)
        ax.plot(curve['x'],curve['mean'],color=COLORS[tid],lw=1.3,marker=['o','s','^'][tid],ms=2.5,mew=.4,mec='white')
    ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.set_ylim(bottom=0)
    ax.set_xlabel('Episode progress (%)');ax.set_ylabel('MSE residual RMS')
    ax.set_title('b  Task progress',loc='left',fontweight='bold',pad=5)
    ax.text(.96,.04,'Shading: 95% CI',transform=ax.transAxes,ha='right',fontsize=5.8,color='.4')
    ax=axs[3];means=np.asarray(ll['means']);ci=np.asarray(ll['ci95'])
    for i,color in enumerate(['#7A7A7A','#6B4C9A']):
        ax.errorbar(i,means[i],yerr=[[means[i]-ci[0,i]],[ci[1,i]-means[i]]],fmt='o',
            color=color,ms=5,capsize=3,elinewidth=1.2,mew=1)
        ax.annotate(f'{means[i]:.3f}',(i,ci[1,i]),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7)
    ax.set_xlim(-.45,1.45);ax.set_xticks([0,1],['Fixed\nscale','Input-dep.\nscale'])
    ax.set_ylabel('Test log likelihood (nat/dim)')
    ax.set_title('d  Gaussian fit',loc='left',fontweight='bold',pad=5)
    gap=max(ci.max()-ci.min(),.04);ax.set_ylim(ci.min()-.35*gap,ci.max()+.75*gap)
    ax.text(.5,.97,rf"$\Delta={ll['gain']:+.3f}$",transform=ax.transAxes,ha='center',va='top',fontsize=8)
    ax.text(.5,.09,'Higher is better',transform=ax.transAxes,ha='center',fontsize=5.8,color='.4')
    ax.text(.5,.02,'Bars: 95% CI',transform=ax.transAxes,ha='center',fontsize=5.8,color='.4')
    for ax in axs:
        ax.grid(axis='y',color='.92',lw=.5,zorder=0);ax.tick_params(length=2,pad=2)
    handles=[Line2D([0],[0],color=COLORS[t],marker=['o','s','^'][t],lw=1.2,ms=3,label=LABELS[t]) for t in range(3)]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.48,1.005),fontsize=7)
    fig.subplots_adjust(left=.073,right=.985,bottom=.245,top=.78,wspace=.79)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.025)
    fig.savefig(output.with_suffix('.png'),dpi=320,bbox_inches='tight',pad_inches=.025);plt.close(fig)
    arrays={k:ll.pop(k) for k in ['scores','variances','folds']}
    np.savez_compressed(output.parent/'likelihood_crossfit.npz',**arrays,episode=episode,task=task)
    summary=dict(metadata=metadata,states=len(task),episodes=len(np.unique(episode)),panel_a=stats_a,panel_c=stats_c,panel_d=ll,
        panel_b=[{k:np.asarray(v).tolist() for k,v in row.items()} for row in curves],
        flow_mean_error_correlation=correlation(label,np.sqrt(np.mean(flow['residual'].astype(float)**2,axis=(1,2))),task,episode))
    output.with_name(output.name+'_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('panel_b','metadata')},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,default=Path('analysis/paper/widowx_heterogeneous_scale/raw'))
    p.add_argument('--output',type=Path,default=Path('analysis/paper/widowx_heterogeneous_scale/fig_widowx_heterogeneous_scale'))
    args=p.parse_args();mse,flow,meta=load_matched(args.raw);draw(mse,flow,meta,args.output)


if __name__=='__main__':main()
