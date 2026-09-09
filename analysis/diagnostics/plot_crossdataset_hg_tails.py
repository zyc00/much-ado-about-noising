"""Two-dataset HG tails, from measured summaries and saved cross-fitted residuals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parent/'crossdataset_hg_tails'
parser=argparse.ArgumentParser()
parser.add_argument('--bridge-groups',type=int,choices=[3,40],default=40)
args=parser.parse_args()
bridge_summary=ROOT/('bridge40/stats/bridge_hg_tail_summary.json' if args.bridge_groups==40 else 'bridge_hg_tail_summary.json')
datasets=[json.loads(path.read_text()) for path in [ROOT/'gr1_hg_tail_summary.json',bridge_summary]]
assert len(datasets[0]['tasks'])==24 and len(datasets[1]['tasks'])==args.bridge_groups
if args.bridge_groups==3:
    with np.load(ROOT/'bridge_hg_crossfitted_z.npz') as data:
        bridge_z=data['z'].copy()
        bridge_task_ids=data['task_id'].copy()
bridge_ids={task['task']:tid for tid,task in enumerate(datasets[1]['tasks'])}
colors=['#0072B2','#D55E00','#009E73']
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':8,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(7.2,4.65),gridspec_kw={'width_ratios':[1,1.65]})
fig.subplots_adjust(left=.16,right=.985,top=.88,bottom=.15,wspace=.30,hspace=.66)
manifest=[]
max_hi=max(t['ratio_ci'][1] for ds in datasets for t in ds['tasks'])
for row,ds in enumerate(datasets):
    left,right=axes[row]
    tasks=sorted(ds['tasks'],key=lambda t:t['tail_ratio'])
    n=len(tasks)
    # Predefined coverage: nearest rank to the 10th, 50th, 90th percentile,
    # not three tasks selected for the strongest tail departure.
    ranks=sorted(set(int(np.rint(q*(n-1))) for q in [.1,.5,.9]))
    color_for={rank:colors[i] for i,rank in enumerate(ranks)}
    if n==3:
        bridge_colors=dict(zip(['sweep into pile','open the drawer','close the drawer'],colors))
        color_for={rank:bridge_colors[tasks[rank]['task']] for rank in ranks}
    ratio=np.array([t['tail_ratio'] for t in tasks])
    old_thresholds=np.asarray(ds['thresholds'])
    thresholds=np.concatenate([np.arange(0,.5,.05),old_thresholds])
    origins=[]
    for rank,t in enumerate(tasks):
        selected=rank in color_for
        c=color_for.get(rank,'#BBC2C8')
        y=n-1-rank
        lo,hi=t['ratio_ci']
        left.plot([lo,hi],[y,y],c=c,lw=1.2 if selected else .6,alpha=1 if selected else .7)
        left.plot(t['tail_ratio'],y,'o',c=c,ms=3.6 if selected else 1.8,zorder=3)
        if row==0:
            with np.load(ROOT/'gr1_stats'/f'hg_task{t["task_id"]:02d}_crossfitted_z.npz') as data:
                z=data['z']
        elif n>3:
            with np.load(ROOT/'bridge40/stats'/f'hg_task{t["task_id"]:02d}_crossfitted_z.npz') as data:
                z=data['z']
        else:
            z=bridge_z[bridge_task_ids==bridge_ids[t['task']]]
        assert np.isfinite(z).all() and z.size==t['coordinates']
        absolute=np.sort(np.abs(z).ravel())
        empirical=(absolute.size-np.searchsorted(absolute,thresholds,side='right'))/absolute.size
        # The existing portion of every curve must remain unchanged.
        np.testing.assert_array_equal(empirical[10:],np.asarray(t['empirical_tail']))
        origins.append(float(empirical[0]))
        right.plot(thresholds,np.where(empirical>0,empirical,np.nan),
                   c=c,lw=1.45 if selected else .65,alpha=1 if selected else .45,
                   zorder=3 if selected else 1)
    left.axvline(1,c='.35',ls=(0,(4,3)),lw=.9,zorder=0)
    left.set(xlim=(0,max_hi*1.08),ylim=(-.8,n-.2),xlabel='3σ tail frequency / Gaussian')
    if n==3:
        nice={'sweep into pile':'Sweep into pile','open the drawer':'Open drawer','close the drawer':'Close drawer'}
        left.set_yticks([n-1-i for i in range(n)], [nice[t['task']] for t in tasks],fontsize=7)
    else:
        left.set_yticks([])
    left.grid(axis='x',color='.93',lw=.5)
    label=f'Bridge · {n} groups' if row==1 and n>3 else f'{ds["dataset"]} · {n} tasks'
    left.set_title(label,loc='left',fontweight='bold',fontsize=9,pad=9)
    left.text(.99,1.01,f'Median {np.median(ratio):.1f}×',ha='right',va='bottom',
              transform=left.transAxes,fontsize=7,color='.35')
    right.plot(thresholds,2*norm.sf(thresholds),c='.18',lw=1.25,ls=(0,(4,3)),zorder=4)
    right.axvline(3,c='.72',lw=.7,ls=':')
    right.set(yscale='log',ylim=(1e-5,1),xlim=(0,6),xticks=[0,1,2,3,4,5,6],
              xlabel=r'Standardized residual $|z|$',ylabel=r'Tail probability $P(|Z|>|z|)$')
    right.set_yticks([1,1e-1,1e-2,1e-3,1e-4,1e-5])
    right.grid(axis='y',color='.92',lw=.5)
    unit='groups' if row==1 and n>3 else 'tasks'
    right.text(.98,.96,f'{int((ratio>1).sum())}/{n} {unit} above Gaussian at 3σ',
               ha='right',va='top',transform=right.transAxes,fontsize=7)
    manifest.append(dict(dataset=ds['dataset'],task_count=n,
        empirical_probability_at_zero=origins,
        tail_curve_source='Saved cross-fitted residuals; exact empirical counts from zero; old thresholds verified unchanged.',
        tasks_above_gaussian=int((ratio>1).sum()),median_tail_ratio=float(np.median(ratio)),
        mean_tail_ratio=float(ratio.mean()),ratio_range=[float(ratio.min()),float(ratio.max())],
        selected_tasks=[tasks[k]['task'] for k in ranks],
        selection='Ranks nearest 10th, 50th and 90th percentile of measured tail ratio; all tasks remain visible.',
        nll_gain_range=[min(t['student_nll_gain'] for t in tasks),max(t['student_nll_gain'] for t in tasks)]))
fig.text(.16,.962,'a  Excess 3σ events',fontweight='bold',fontsize=10)
fig.text(.54,.962,'b  Standardized residual tails',fontweight='bold',fontsize=10)
fig.legend(handles=[Line2D([],[],c='#BBC2C8',lw=1,label='Other tasks'),
 Line2D([],[],c=colors[0],lw=1.6,label='Three highlighted tasks per dataset'),
 Line2D([],[],c='.18',lw=1.25,ls=(0,(4,3)),label='Gaussian')],
 loc='lower center',bbox_to_anchor=(.55,.02),frameon=False,ncol=3,fontsize=7)
for ext in ['png','pdf']:
    fig.savefig(ROOT/f'fig_crossdataset_hg_tails.{ext}',dpi=280)
plt.close(fig)
(ROOT/'crossdataset_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
