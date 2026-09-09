"""Explicitly selected exploratory examples; never an independent validation claim."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts'))
from analyze_mse_stage_scale import load_npz, metric_arrays, task_matrices, available_mean

OUT = Path(__file__).resolve().parent/'crossdataset_residual'
SOURCE = OUT/'expanded_summary.json'
original = SOURCE.read_bytes()
data = json.loads(original)
stacks = ['gr1', 'pi05', 'bridge', 'fractal']
names = ['RoboCasa-GR1', 'LIBERO', 'Bridge', 'Fractal']
colors = ['#7657A5', '#009E73', '#D97924', '#C99714']
short_names = {
    'gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA': 'Tray to tiered shelf',
    'turn on the stove and put the moka pot on it': 'Turn on stove; place moka pot',
    'upright hot sauce bottle cardboard fence': 'Upright hot-sauce bottle',
    'pick green can from top shelf of fridge': 'Pick green can from fridge',
}
manifest = dict(source_sha256=hashlib.sha256(original).hexdigest(),
    scope='Training-demonstration fit diagnostic, not policy-held-out.',
    selection=('Exploratory effect-based selection, using BOTH episode halves. For each task, '
               'fixed-seed split episodes in half; rank common progress bins in each half, '
               'measure other-half RMS ratio for top/bottom three bins. Select the task with '
               'the largest minimum of the two ratios, requiring at least eight common bins. '
               'This selection is not an independent validation or a prevalence estimate.'),
    seed=20260908, normalization='Common full-task RMS scale for both halves and all example episodes.',
    datasets={})

for stack in stacks:
    if stack in ('gr1', 'pi05'):
        source = REPO/'analysis/paper/mse_scale'/f'{stack}.npz'
        z = load_npz(source)
        energy = np.mean(metric_arrays(z, 'pred_final')**2, axis=1)
        rows = [(t, m, np.asarray(sorted(set(z['episode'][z['task']==t]))))
                for t, m, _ in task_matrices(z, energy)]
    else:
        source = OUT/f'{stack}_episode_energy.npz'
        z = load_npz(source)
        rows = list(zip(z['tasks'], z['energy'].astype(float), z['episodes']))
    eligible = {r['task']: r for r in data['per_task'] if r['dataset']==stack}
    assert {str(t) for t,_,_ in rows} == set(eligible)
    candidates = []
    for task, matrix, eps in rows:
        full = np.sqrt(available_mean(matrix, axis=0))
        expected = np.asarray(eligible[str(task)]['stage_rms'], dtype=float)
        assert np.allclose(full, expected, rtol=1e-5, equal_nan=True)
        perm = np.random.default_rng(manifest['seed']).permutation(len(matrix))
        groups = [perm[:len(matrix)//2], perm[len(matrix)//2:]]
        folds = np.asarray([np.sqrt(available_mean(matrix[g], axis=0)) for g in groups])
        valid = np.isfinite(folds).all(axis=0) & (folds>0).all(axis=0)
        if valid.sum()<8:
            continue
        ratios = []
        for a,b in (folds, folds[::-1]):
            ids = np.flatnonzero(valid)
            order = ids[np.argsort(a[ids])]
            ratios.append(float(np.sqrt(np.mean(b[order[-3:]]**2)/np.mean(b[order[:3]]**2))))
        candidates.append((min(ratios), str(task), matrix, eps, groups, folds, full, ratios))
    score,task,matrix,eps,groups,folds,full,ratios = sorted(candidates,key=lambda t:(-t[0],t[1]))[0]
    scale = np.sqrt(np.nanmean(full**2))
    # Three typical episode SHAPES in this selected task, not the most extreme trajectories.
    episode_rms = np.sqrt(matrix)
    shapes = episode_rms / np.sqrt(np.nanmean(matrix, axis=1))[:,None]
    template = full/scale
    distances = np.nanmean((shapes-template[None,:])**2,axis=1)
    chosen = np.argsort(distances, kind='stable')[:3]
    def safe(x):
        a=np.asarray(x)
        if a.ndim==1: return [float(v) if np.isfinite(v) else None for v in a]
        return [safe(row) for row in a]
    manifest['datasets'][stack] = dict(task=task, source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selection_score=score, both_fold_ratios=ratios,
        selection_candidate_count=len(candidates), all_task_count=len(rows),
        episode_halves=[[int(eps[i]) for i in g] for g in groups],
        normalized_folds=safe(folds/scale), normalized_full_profile=safe(full/scale),
        representative_episode_rule='Three smallest shape distances to the selected full-task profile; exploratory examples.',
        representative_episodes=[int(eps[i]) for i in chosen],
        representative_curves=safe(episode_rms[chosen]/scale),
        common_normalizer=float(scale))

plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'],
    'mathtext.fontset':'stix', 'font.size':8, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.linewidth':.65, 'pdf.fonttype':42})
x = np.arange(5,100,10)
for mode in ('replicated', 'trajectories'):
    fig = plt.figure(figsize=(7.2,5.5))
    gs = fig.add_gridspec(4,2,width_ratios=[1,1.9],left=.22,right=.98,
                         top=.86,bottom=.20,wspace=.32,hspace=.65)
    for i,(stack,name,color) in enumerate(zip(stacks,names,colors)):
        rows = [r for r in data['per_task'] if r['dataset']==stack]
        info = manifest['datasets'][stack]
        ax,bx = fig.add_subplot(gs[i,0]),fig.add_subplot(gs[i,1])
        rho = np.asarray([r['label_residual_rho'] for r in rows])
        grid = np.linspace(-1,1,401)
        kde = gaussian_kde(rho)
        density = kde(grid)+kde(-2-grid)+kde(2-grid)
        ax.fill_between(grid,0,.5*density/density.max(),color=color,alpha=.1,lw=0)
        ax.plot(grid,.5*density/density.max(),color=color,lw=1.2)
        ax.axvline(0,color='.7',lw=.6,ls=(0,(3,3)))
        ax.text(.98,.98,f'median {np.median(rho):.2f}\nmean {rho.mean():.2f}',
                transform=ax.transAxes,ha='right',va='top',fontsize=7,color='.35')
        ax.set(xlim=(-1,1),ylim=(-.28,1),yticks=[],xticks=[-1,0,1])
        ax.spines['left'].set_visible(False)
        unit='groups' if stack in ('bridge','fractal') else 'tasks'
        ax.set_ylabel(f'{name}\n{len(rows)} {unit}',rotation=0,ha='right',va='center',labelpad=8)
        if mode=='replicated':
            for row in rows:
                y=np.asarray(row['stage_rms'],dtype=float)
                bx.plot(x,y/np.sqrt(np.nanmean(y*y)),color='#ABB2BB',lw=.4,
                        ls=(0,(2,2)),alpha=.07 if len(rows)>100 else .18,zorder=0)
            folds=np.asarray(info['normalized_folds'],dtype=float)
            bx.plot(x,folds[0],color='#0072B2',lw=1.4,ls=(0,(4,2)),zorder=3)
            bx.plot(x,folds[1],color='#D55E00',lw=1.6,marker='o',ms=2.6,zorder=4)
        else:
            for curve,c in zip(info['representative_curves'],['#0072B2','#009E73','#D55E00']):
                bx.plot(x,np.asarray(curve,dtype=float),color=c,lw=1.3,marker='o',ms=2.5)
            bx.plot(x,np.asarray(info['normalized_full_profile'],dtype=float),color='.25',lw=1,ls='--')
        bx.set_title(short_names[info['task']],fontsize=7.5,loc='left',pad=5)
        bx.axhline(1,color='.65',lw=.6,ls=(0,(3,3)),zorder=0)
        bx.set(xlim=(0,100),ylim=(0,2.8),xticks=[0,50,100],yticks=[0,1,2])
        bx.grid(axis='y',color='.93',lw=.5)
        if i<3:
            ax.tick_params(labelbottom=False)
            bx.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r'Action–residual correlation $\rho$',fontsize=8)
            bx.set_xlabel('Episode progress (%)',fontsize=8)
    fig.text(.22,.955,'a  Action magnitude',fontsize=10,fontweight='bold')
    fig.text(.52,.955,'b  Selected stage-change examples',fontsize=10,fontweight='bold')
    fig.text(.22,.915,'All qualifying tasks / groups',fontsize=7,color='.4')
    fig.text(.52,.915,'Relative residual RMS · exploratory selection',fontsize=7,color='.4')
    if mode=='replicated':
        handles=[Line2D([],[],color='#0072B2',lw=1.4,ls='--',label='6 demonstrations: half A'),
                 Line2D([],[],color='#D55E00',lw=1.5,marker='o',ms=3,label='6 demonstrations: half B'),
                 Line2D([],[],color='#ABB2BB',lw=.7,ls='--',label='All task profiles')]
    else:
        handles=[Line2D([],[],color=c,lw=1.3,label=f'Example episode {j+1}')
                 for j,c in enumerate(['#0072B2','#009E73','#D55E00'])]
        handles.append(Line2D([],[],color='.25',lw=1,ls='--',label='12-demo task profile'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.6,.035),ncol=2,
               frameon=False,fontsize=7,handlelength=2.2)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'all_datasets_progress_selected_{mode}.{ext}',dpi=240)
    plt.close(fig)
assert SOURCE.read_bytes()==original
(OUT/'selected_progress_examples_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
print(json.dumps({s:{k:manifest['datasets'][s][k] for k in ('task','both_fold_ratios','representative_episodes')}
                  for s in stacks},indent=2))
