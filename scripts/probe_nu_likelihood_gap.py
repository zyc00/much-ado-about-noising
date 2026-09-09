"""Compare marginal and joint Student-t df on the SAME frozen residual vectors.

Includes common-scale and independent-coordinate synthetic controls. All
likelihood comparisons retain nu-dependent normalization constants. Episode
holdout is for density calibration, NOT held-out policy training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln
from scipy.stats import multivariate_t

GRID = np.array([1.5,2,3,4,5,7,10,14,20,32,56,112,224,448,896,1792,4096,16384,np.inf])


def nll_q(q, d, nu, scale2):
    if np.isinf(nu):
        return .5*(d*np.log(2*np.pi*scale2)+q/scale2)
    return (gammaln(nu/2)-gammaln((nu+d)/2)+.5*d*np.log(nu*np.pi*scale2)
            +.5*(nu+d)*np.log1p(q/(nu*scale2)))


def scale2_mle(q, d, nu):
    """Solve the monotone scale score, avoiding slow low-nu fixed points."""
    if np.isinf(nu):
        return float(q.mean()/d)
    positive=q[q>0]
    if not len(positive):
        raise ValueError('All-zero residuals have no finite positive scale MLE.')
    lo=float(np.log(positive.min()/d)-30)
    hi=float(np.log(positive.max()/d)+2)
    def score(log_s2):
        return float(np.mean((nu+d)*q/(nu*np.exp(log_s2)+q))-d)
    if score(lo)<=0:
        raise ValueError('Residual atom at zero makes the positive scale fit degenerate.')
    return float(np.exp(brentq(score,lo,hi,xtol=1e-12)))


def profile(x, y, calibrate=True):
    d = x.shape[1]
    q = np.sum(x*x,axis=1)
    qt = np.sum(y*y,axis=1)
    rows = []
    def fit(nu):
        s2=scale2_mle(q,d,nu) if calibrate else 1.
        return dict(nu=float(nu) if np.isfinite(nu) else 'Gaussian',
                         scale=float(np.sqrt(s2)),train_nll=float(nll_q(q,d,nu,s2).mean()/d),
                         test_nll=float(nll_q(qt,d,nu,s2).mean()/d))
    for nu in GRID:
        rows.append(fit(nu))
    best = min(rows,key=lambda r:r['train_nll'])
    # Refine around the best finite grid point; use calibration likelihood only.
    finite=[r for r in rows if r['nu']!='Gaussian']
    index=min(range(len(finite)),key=lambda i:finite[i]['train_nll'])
    lo=.3 if index==0 else finite[index-1]['nu']
    hi=1e5 if index==len(finite)-1 else finite[index+1]['nu']
    opt=minimize_scalar(lambda v:fit(float(np.exp(v)))['train_nll'],
                        bounds=(np.log(lo),np.log(hi)),method='bounded',options={'xatol':1e-7})
    continuous=fit(float(np.exp(opt.x)))
    if rows[-1]['train_nll']<continuous['train_nll']:continuous=dict(rows[-1])
    return dict(best=best,best_continuous=continuous,profile=rows)


def geometry(x):
    rms = np.sqrt(np.mean(x*x,axis=0)).clip(1e-8)
    z = x/rms
    corr = np.corrcoef(z,rowvar=False)
    eig = np.linalg.eigvalsh(np.nan_to_num(corr)).clip(0)
    q = np.mean(x*x,axis=1)
    return dict(d=x.shape[1],effective_correlation_dimension=float(eig.sum()**2/(eig@eig)),
                radial_squared_cv=float(np.std(q)/np.mean(q)),
                radial_squared_quantiles=np.quantile(q,[.01,.1,.5,.9,.99]).tolist(),
                marginal_excess3=float((np.abs(z)>3).mean()/.0026997960632601913))


def fit_views(x,y):
    d=x.shape[1]
    rms=np.sqrt(np.mean(x*x,axis=0)).clip(1e-8)
    # Fixed train-estimated Gaussian second moment, shrinkage avoids unstable
    # inverse directions. This is a covariance-model diagnostic, not full t EM.
    c=x.T@x/len(x)
    shrink=.05
    c=(1-shrink)*c+shrink*np.diag(np.diag(c))
    c+=np.eye(d)*max(float(np.trace(c)/d)*1e-6,1e-10)
    ch=np.linalg.cholesky(c)
    transforms={
        'isotropic': (x,y,0.),
        'diagonal': (x/rms,y/rms,float(np.log(rms).sum()/d)),
        'full_cov_shrink05': (solve_triangular(ch,x.T,lower=True).T,
                              solve_triangular(ch,y.T,lower=True).T,
                              float(np.log(np.diag(ch)).sum()/d)),
    }
    result={}
    for name,(a,b,jac) in transforms.items():
        fit=profile(a,b)
        for row in fit['profile']:
            row['train_nll']+=jac
            row['test_nll']+=jac
        fit['best_continuous']['train_nll']+=jac
        fit['best_continuous']['test_nll']+=jac
        result[name]=fit
    marginal=profile((x/rms).reshape(-1,1),(y/rms).reshape(-1,1))
    # This pooled marginal diagnostic shares a single df/scale across axes;
    # it is not evidence of a common latent scale across the full vector.
    result['pooled_standardized_marginal']=marginal
    result['fixed_sigma_isotropic']=profile(x,y,calibrate=False)
    result['geometry_train']=geometry(x)
    return result


def synthetic(seed=20260908):
    rng=np.random.default_rng(seed);n=12000;d=56;nu=7.
    normal=rng.normal(size=(n,d))
    common=normal/np.sqrt(rng.chisquare(nu,size=(n,1))/nu)
    independent=normal/np.sqrt(rng.chisquare(nu,size=(n,d))/nu)
    blocks=normal/np.sqrt(np.repeat(rng.chisquare(nu,size=(n,7))/nu,8,axis=1))
    result={}
    for name,x in [('common_scale_t7',common),('independent_t7',independent),('seven_block_t7',blocks)]:
        a,b=x[:8000],x[8000:]
        result[name]=dict(joint=profile(a,b),marginal=profile(a.reshape(-1,1),b.reshape(-1,1)))
    test=common[:10]
    expected=-multivariate_t.logpdf(test,df=7,shape=np.eye(d))
    np.testing.assert_allclose(nll_q((test*test).sum(1),d,7,1),expected,atol=1e-10)
    assert result['common_scale_t7']['joint']['best']['nu']==7
    return result


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    syn=synthetic()
    (args.output/'synthetic.json').write_text(json.dumps(syn,indent=2))
    for name,rows in syn.items():
        print('SYNTHETIC',name,'joint',rows['joint']['best'],'marginal',rows['marginal']['best'],flush=True)
    if args.synthetic_only:return
    for ds in args.datasets:
        root=Path('/mnt/pfs/yuchen/ht_sigma_context_20260908')/ds
        with np.load(root/'probe.npz') as z:
            print('KEYS',ds,{k:z[k].shape for k in z.files},flush=True)
            residual=(z['prediction'].astype(np.float64)-z['target'].astype(np.float64))
            sigma=z['sigma'].astype(np.float64)
            episode=z['episode_uid'] if 'episode_uid' in z.files else z['episode']
            task=z['task_id']
        rng=np.random.default_rng(20260908)
        test=np.zeros(len(episode),bool)
        for tid in np.unique(task):
            eps=np.unique(episode[task==tid]);rng.shuffle(eps)
            test |= (task==tid)&np.isin(episode,eps[:max(1,len(eps)//3)])
        views={'all_training_channels':residual.reshape(len(residual),-1)}
        if ds!='gr1': views['continuous_only']=residual[:,:,:6].reshape(len(residual),-1)
        result=dict(dataset=ds,source=str(root/'probe.npz'),protocol=json.loads((root/'protocol.json').read_text()),
                    train_states=int((~test).sum()),test_states=int(test.sum()),
                    calibration_holdout='Per-task disjoint episode split, 2/3 calibration, 1/3 test; not policy holdout.',
                    covariance_protocol='Fixed calibration second-moment whitening, 5% offdiagonal shrinkage; not joint Student-t covariance EM.',
                    views={})
        for name,r in views.items():
            for own in [False,True]:
                key=name+('_own_sigma' if own else '_fixed_scale')
                x=r/sigma[:,None] if own else r
                fit=fit_views(x[~test],x[test])
                jac_train=float(np.log(sigma[~test]).mean()) if own else 0.
                jac_test=float(np.log(sigma[test]).mean()) if own else 0.
                for model,f in fit.items():
                    if model in ('geometry_train','pooled_standardized_marginal'):continue
                    for row in f['profile']:
                        row['train_nll']+=jac_train;row['test_nll']+=jac_test
                    f['best_continuous']['train_nll']+=jac_train
                    f['best_continuous']['test_nll']+=jac_test
                result['views'][key]=fit
                print('DATA_FIT',ds,key,{k:v['best'] for k,v in fit.items() if 'best' in v},flush=True)
                (args.output/f'{ds}.json').write_text(json.dumps(result,indent=2))
        print('DATASET_DONE',ds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--synthetic-only',action='store_true')
    p.add_argument('--datasets',nargs='+',default=['fractal','bridge','gr1'])
    run(p.parse_args())
