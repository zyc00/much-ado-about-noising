"""CPU reanalysis of measured train-mode residuals; no policy updates.

Gradient budgets below are prediction-space per-example norms, NOT parameter
gradients or Adam steps. The full parameter gradients are in the source audit.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_nu_likelihood_gap import profile


def run():
    source=ROOT/'analysis/diagnostics/nu_gradient_switch/global1024'
    meta=json.loads((source/'protocol.json').read_text())['batches']['mixed_0']
    rules={'move_near':lambda t:'move' in t and 'near' in t,
           'close_drawer':lambda t:'close' in t and 'drawer' in t,
           'pick_coke':lambda t:'pick coke can' in t}
    cohorts={name:np.array([any(rule(t.lower()) for t in r['tasks']) for r in meta])
             for name,rule in rules.items()}
    result=dict(scope='Measured train-mode mixed Fractal residuals, 1024 disjoint episodes per batch; '
                      'first 768 density calibration / last 256 density test, NOT policy holdout.',
                gradient_scope='Per-example prediction-space norms, not parameter-gradient aggregation.',
                checkpoints={})
    for step in (10000,12000):
        path=source/f'raw_{step}_mixed_0_before.npz'
        with np.load(path) as z:
            q=z['q'].astype(float);s=z['s'].astype(float);sigma=z['sigma'].astype(float)
            dims=z['d'].astype(float);nu=float(z['nu']);episodes=z['episode']
        assert len(np.unique(episodes))==len(episodes)
        assert np.all(dims==56)
        np.testing.assert_allclose(q,s/sigma**2,rtol=1e-6)
        # For an isotropic joint likelihood, radius is a sufficient statistic
        # for fitting df/scale. An artificial direction preserves the exact NLL.
        radial=np.zeros((len(q),56));radial[:,0]=np.sqrt(q)
        row=dict(source=str(path),trained_nu=nu,n=len(q),
                 fit=profile(radial[:768],radial[768:]),
                 fixed_sigma_fit=profile(radial[:768],radial[768:],calibrate=False),
                 sigma_quantiles=np.quantile(sigma,[.01,.1,.5,.9,.99]).tolist(),
                 q_over_d_quantiles=np.quantile(q/56,[.01,.1,.5,.9,.99]).tolist(),
                 reweighting=[])
        low=q<=np.quantile(q,.1);high=q>=np.quantile(q,.9)
        for candidate in (1024,448,224,128,64,32,14,7):
            gate=(candidate+56)/(candidate+q)
            gmu=gate*np.sqrt(s)/sigma**2/56
            gsigma=(56-gate*q)/56
            row['reweighting'].append(dict(nu=candidate,
                gate_quantiles=np.quantile(gate,[.01,.1,.5,.9,.99]).tolist(),
                gate_below_half=float(np.mean(gate<.5)),
                gate_above_two=float(np.mean(gate>2)),
                low_q_decile_prediction_norm_share=float(gmu[low].sum()/gmu.sum()),
                high_q_decile_prediction_norm_share=float(gmu[high].sum()/gmu.sum()),
                prediction_norm_sum=float(gmu.sum()),
                log_sigma_gradient_mean=float(gsigma.mean()),
                log_sigma_gradient_abs_mean=float(np.abs(gsigma).mean()),
                cohorts={name:dict(n=int(ii.sum()),prediction_norm_share=float(gmu[ii].sum()/gmu.sum()),
                                  median_gate=float(np.median(gate[ii])))
                         for name,ii in cohorts.items()}))
        result['checkpoints'][str(step)]=row
    dest=Path(__file__).with_name('trainmode_radial.json')
    dest.write_text(json.dumps(result,indent=2)+'\n')
    for step,row in result['checkpoints'].items():
        print(step,'trained nu',row['trained_nu'],'joint fit',row['fit']['best_continuous'],
              'fixed sigma fit',row['fixed_sigma_fit']['best_continuous'])
        for r in row['reweighting']:
            if r['nu'] in (224,14,7):
                print(r)


if __name__=='__main__':run()
