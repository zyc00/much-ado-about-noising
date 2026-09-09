"""CPU checks on the exact training objective, not a separate approximation."""
import math
import torch
from scale_score_ablation_launch import (
    ARMS, D, NU, SIGMA0, LOG_SIGMA0, EPS, SBIAS,
    kappa, sample_joint_t, scale_from_raw, objective,
)
from nu14_gaussian_aux_launch import gaussian_scale_aux


def main():
    torch.set_num_threads(4)
    torch.manual_seed(123)
    batch = 4
    mask = torch.zeros(batch, 8, 10)
    mask[:, :, :7] = 1
    pred = torch.randn_like(mask, requires_grad=True)
    raw = torch.zeros_like(mask, requires_grad=True)
    target = torch.randn_like(mask)
    generator = torch.Generator().manual_seed(456)
    global_rng = torch.get_rng_state().clone()
    samples = sample_joint_t(batch, 'cpu', generator)
    assert torch.equal(global_rng, torch.get_rng_state())
    for arm in ARMS:
        loss, info = objective(pred, raw, target, mask, arm, samples)
        torch.testing.assert_close(info['sigma'], torch.full((batch,), SIGMA0))
        gs, gp, gr = torch.autograd.grad(info['per'].sum(), (info['log_sigma'], pred, raw), retain_graph=True)
        assert torch.isfinite(loss) and torch.isfinite(gs).all()
        assert (gp[mask == 0] == 0).all() and (gr[mask == 0] == 0).all()
        if arm.startswith('se_'):
            z = (target-pred).masked_select(mask.bool()).reshape(batch,D)/info['sigma'][:,None]
            delta = z[None]-samples
            expected = 1-2/kappa()*((z[None]*delta).sum(-1)/delta.norm(dim=-1)).mean(0)
            torch.testing.assert_close(gs, expected)
            assert (gp.flatten(1).norm(dim=1) <= 2/(kappa()*info['sigma'])+1e-6).all()
        else:
            q = info['q']
            torch.testing.assert_close(gs, (NU/(NU+q)+1)*(1-q/D))
            gm_ht = torch.autograd.grad(info['ht'].mean(), pred, retain_graph=True)[0]
            torch.testing.assert_close(gp/batch, gm_ht)
            native_aux = gaussian_scale_aux(pred,raw,target,mask,SBIAS)[0]
            torch.testing.assert_close(native_aux, info['aux'].mean())
        # Exact first-update raw-scale derivative, including pooling.
        chain = (torch.sigmoid(torch.tensor(SBIAS))/SIGMA0) if arm=='se_softplus' else 1.
        torch.testing.assert_close(gr, gs[:,None,None]*mask/D*chain)

    # Both SE arms start with identical losses and mu gradients, but different
    # raw-scale gradients: this is the intended parameterization experiment.
    le, ie = objective(pred,raw,target,mask,'se_exp',samples)
    ls, is_ = objective(pred,raw,target,mask,'se_softplus',samples)
    torch.testing.assert_close(le,ls)
    torch.testing.assert_close(torch.autograd.grad(le,pred,retain_graph=True)[0],
                               torch.autograd.grad(ls,pred,retain_graph=True)[0])

    # Nonzero raw head values: Gaussian auxiliary detaches ONLY mu, not sigma.
    varying = torch.randn_like(raw, requires_grad=True)
    loss, info = objective(pred,varying,target,mask,'ht_exp_gaussaux1')
    gm, gr = torch.autograd.grad(info['aux'].sum(), (pred,varying), allow_unused=True, retain_graph=True)
    assert gm is None and gr.norm()>0
    gs = torch.autograd.grad(info['per'].sum(),info['log_sigma'],retain_graph=True)[0]
    torch.testing.assert_close(gs,(NU/(NU+info['q'])+1)*(1-info['q']/D))

    # Scaling a,mu,sigma by five leaves g_log_sigma unchanged (common MC draws).
    for arm in ('se_exp','ht_exp_gaussaux1'):
        values=[]
        for factor in (1.,5.):
            scale_raw=torch.full_like(raw,math.log(factor),requires_grad=True)
            _,info=objective(pred*factor,scale_raw,target*factor,mask,arm,samples)
            values.append(torch.autograd.grad(info['per'].sum(),info['log_sigma'])[0])
        torch.testing.assert_close(values[0],values[1],rtol=1e-5,atol=1e-5)

    # Statistical checks of the actual joint-t generator, nu=14,d=56.
    draws=sample_joint_t(1,'cpu',generator,count=100000)[:,0]
    assert abs(float(draws.square().mean())-NU/(NU-2))<.025
    square=draws[:,:2].square()
    cov=(square[:,0]*square[:,1]).mean()-square[:,0].mean()*square[:,1].mean()
    expected_cov=2*NU**2/((NU-2)**2*(NU-4))
    assert abs(float(cov)-expected_cov)<.05, (cov,expected_cov)
    print(f'PASS: sigma0={SIGMA0:.10f}, log_sigma0={LOG_SIGMA0:.10f}, kappa={kappa():.10f}')
    print('PASS: identical initialization, masking, joint-t RNG, detached auxiliary, analytic gradients, scale equivariance.')


if __name__=='__main__':
    main()
