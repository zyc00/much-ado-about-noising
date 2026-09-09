"""Numerical checks for nu schedule and scale-curvature calculations."""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).parent/'cluster'))
from nu_recipe_fork_launch import nu_value

for step,nu in [(7000,512),(8000,256),(9000,128),(10000,64),(11000,32),(12000,14),(13000,14)]:
    assert nu_value(step,'stair14')==nu
for arm,end in [('smooth14',14),('smooth7',7)]:
    assert abs(nu_value(7000,arm)-1024)<1e-9
    assert abs(nu_value(12000,arm)-end)<1e-9
    values=[nu_value(s,arm) for s in range(7000,13001)]
    assert all(a>=b for a,b in zip(values,values[1:]))
    assert max(abs(a-b)/a for a,b in zip(values,values[1:]))<.0011

d=56
for nu in [7,14,224,1024]:
    logsigma=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    loss=(.5*(nu+d)*torch.log1p(torch.tensor(float(d))/nu*torch.exp(-2*logsigma))+d*logsigma)/d
    first=torch.autograd.grad(loss,logsigma,create_graph=True)[0]
    curvature=torch.autograd.grad(first,logsigma)[0]
    assert abs(float(first.detach()))<1e-10
    expected=2*nu/(nu+d)
    assert abs(float(curvature)-expected)<1e-10
    q=1e14
    score=(d-(nu+d)*q/(nu+q))/d
    assert abs(score+nu/d)<1e-8
    print('nu',nu,'per-dimension log-sigma curvature',float(curvature))
print('Schedule and curvature checks passed.')

# Radial curvature switches sign at ||r||^2 / sigma^2 = nu.
# The frozen-weight Gaussian surrogate has the SAME first gradient: doing
# one ordinary gradient step with it is not a new optimization algorithm.
for nu in (7.,224.):
    r=torch.arange(1,57,dtype=torch.float64)/30
    r.requires_grad_(True)
    logscale=torch.tensor(-.2,dtype=torch.float64,requires_grad=True)
    q=(r*r).sum()*torch.exp(-2*logscale)
    true=.5*(nu+d)*torch.log1p(q/nu)+d*logscale
    weight=((nu+d)/(nu+q)).detach()
    surrogate=.5*weight*q+d*logscale
    first=torch.autograd.grad(true,(r,logscale),retain_graph=True,create_graph=True)
    second=torch.autograd.grad(surrogate,(r,logscale),retain_graph=True)
    for a,b in zip(first,second):torch.testing.assert_close(a,b)
    direction=r.detach()/r.detach().norm()
    hv=torch.autograd.grad((first[0]*direction).sum(),r)[0]
    measured=(hv*direction).sum()
    s=(r.detach()*r.detach()).sum();sig2=torch.exp(2*logscale.detach())
    expected=(nu+d)*(nu*sig2-s)/(nu*sig2+s)**2
    torch.testing.assert_close(measured,expected)
print('Radial curvature and frozen-weight surrogate gradient checks passed.')
