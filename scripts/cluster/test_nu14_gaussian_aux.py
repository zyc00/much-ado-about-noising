"""Gradient isolation and masked-reduction tests; no network/data needed."""
import torch
from nu14_gaussian_aux_launch import gaussian_scale_aux


def main():
    torch.manual_seed(20260908)
    pred = torch.randn(3, 8, 7, requires_grad=True)
    raw = torch.randn(3, 8, 7, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.ones_like(pred)
    mask[1, -2:] = 0
    aux, sigma, squared, d = gaussian_scale_aux(pred, raw, target, mask, -.5093)
    gp, gr = torch.autograd.grad(aux, (pred, raw), allow_unused=True, retain_graph=True)
    assert gp is None and gr is not None and gr.norm() > 0
    assert (gr[mask == 0] == 0).all()
    gs = torch.autograd.grad(aux, sigma, retain_graph=True)[0]
    torch.testing.assert_close(gs*sigma, (d-squared/sigma.square())/d.sum())
    nu = 14.
    live_squared = ((pred-target).square()*mask).sum((1,2))
    ht = (.5*(nu+d)*torch.log1p(live_squared/(nu*sigma.square()))+d*sigma.log()).sum()/d.sum()
    ht_gp = torch.autograd.grad(ht, pred, retain_graph=True)[0]
    total_gp = torch.autograd.grad(ht+aux, pred, retain_graph=True)[0]
    torch.testing.assert_close(ht_gp, total_gp)
    # Shared features still receive Gaussian gradients exclusively via sigma.
    x = torch.randn(3, 8, 5, requires_grad=True)
    w_mu = torch.randn(5, 7, requires_grad=True)
    w_sigma = torch.randn(5, 7, requires_grad=True)
    other = gaussian_scale_aux(x@w_mu, x@w_sigma, target, mask, -.5093)[0]
    gx, gm, gs = torch.autograd.grad(other, (x, w_mu, w_sigma), allow_unused=True)
    assert gm is None and gx.norm() > 0 and gs.norm() > 0
    print('PASS: detached mean; live sigma/shared features; native masked scale; combined mean gradient unchanged.')


if __name__ == '__main__':
    main()
