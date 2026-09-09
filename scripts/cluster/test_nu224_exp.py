"""CPU checks for the exact nu=224 exp-scale training loss."""
import math
import torch

from nu224_exp_launch import (
    D, EPS, LOG_SIGMA0, NU, SBIAS, SIGMA0,
    exp_ht_objective, scales_from_raw,
)


def main():
    torch.manual_seed(123)
    batch = 4
    mask = torch.zeros(batch, 8, 10)
    mask[:, :, :7] = 1
    prediction = torch.randn_like(mask, requires_grad=True)
    target = torch.randn_like(mask)
    raw = torch.zeros_like(mask, requires_grad=True)

    sigma, log_sigma, reference, d = scales_from_raw(raw, mask)
    torch.testing.assert_close(sigma, torch.full((batch,), SIGMA0))
    torch.testing.assert_close(reference, torch.full((batch,), SIGMA0))
    torch.testing.assert_close(log_sigma, torch.full((batch,), LOG_SIGMA0))
    assert (d == D).all()

    loss, info = exp_ht_objective(prediction, raw, target, mask)
    manual = (
        0.5 * (NU + D) / D * torch.log1p(info["squared"] / sigma.square() / NU)
        + sigma.log()
    ).mean()
    torch.testing.assert_close(loss, manual)
    g_log, g_pred, g_raw = torch.autograd.grad(
        info["per"].sum(), (info["log_sigma"], prediction, raw), retain_graph=True
    )
    expected = NU / (NU + info["q"]) * (1 - info["q"] / D)
    torch.testing.assert_close(g_log, expected)
    assert torch.isfinite(g_pred).all() and torch.isfinite(g_raw).all()
    assert (g_pred[mask == 0] == 0).all() and (g_raw[mask == 0] == 0).all()
    torch.testing.assert_close(g_raw, g_log[:, None, None] * mask / D)

    # The calibrated offset is the inverse initialization map.  No accidental
    # reuse of the softplus raw bias as an exponential log-scale bias.
    assert math.isclose(math.exp(LOG_SIGMA0), SIGMA0, rel_tol=1e-12)
    assert not math.isclose(math.exp(SBIAS), SIGMA0, rel_tol=1e-3)
    assert EPS == 0.001
    print(f"PASS nu={NU:g}, d={D}, sigma0={SIGMA0:.10f}, log_sigma0={LOG_SIGMA0:.10f}")
    print("PASS calibrated exp initialization, masking, objective, and analytic gradients")


if __name__ == "__main__":
    main()
