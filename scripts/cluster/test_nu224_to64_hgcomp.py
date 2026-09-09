"""CPU checks for the 224-to-64 schedule and exact HG scale compensation."""
import math
import torch

from nu224_to64_hgcomp_launch import (
    D, NU_REFERENCE, NU_TARGET, compensated_objective, nu_for_completed_steps,
)


def main():
    assert nu_for_completed_steps(0) == NU_REFERENCE
    assert nu_for_completed_steps(6999) == NU_REFERENCE
    assert NU_TARGET < nu_for_completed_steps(7000) < NU_REFERENCE
    assert math.isclose(nu_for_completed_steps(11999), NU_TARGET)
    assert math.isclose(nu_for_completed_steps(20000), NU_TARGET)

    torch.manual_seed(123)
    batch = 8
    mask = torch.zeros(batch, 8, 10)
    mask[:, :, :7] = 1
    target = torch.randn_like(mask)
    for nu in (224.0, 160.0, 100.0, 64.0):
        prediction = torch.randn_like(mask, requires_grad=True)
        raw = torch.randn_like(mask, requires_grad=True) * 0.1
        loss, info = compensated_objective(prediction, raw, target, mask, nu)
        assert torch.isfinite(loss) and (info["d"] == D).all()
        assert (info["weight"] >= -1e-8).all()
        if nu == NU_REFERENCE:
            torch.testing.assert_close(info["weight"], torch.zeros_like(info["weight"]))
            torch.testing.assert_close(info["auxiliary"], torch.zeros_like(info["auxiliary"]))

        g_log, g_prediction = torch.autograd.grad(
            info["total"].sum(), (info["log_sigma"], prediction), retain_graph=True
        )
        q = info["q"].detach()
        expected_scale = NU_REFERENCE / (NU_REFERENCE + q) * (1 - q / D)
        torch.testing.assert_close(g_log, expected_scale, atol=2e-6, rtol=2e-5)

        # Auxiliary contributes no direct mean gradient.
        aux_mean = torch.autograd.grad(
            info["auxiliary"].sum(), prediction, allow_unused=True, retain_graph=True
        )[0]
        assert aux_mean is None
        assert torch.isfinite(g_prediction).all()
        assert (g_prediction[mask == 0] == 0).all()

    print("PASS schedule: 224 for 7k, log-linear decay to 64 over 5k, then hold")
    print("PASS exact per-example nu=224 log-sigma gradient; no auxiliary mu gradient or overshoot")


if __name__ == "__main__":
    main()
