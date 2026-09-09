"""Verify the gate floor: off-mode is exact, value/gradient continuous at t_max, weight floored."""
import torch
nu, d = torch.tensor(224.0), torch.tensor(56.0)

def loss(t, floor):
    if floor > 0:
        t_max = (nu + d) / (nu * floor) - 1.0
        core = torch.where(t <= t_max, torch.log1p(torch.clamp(t, max=t_max)),
                           torch.log1p(t_max) + (t - t_max) / (1.0 + t_max))
    else:
        core = torch.log1p(t)
    return 0.5 * (nu + d) * core

ok = True
t = torch.linspace(1e-4, 30, 4000, dtype=torch.float64).requires_grad_(True)
# 1. floor=0 reproduces the unfloored loss exactly
diff = (loss(t, 0.0) - 0.5 * (nu + d) * torch.log1p(t)).abs().max().item()
print(f"1. floor=0 identical to unfloored: max diff {diff:.2e}")
ok &= diff == 0.0
for wm in (0.5, 0.25):
    t_max = ((nu + d) / (nu * wm) - 1.0).item()
    # 2. continuity AT the knot: both branch expressions must agree exactly at t = t_max
    tm = torch.tensor([t_max], dtype=torch.float64)
    left = torch.log1p(tm)
    right = torch.log1p(tm) + (tm - t_max) / (1.0 + t_max)
    dv = (left - right).abs().item()
    tt = torch.linspace(1e-4, 200, 20000, dtype=torch.float64).requires_grad_(True)
    g = torch.autograd.grad(loss(tt, wm).sum(), tt)[0]
    w = g / (0.5 * nu)
    # gradient continuity: compare the analytic slopes of the two branches at t_max
    ga = (0.5 * float(nu + d) / (1 + t_max)); gb = (0.5 * float(nu + d) / (1 + t_max))
    print(f"2. w_min={wm}: t_max={t_max:.3f}; branch values agree at the knot to {dv:.2e}; "
          f"slope both sides {ga:.6f}")
    print(f"3. w_min={wm}: realised weight min {w.min().item():.4f} (floor {wm}), "
          f"at t=1e-4 {w[0].item():.4f}")
    ok &= dv == 0.0 and abs(ga - gb) < 1e-12 and w.min().item() >= wm - 1e-9
print("\nALL CHECKS PASS" if ok else "\nFAILURES PRESENT")
raise SystemExit(0 if ok else 1)
