"""Muon optimizer (Jordan et al. 2024): momentum orthogonalized by Newton-Schulz.
Applied to parameters with ndim >= 2 (matrices/convs, flattened to 2D); scalars,
biases, and embeddings should use AdamW. Minimal single-file implementation."""
import torch


def zeropower_via_newtonschulz5(G, steps: int = 5):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=group["momentum"])
                else:
                    g = buf
                g2 = g.reshape(g.size(0), -1)
                u = zeropower_via_newtonschulz5(g2, steps=group["ns_steps"]).reshape_as(p)
                scale = max(1, p.size(0) / g2.size(1)) ** 0.5
                p.add_(u, alpha=-group["lr"] * scale)
        return loss


class MuonWithAdamW(torch.optim.Optimizer):
    """Muon for ndim>=2 params, AdamW for the rest, behind one optimizer interface."""

    def __init__(self, named_params, muon_lr=0.02, adamw_lr=1e-4, weight_decay=1e-5,
                 momentum=0.95):
        params = list(named_params)
        muon_p = [p for _, p in params if p.ndim >= 2]
        adamw_p = [p for _, p in params if p.ndim < 2]
        self.muon = Muon(muon_p, lr=muon_lr, momentum=momentum) if muon_p else None
        self.adamw = torch.optim.AdamW(adamw_p, lr=adamw_lr, weight_decay=weight_decay) if adamw_p else None
        # expose combined param_groups for LR schedulers (scale both by the same factor)
        self.param_groups = ([] if self.muon is None else self.muon.param_groups) + \
                            ([] if self.adamw is None else self.adamw.param_groups)
        self.defaults = {"lr": adamw_lr}
        self.state = {}

    def zero_grad(self, set_to_none=True):
        if self.muon: self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw: self.adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.muon: self.muon.step()
        if self.adamw: self.adamw.step()
        return loss

    def state_dict(self):
        return {"muon": self.muon.state_dict() if self.muon else None,
                "adamw": self.adamw.state_dict() if self.adamw else None}

    def load_state_dict(self, sd):
        if self.muon and sd.get("muon"): self.muon.load_state_dict(sd["muon"])
        if self.adamw and sd.get("adamw"): self.adamw.load_state_dict(sd["adamw"])
