"""Tube-funnel toy (user-specified geometry, 2026-07):
  tube (thin, straight)      y in (0.6, 1.0]    <- SPEED noise here
  thin funnel                y in (0.25, 0.6]   <- DIRECTION noise here
  (flip machinery removed per user 2026-07-21; see tube_flip_r*.json archives)
  insert band (current)      y in (0.05, 0.25]  <- dogleg fine structure
  dock                       y < 0.05, DOCK_TOL
Funnel mouth = tube half-width, so a thinner tube thins the funnel.
The two uncertainties are spatially disjoint and toggle independently:
NOISE_TYPE = none | dir | speed | both.
Momentum (MOM) is a world constant (hidden velocity); the none-session
stays deterministic given the start, so clean parity is well-defined.
Insert band scrapes (precision zone); upstream bands slide.
"""
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = int(os.environ.get("H_OVR", "8"))
G = np.array([0.0, 0.0])
Y_TUBE_TOP, Y_TUBE_LO, Y_STR_LO, Y_FUN_LO, Y_DOCK = 1.4, 0.9, 0.5, 0.25, 0.05
# geometry v2.1: longer tube (speed band 0.5 tall) and longer wide Z-turn
# band (Y_STR_LO, Y_TUBE_LO], 0.4 tall, between tube and funnel
X_OFF = float(os.environ.get("X_OFF", "0.06"))   # Z lateral offset (6cm)
Z_BOX = 0.03                                      # slide-box margin around [0, X_OFF]
# canonical path is STRAIGHT; the Z detour is an aleatoric draw (zturn
# session only), parallel to pace/tremor draws. T=-1 disables the ramps.
T1_FIX, T2_FIX = -1.0, -1.0
ZW = float(os.environ.get("ZW", "0.08"))          # turn ramp height (smooth jog)
NU = 2.0
CLIP = 0.08
DOCK_TOL = float(os.environ.get("DOCK_TOL", "0.002"))
DOGLEG = float(os.environ.get("DOGLEG", "0.02"))
HW_TUBE = float(os.environ.get("HW_TUBE", "0.02"))
HW_COR = float(os.environ.get("HW_COR", "0.005"))
HW_CHAMFER = float(os.environ.get("HW_CHAMFER", "0.006"))  # insert mouth 1mm wider
HW_DOCK = float(os.environ.get("HW_DOCK", "0.002"))
MOM = float(os.environ.get("MOM", "0.5"))
PROC = float(os.environ.get("PROC", "0.0005"))  # world process noise (0.5mm/step)
# scripted-data axis: demo-side process noise (0 = deterministic script,
# no corrective coverage, no local label variation); eval keeps PROC
PROC_DEMO = float(os.environ.get("PROC_DEMO", str(PROC)))
HOLD = int(os.environ.get("HOLD", "0"))          # settle-analog: hold steps at Y_HOLD
Y_HOLD = 0.45                                     # hold station (upper funnel)
S_HI = float(os.environ.get("S_HI", "0.07"))      # dir tremor scale (funnel)
T1S3 = float(os.environ.get("T1S3", "1.2"))       # pace jitter scale (tube)
PAUSE_P = float(os.environ.get("PAUSE_P", "0.12"))
NEP = int(os.environ.get("NEP", "40"))
STEPS = int(os.environ.get("STEPS", "12000"))
WIDTH = int(os.environ.get("WIDTH", "64"))
ROLL_BUDGET = int(os.environ.get("ROLL_BUDGET", "450"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3").split(",")]
ARMS = tuple(os.environ.get("ARMS", "l2,ht").split(","))
AS_DEF = int(os.environ.get("AS", "2"))  # replan interval (mm scale needs 1-2)
FLOW_NS = int(os.environ.get("FLOW_NS", "8"))  # flow inference integration steps
NOISE_TYPE = os.environ.get("NOISE_TYPE", "none")  # none|dir|speed|both|zturn
ACT_D, OBS_D = 2, 4  # obs = (x, y, vel_x, vel_y): proprioception


def center(y):
    if Y_DOCK <= y < Y_FUN_LO:
        return DOGLEG * float(np.sin(np.pi * (Y_FUN_LO - y) / (Y_FUN_LO - Y_DOCK)))
    return 0.0


def half_width(y):
    if y > Y_STR_LO:
        return HW_TUBE
    if y > Y_FUN_LO:
        f = (y - Y_FUN_LO) / (Y_STR_LO - Y_FUN_LO)
        return HW_COR + f * (HW_TUBE - HW_COR)
    if y > Y_DOCK:
        f = (y - Y_DOCK) / (Y_FUN_LO - Y_DOCK)
        return HW_DOCK + f * (HW_CHAMFER - HW_DOCK)
    return HW_DOCK


def wall(p):
    """insert band scrapes (precision zone); upstream bands slide (demos
    ride the thin funnel walls, canonical-cell convention)."""
    if Y_DOCK < p[1] <= Y_FUN_LO:
        if abs(p[0] - center(p[1])) > half_width(p[1]):
            p[0] = float("nan")  # scrape/crash
    elif in_z_band(p[1]):
        if p[1] > Y_STR_LO + 0.05:  # enclosure contact = failure (mid/upper band)
            if p[0] < -Z_BOX or p[0] > X_OFF + Z_BOX:
                p[0] = float("nan")  # boundary contact (no demo goes within 3cm)
        else:  # bottom lead-in strip: sliding taper into the funnel mouth
            f = (p[1] - Y_STR_LO) / 0.05
            lo = -HW_TUBE + f * (-Z_BOX + HW_TUBE)
            hi = HW_TUBE + f * (X_OFF + Z_BOX - HW_TUBE)
            p[0] = float(np.clip(p[0], lo, hi))
    elif Y_FUN_LO < p[1] <= Y_TUBE_TOP:  # free space above the tube mouth
        c, hw = 0.0, half_width(p[1])
        p[0] = float(np.clip(p[0], c - hw, c + hw))
    return p


def in_speed_band(y):
    return Y_TUBE_LO < y <= Y_TUBE_TOP  # tube only, not the approach


def in_z_band(y):
    return Y_STR_LO < y <= Y_TUBE_LO


def x_target(y, t1, t2):
    """Z center-line: smooth ramp out to X_OFF and back (turns are ramps
    over ZW of height, not single-step jumps)."""
    r1 = np.clip((t1 - y) / ZW, 0.0, 1.0)
    r2 = np.clip((t2 + ZW - y) / ZW, 0.0, 1.0)
    return X_OFF * (r1 - r2) + center(y)


def in_dir_band(y):
    # margin above the scrape zone: last kicks stay in slide territory so
    # the demonstrator re-centers off the wall before the precision band
    return Y_FUN_LO + 0.06 < y <= Y_STR_LO


def noise_band(y):
    if NOISE_TYPE == "dir":
        return in_dir_band(y)
    if NOISE_TYPE == "speed":
        return in_speed_band(y)
    if NOISE_TYPE == "both":
        return in_dir_band(y) or in_speed_band(y)
    if NOISE_TYPE == "zturn":
        return in_z_band(y)
    return in_dir_band(y)  # placeholder mask for none


def expert_action(p, t1=T1_FIX, t2=T2_FIX):
    if p[1] > Y_TUBE_TOP:  # approach: fast clean transit into the tube mouth
        vy = -min(CLIP, 0.3 * (p[1] - Y_TUBE_TOP) + 0.02)
    elif in_z_band(p[1]):
        vy = -0.03  # Z band is wide; tube pace
    else:  # pace scales with clearance: slow-careful where it is tight
        vy = -min(1.5 * half_width(p[1]), 0.5 * max(p[1], 0.0) + 0.001)
    return np.clip(np.array([x_target(p[1] + vy, t1, t2) - p[0], vy]),
                   -CLIP, CLIP)


def gen_episode(rng):
    p = np.array([rng.uniform(-0.2, 0.2), rng.uniform(1.6, 1.7)])
    states, acts, tail = [], [], H + 2
    docked = False
    vel = np.zeros(2)
    if NOISE_TYPE == "zturn":
        t1 = rng.uniform(0.68, 0.86)
        t2 = rng.uniform(0.52, 0.58)
    else:
        t1, t2 = T1_FIX, T2_FIX
    held = 0
    for _ in range(300):
        a = expert_action(p, t1, t2)
        if HOLD > 0 and held < HOLD and p[1] <= Y_HOLD:
            a = np.array([0.0, 0.0])
            held += 1
        if NOISE_TYPE in ("speed", "both") and in_speed_band(p[1]):
            if rng.rand() < PAUSE_P:
                eta = 0.0
            else:
                eta = float(np.clip(1.0 + T1S3 * rng.standard_t(2), 0.0, 3.5))
            a[1] = float(np.clip(a[1] * eta, -CLIP, CLIP))
            a[0] = float(np.clip(center(p[1] + a[1]) - p[0], -CLIP, CLIP))
        if NOISE_TYPE in ("dir", "both") and in_dir_band(p[1]):
            a[0] = float(np.clip(a[0] + S_HI * rng.standard_t(2), -CLIP, CLIP))
        states.append(np.concatenate([p, vel]))
        a[0] = float(np.clip(a[0] - MOM * vel[0], -CLIP, CLIP))
        a[1] = float(np.clip(a[1] - MOM * vel[1], -CLIP, CLIP))
        acts.append(a.copy())
        p_new = wall(p[:2] + a[:2] + MOM * vel + PROC_DEMO * rng.randn(2))
        vel = p_new - p
        p = p_new
        if np.isnan(p[0]):
            raise RuntimeError("demo scraped in insert band — knobs broken")
        docked = docked or np.linalg.norm(p - G) < DOCK_TOL
        if docked:
            tail -= 1
            if tail <= 0:
                break
    return states, acts


def build_dataset(rng):
    X, Y, Yc, AL = [], [], [], []
    for _ in range(NEP):
        for _retry in range(50):
            try:
                st, ac = gen_episode(rng)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("demo generation keeps scraping")
        for t in range(len(st) - H):
            X.append(st[t])
            Y.append(np.concatenate(ac[t:t + H]))
            Yc.append(np.concatenate([expert_action(st[t + j][:2]) for j in range(H)]))
            AL.append(noise_band(st[t][1]))  # y is dim 1 of 4
    return (np.array(X, np.float32), np.array(Y, np.float32),
            np.array(Yc, np.float32), np.array(AL, dtype=bool))


class Reg(nn.Module):
    def __init__(self, hetero=False):
        super().__init__()
        self.hetero = hetero
        self.trunk = nn.Sequential(nn.Linear(OBS_D, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, ACT_D * H + (1 if hetero else 0))
        self.register_buffer("gsig2", torch.ones(1))

    def full(self, xn):
        return self.head(self.trunk(xn))

    def forward(self, xn):
        out = self.head(self.trunk(xn))
        return out[:, :ACT_D * H] if self.hetero else out


class FlowNet(nn.Module):
    def __init__(self, nsteps=2):
        super().__init__()
        self.nsteps = nsteps
        self.trunk = nn.Sequential(nn.Linear(OBS_D + ACT_D * H + 1, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, ACT_D * H)

    def vel(self, xn, yt, t):
        return self.head(self.trunk(torch.cat([xn, yt, t], 1)))

    def forward(self, xn):
        y = torch.zeros(len(xn), ACT_D * H, device=xn.device)
        for j in range(self.nsteps):
            t = torch.full((len(xn), 1), j / self.nsteps, device=xn.device)
            y = y + (1.0 / self.nsteps) * self.vel(xn, y, t)
        return y


class Norm:
    def __init__(self, X, Y):
        # obs scale from the corridor mass, not approach extremes: the
        # policy must resolve mm offsets, so p2-p98 sets the input unit
        self.xlo, self.xhi = np.percentile(X, 2, 0), np.percentile(X, 98, 0)
        self.ylo, self.yhi = Y.min(0), Y.max(0)

    def nx(self, x):
        return (2 * (x - self.xlo) / (self.xhi - self.xlo + 1e-8) - 1).astype(np.float32)

    def ny(self, y):
        return (2 * (y - self.ylo) / (self.yhi - self.ylo + 1e-8) - 1).astype(np.float32)

    def uy(self, yn):
        return (yn + 1) / 2 * (self.yhi - self.ylo + 1e-8) + self.ylo


def sigma_of(net, xb):
    s_raw = net.full(xb)[:, ACT_D * H:]
    return torch.nn.functional.softplus(s_raw).squeeze(-1) + 1e-3


def loss_fn(arm, net, xb, yb):
    if arm in ("mip", "flow8"):
        eps = torch.randn_like(yb)
        t = torch.rand(len(yb), 1, device=DEV)
        yt = (1 - t) * eps + t * yb
        return ((net.vel(xb, yt, t) - (yb - eps)) ** 2).mean()
    D = yb.shape[1]
    if arm == "l2":
        return ((net(xb) - yb) ** 2).mean()
    if arm == "cauchy":
        r2 = (net(xb) - yb) ** 2
        c = 0.2
        return torch.log1p(r2 / (c * c)).mean()
    if arm == "globalt":
        pred = net(xb)
        m = ((pred - yb) ** 2).sum(1) / D
        sig2 = net.gsig2.clamp_min(1e-8)
        with torch.no_grad():
            wg = (NU + 1.0) / (NU + m.detach() / sig2)
            net.gsig2.mul_(0.99).add_(0.01 * (wg * m.detach()).mean())
        per = 0.5 * (NU + 1.0) * torch.log1p(m / (NU * sig2)) * D \
            + D * torch.log(sig2.sqrt())
        return per.mean() / D
    out = net.full(xb)
    mu = out[:, :ACT_D * H]
    sigma = torch.nn.functional.softplus(out[:, ACT_D * H:]).squeeze(-1) + 1e-3
    m = ((mu - yb) ** 2).sum(1) / D
    if arm == "hg":
        per = 0.5 * m / sigma ** 2 * D + D * torch.log(sigma)
    else:  # ht
        per = 0.5 * (NU + 1.0) * torch.log1p(m / (NU * sigma ** 2)) * D \
            + D * torch.log(sigma)
    return per.mean() / D


def weight_of(arm, net, xb, yb):
    with torch.no_grad():
        D = yb.shape[1]
        if arm in ("mip", "flow8"):
            return (net(xb) - yb).abs().mean(1)
        if arm == "globalt":
            pred = net(xb)
            m = ((pred - yb) ** 2).sum(1) / D
            sig2 = net.gsig2.clamp_min(1e-8)
            return (NU + 1.0) * m.sqrt() / (NU * sig2 + m)
        if arm in ("l2", "cauchy"):
            r = (net(xb) - yb)
            if arm == "l2":
                return r.abs().mean(1)
            c = 0.2
            return (r.abs() / (1 + r ** 2 / (c * c))).mean(1)
        out = net.full(xb)
        mu = out[:, :ACT_D * H]
        s = torch.nn.functional.softplus(out[:, ACT_D * H:]).squeeze(-1) + 1e-3
        m = ((mu - yb) ** 2).sum(1) / D
        if arm == "hg":
            return m.sqrt() / s ** 2
        return (NU + 1.0) * m.sqrt() / (NU * s ** 2 + m)


def train(arm, seed, Xn, Yn):
    torch.manual_seed(seed)
    net = (FlowNet(2 if arm == "mip" else FLOW_NS) if arm in ("mip", "flow8")
           else Reg(hetero=arm in ("hg", "ht"))).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xn, device=DEV)
    Yt = torch.tensor(Yn, device=DEV)
    rng = np.random.RandomState(seed)
    for it in range(STEPS):
        idx = rng.randint(0, len(Xt), 256)
        loss = loss_fn(arm, net, Xt[idx], Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def rollout(net, norm, rng, AS=AS_DEF, expert=False):
    p = np.array([rng.uniform(-0.2, 0.2), rng.uniform(1.6, 1.7)])
    vel = np.zeros(2)
    held_e = 0
    for _ in range(ROLL_BUDGET // (1 if expert else AS)):
        if expert:  # scripted demonstrator: per-step feedback, no chunking
            a2 = expert_action(p)
            if HOLD > 0 and held_e < HOLD and p[1] <= Y_HOLD:
                a2 = np.array([0.0, 0.0])
                held_e += 1
            a2 = np.clip(a2 - MOM * vel, -CLIP, CLIP)
            chunk = a2[None, :]
        else:
            with torch.no_grad():
                ob = np.concatenate([p, vel])[None, :]
                xn = torch.tensor(norm.nx(ob), device=DEV)
                yn = net(xn).cpu().numpy()[0]
            chunk = norm.uy(yn).reshape(H, ACT_D)[:AS]
        for a in chunk:
            p_new = wall(p[:2] + np.clip(a[:2], -CLIP, CLIP) + MOM * vel
                         + PROC * rng.randn(2))
            vel = p_new - p[:2]
            p = p_new
            if np.isnan(p[0]):
                return "scrape"
            if np.linalg.norm(p - G) < DOCK_TOL:
                return "success"
    return "timeout"


def evaluate(net, norm, n=100, expert=False, seed0=0):
    rng = np.random.RandomState(9000 + seed0)
    outs = [rollout(net, norm, rng, expert=expert) for _ in range(n)]
    return {k: float(np.mean([o == k for o in outs]))
            for k in ("success", "scrape", "timeout")}


def main():
    results = {}
    ex = evaluate(None, None, n=100, expert=True)
    print(f"expert: {ex}", flush=True)
    assert ex["success"] == 1.0, "expert anchor broken"
    for seed in SEEDS:
        drng = np.random.RandomState(1000 + seed)
        X, Y, Yc, AL = build_dataset(drng)
        norm = Norm(X, Y)
        Xn, Yn, Ycn = norm.nx(X), norm.ny(Y), norm.ny(Yc)
        ins = (X[:, 1] > Y_DOCK) & (X[:, 1] <= Y_FUN_LO)
        for arm in ARMS:
            net = train(arm, seed, Xn, Yn)
            ev = evaluate(net, norm, n=100, seed0=seed)
            Xt = torch.tensor(Xn, device=DEV)
            Yt = torch.tensor(Yn, device=DEV)
            with torch.no_grad():
                w = weight_of(arm, net, Xt, Yt).cpu().numpy()
                pred = net(Xt).cpu().numpy()
            r = {"sr": ev["success"], "scrape": ev["scrape"], "to": ev["timeout"],
                 "led_noise": float(w[AL].sum() / (w.sum() + 1e-12)),
                 "ins_err": float(np.abs(pred[ins] - Yn[ins]).mean())}
            if arm in ("hg", "ht"):
                with torch.no_grad():
                    sig = sigma_of(net, Xt).cpu().numpy()
                r["sig_ratio"] = float(sig[AL].mean() / sig[~AL].mean())
            results[f"{arm}_s{seed}"] = r
            print(f"{arm} s{seed}: SR {r['sr']:.2f} (scr {r['scrape']:.2f} "
                  f"to {r['to']:.2f}) | ledN {r['led_noise']:.2f} | "
                  f"insErr {r['ins_err']:.4f}"
                  + (f" | sigR {r['sig_ratio']:.1f}" if "sig_ratio" in r else ""),
                  flush=True)
    print("\n=== TABLE (mean±sd over seeds) ===", flush=True)
    for arm in ARMS:
        vs = [results[f"{arm}_s{s}"]["sr"] for s in SEEDS]
        le = [results[f"{arm}_s{s}"]["led_noise"] for s in SEEDS]
        ie = [results[f"{arm}_s{s}"]["ins_err"] for s in SEEDS]
        print(f"{arm:>8}: SR {np.mean(vs):.2f}±{np.std(vs):.2f} | ledN "
              f"{np.mean(le):.2f} | insErr {np.mean(ie):.4f}", flush=True)
    json.dump(results, open("toytube.json", "w"), indent=1)


if __name__ == "__main__":
    main()
