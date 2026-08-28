"""Story-proof toy (pre-reg PART CCCXIX). Funnel-corridor docking with
heavy-tailed label tremor ON the critical align band (no transmission
claims). Arms = the paper story: l2 | cauchy | hg | ht | globalt | mip.
Instruments: SR/taxonomy, seed-sd (selector), sigma-map fidelity,
repricing ledger, insert-band fit error, hg early poison signature."""
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = int(os.environ.get("H_OVR", "8"))
G = np.array([0.0, 0.0])
Y_TOP, Y_ALIGN_LO, Y_DOCK = 0.5, 0.25, 0.05
S_HI = float(os.environ.get("S_HI", "0.06"))
NU = 2.0
CLIP = 0.08
DOCK_TOL = float(os.environ.get("DOCK_TOL", "0.008"))
DOGLEG = float(os.environ.get("DOGLEG", "0.02"))
NEP = int(os.environ.get("NEP", "60"))
STEPS = int(os.environ.get("STEPS", "25000"))
WIDTH = int(os.environ.get("WIDTH", "64"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7").split(",")]
ARMS = tuple(os.environ.get("ARMS", "l2,cauchy,hg,ht,globalt,mip").split(","))


def center(y, sgn=1.0):
    """fine structure in the CLEAN insert band: a lateral dogleg the policy
    must track (the robot-insert analog of hard-to-learn precision content)."""
    if Y_DOCK <= y < Y_ALIGN_LO:
        return sgn * DOGLEG * float(np.sin(np.pi * (Y_ALIGN_LO - y) / (Y_ALIGN_LO - Y_DOCK)))
    return 0.0


def half_width(y):
    if y >= Y_TOP:
        return 1.0  # no corridor above
    f = np.clip((y - Y_DOCK) / (Y_TOP - Y_DOCK), 0.0, 1.0)
    return WSCALE * (0.03 + 0.07 * f)


def wall(p, sgn=1.0):
    if Y_DOCK < p[1] < Y_TOP:
        hw = half_width(p[1])
        if SCRAPE:
            c = center(p[1], sgn)
            if abs(p[0] - c) > hw:
                p[0] = float("nan")  # scrape/crash marker
            return p
        if FORK and abs(center(p[1], 1.0)) > 0.006:
            cp = center(p[1], 1.0)
            hf = FORKW * hw
            lo_gap, hi_gap = -cp + hf, cp - hf  # island between the channels
            if lo_gap < p[0] < hi_gap:
                p[0] = float("nan")  # island crash marker
                return p
            side = 1.0 if p[0] >= 0 else -1.0
            c = cp * side
            p[0] = float(np.clip(p[0], c - hf, c + hf))
            return p
        c = center(p[1], sgn)
        p[0] = float(np.clip(p[0], c - hw, c + hw))
    return p


S_MID = float(os.environ.get("S_MID", "0"))
S_INS = float(os.environ.get("S_INS", "0"))
NOISE_TYPE = os.environ.get("NOISE_TYPE", "dir")  # none|dir|speed|flip|strat
T1S = float(os.environ.get("T1S", "0.8"))
STRAT_P = float(os.environ.get("STRAT_P", "0.8"))
ACT_D = 3 if NOISE_TYPE in ("flip", "flip3") else 2
Y_FLIP_HI, Y_FLIP_LO, Y_JAM = 0.20, 0.10, 0.22
GRIP_LO, GRIP_HI = 0.2, 0.8
SPEED_OBSV = os.environ.get("SPEED_OBSV", "0") == "1"
# v3 session knobs (speed3 / flip3 / strat3)
T1S3 = float(os.environ.get("T1S3", "0.9"))     # pace jitter scale, t(2)
PAUSE_P = float(os.environ.get("PAUSE_P", "0.06"))
Y_OBJ = 0.30                                     # grasp object height (flip3)
OBJ_W = float(os.environ.get("OBJ_W", "0.025"))  # object window half-width
CLOSE_RATE = float(os.environ.get("CLOSE_RATE", "0.5"))
Y_LEDGE = 0.20                                   # regrip ledge (flip3)
LEDGE_W = 0.015
REGRIP_MAX = int(os.environ.get("REGRIP_MAX", "4"))
SCRAPE = os.environ.get("SCRAPE", "0") == "1"    # walls fail instead of slide
WSCALE = float(os.environ.get("WSCALE", "1.0"))
FORKW = float(os.environ.get("FORKW", "0.5"))
SPEED_LAG = os.environ.get("SPEED_LAG", "0") == "1"
MOM = float(os.environ.get("MOM", "0"))  # momentum dynamics; velocity hidden from obs
MIRROR_OBS = os.environ.get("MIRROR_OBS", "0") == "1"  # strat3 aliasing
OBS_D = 3 if SPEED_OBSV else 2
ROLL_BUDGET = int(os.environ.get("ROLL_BUDGET", "150"))
FORK = os.environ.get("FORK", "0") == "1"  # strat round-2: true fork w/ island


def noise_scale(y):
    if Y_ALIGN_LO <= y <= Y_TOP:
        return S_HI
    if S_MID > 0 and Y_TOP < y <= 0.7:
        return S_MID
    return 0.0


def ins_noise(y):
    return S_INS if Y_DOCK <= y < Y_ALIGN_LO else 0.0


def expert_action(p, sgn=1.0, y_flip=None):
    if p[1] < Y_TOP + 0.05:  # slow corridor descent, decelerating into the dock
        vy = -min(0.03, 0.5 * max(p[1], 0.0) + 0.004)
    else:
        vy = np.clip(0.3 * (G[1] - p[1]) - 0.02, -CLIP, CLIP)
    # feedforward center tracking (competent demonstrator: no lag on the dogleg)
    a = np.clip(np.array([center(p[1] + vy, sgn) - p[0], vy]), -CLIP, CLIP)
    if NOISE_TYPE == "flip":
        g = 1.0 if (y_flip is not None and p[1] < y_flip) else 0.0
        return np.append(a, g)
    if NOISE_TYPE == "flip3":
        # y_flip carries (grasp_y, regrip_steps_done_ref, regrip_K)
        return a  # flip3 expert handled in gen/rollout state machines
    return a


def gen_episode(rng):
    p = np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.8, 1.0)])
    sgn = 1.0
    if NOISE_TYPE in ("strat", "strat3"):
        sgn = 1.0 if rng.rand() < STRAT_P else -1.0
    y_flip = rng.uniform(Y_FLIP_LO, Y_FLIP_HI) if NOISE_TYPE == "flip" else None
    regrip_K = rng.randint(1, REGRIP_MAX + 1) if NOISE_TYPE == "flip3" else 0
    regrip_done, g_now, closed = 0, 0.0, False
    states, acts, tail = [], [], H + 2
    vprev = 0.0
    vel = np.zeros(2)
    for _ in range(120):
        a = expert_action(p, sgn, y_flip)
        if NOISE_TYPE == "flip3":
            in_creep = 0.26 <= p[1] <= 0.34
            at_ledge = abs(p[1] - Y_LEDGE) < LEDGE_W
            if not closed and p[1] > Y_OBJ:
                g_now = 1.0 if p[1] <= Y_OBJ + OBJ_W else 0.0
            if p[1] <= Y_OBJ:
                closed = True
                g_now = 1.0
            if in_creep:
                a[1] = -0.0075
                a[0] = float(np.clip(center(p[1] + a[1], sgn) - p[0], -CLIP, CLIP))
            if closed and at_ledge and regrip_done < regrip_K:
                a = np.array([0.0, 0.0])  # stationary regrip ritual
                g_now = 0.0
                regrip_done += 1
            elif closed and regrip_done >= regrip_K:
                g_now = 1.0
            a = np.append(a[:2], g_now)
        if NOISE_TYPE == "dir":
            sc = noise_scale(p[1])
            if sc > 0:
                a[0] = float(np.clip(a[0] + sc * rng.standard_t(2), -CLIP, CLIP))
        elif NOISE_TYPE == "speed3":
            if p[1] < Y_TOP + 0.05:  # slow careful windows = the whole corridor
                if rng.rand() < PAUSE_P:
                    eta = 0.0
                else:
                    eta = float(np.clip(1.0 + T1S3 * rng.standard_t(2), 0.0, 3.5))
                a[1] = float(np.clip(a[1] * eta, -CLIP, CLIP))
                if not SPEED_LAG:
                    a[0] = float(np.clip(center(p[1] + a[1], sgn) - p[0], -CLIP, CLIP))
                # SPEED_LAG: keep the nominal-pace x-target (compensation lag) —
                # pace error on the curve becomes lateral consequence
        elif NOISE_TYPE == "speed":
            if Y_ALIGN_LO <= p[1] <= Y_TOP:
                eta = float(np.clip(1.0 + T1S * rng.standard_t(2), 0.0, 3.0))
                a[1] = float(np.clip(a[1] * eta, -CLIP, CLIP))
        ob = p.copy()
        if MIRROR_OBS and Y_DOCK < p[1] < Y_ALIGN_LO:
            ob = np.array([abs(p[0]), p[1]])
        states.append(np.append(ob, vprev) if SPEED_OBSV else ob)
        if MOM > 0:
            a[1] = float(np.clip(a[1] - MOM * vel[1], -CLIP, CLIP))
            a[0] = float(np.clip(a[0] - MOM * vel[0], -CLIP, CLIP))
        acts.append(a.copy())
        p_new = wall(p + a[:2] + MOM * vel, sgn)
        vel = p_new - p
        p = p_new
        vprev = float(a[1])
        if np.linalg.norm(p - G) < DOCK_TOL:
            tail -= 1
            if tail <= 0:
                break
    return states, acts


def build_dataset(rng):
    X, Y, Yc, AL = [], [], [], []
    for _ in range(NEP):
        st, ac = gen_episode(rng)
        for t in range(len(st) - H):
            X.append(st[t])
            Y.append(np.concatenate(ac[t:t + H]))
            if ACT_D == 3:
                # clean xy from the expert law; g copied from the demo (its
                # timing is the aleatoric content, no unique clean reference)
                Yc.append(np.concatenate(
                    [np.append(expert_action(st[t + j][:2])[:2], ac[t + j][2])
                     for j in range(H)]))
            else:
                Yc.append(np.concatenate([expert_action(st[t + j][:2]) for j in range(H)]))
            AL.append(noise_scale(st[t][1]) > 0)
    return (np.array(X, np.float32), np.array(Y, np.float32),
            np.array(Yc, np.float32), np.array(AL, dtype=bool))


class Reg(nn.Module):
    def __init__(self, hetero=False):
        super().__init__()
        self.hetero = hetero
        self.trunk = nn.Sequential(nn.Linear(OBS_D, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.htd = hetero == "htd"
        nh = (2 * ACT_D * H if self.htd
              else ACT_D * H + (1 if hetero else 0))
        self.head = nn.Linear(WIDTH, nh)
        self.register_buffer("gsig2", torch.ones(1))

    def full(self, xn):
        return self.head(self.trunk(xn))

    def forward(self, xn):
        out = self.head(self.trunk(xn))
        return out[:, :ACT_D * H] if (self.hetero or self.htd) else out


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
        self.xlo, self.xhi = X.min(0), X.max(0)
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


def loss_fn(arm, net, xb, yb, wb=None):
    if arm in ("l2ow", "l2sw"):  # fixed-weight plain MSE (oracle / sigma-transplant)
        per = ((net(xb) - yb) ** 2).mean(1)
        return (per * wb).mean()
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
            w = (NU + 1.0) / (NU + m.detach() / sig2)
            net.gsig2.mul_(0.99).add_(0.01 * (w * m.detach()).mean())
        per = 0.5 * (NU + 1.0) * torch.log1p(m / (NU * sig2)) * D \
            + D * torch.log(sig2.sqrt())
        return per.mean() / D
    if arm == "htd":  # per-dim-per-step sigma, t-NLL summed over dims
        out = net.full(xb)
        D = ACT_D * H
        mu, sig = out[:, :D], torch.nn.functional.softplus(out[:, D:]) + 1e-3
        r2 = (mu - yb) ** 2
        per = (0.5 * (NU + 1.0) * torch.log1p(r2 / (NU * sig ** 2))
               + torch.log(sig)).sum(1)
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
    """per-sample gradient-weight proxy for the repricing ledger."""
    with torch.no_grad():
        D = yb.shape[1]
        if arm in ("mip", "flow8"):
            return (net(xb) - yb).abs().mean(1)
        if arm == "htd":
            out = net.full(xb)
            D = ACT_D * H
            mu, sig = out[:, :D], torch.nn.functional.softplus(out[:, D:]) + 1e-3
            r = (mu - yb).abs()
            return ((NU + 1.0) * r / (NU * sig ** 2 + r ** 2)).mean(1)
        if arm in ("l2", "cauchy", "l2ow", "l2sw"):
            r = (net(xb) - yb)
            if arm in ("l2", "l2ow", "l2sw"):
                return r.abs().mean(1)
            c = 0.2
            return (r.abs() / (1 + r ** 2 / (c * c))).mean(1)
        if arm == "globalt":
            pred = net(xb)
            m = ((pred - yb) ** 2).sum(1) / D
            sig2 = net.gsig2.clamp_min(1e-8)
            return (NU + 1.0) * m.sqrt() / (NU * sig2 + m)
        out = net.full(xb)
        mu, s = out[:, :ACT_D * H], torch.nn.functional.softplus(out[:, ACT_D * H:]).squeeze(-1) + 1e-3
        m = ((mu - yb) ** 2).sum(1) / D
        if arm == "hg":
            return m.sqrt() / s ** 2
        return (NU + 1.0) * m.sqrt() / (NU * s ** 2 + m)


def train(arm, seed, Xn, Yn, w_all=None):
    torch.manual_seed(seed)
    net = (FlowNet(2 if arm == "mip" else 8) if arm in ("mip", "flow8")
           else Reg(hetero=("htd" if arm == "htd"
                            else arm in ("hg", "ht")))).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xn, device=DEV)
    Yt = torch.tensor(Yn, device=DEV)
    rng = np.random.RandomState(seed)
    Wt = torch.tensor(w_all, device=DEV) if w_all is not None else None
    snap3k = None
    for it in range(STEPS):
        idx = rng.randint(0, len(Xt), 256)
        loss = loss_fn(arm, net, Xt[idx], Yt[idx],
                       Wt[idx] if Wt is not None else None)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it + 1 == 3000:
            snap3k = copy.deepcopy(net)
    return net, snap3k


def rollout(net, norm, ep, AS=8, expert=False, sgn=1.0):
    p = ep.copy()
    budget = ROLL_BUDGET
    vprev = 0.0
    vel = np.zeros(2)
    g_state, flipped, jammed = 0.0, False, False
    grip_closed, grasped, missed_obj = 0.0, False, False
    e_regripK, e_regrip_done, e_closed = 2, 0, False
    y_flip_e = 0.15 if NOISE_TYPE == "flip" else (Y_OBJ if NOISE_TYPE == "flip3" else None)
    while budget > 0:
        if expert:
            ea = expert_action(p, sgn, y_flip_e)
            if NOISE_TYPE == "flip3":
                in_creep = 0.26 <= p[1] <= 0.34
                at_ledge = abs(p[1] - Y_LEDGE) < LEDGE_W
                g_now = 1.0 if (e_closed or (Y_OBJ < p[1] <= Y_OBJ + OBJ_W)) else 0.0
                if p[1] <= Y_OBJ:
                    e_closed = True
                    g_now = 1.0
                if in_creep:
                    ea[1] = -0.0075
                    ea[0] = float(np.clip(center(p[1] + ea[1], sgn) - p[0], -CLIP, CLIP))
                if e_closed and at_ledge and e_regrip_done < e_regripK:
                    ea = np.array([0.0, 0.0])
                    g_now = 0.0
                    e_regrip_done += 1
                elif e_closed and e_regrip_done >= e_regripK:
                    g_now = 1.0
                ea = np.append(ea[:2], g_now)
            chunk = np.array([ea])
        else:
            ob = p.copy()
            if MIRROR_OBS and Y_DOCK < p[1] < Y_ALIGN_LO:
                ob = np.array([abs(p[0]), p[1]])
            ob = np.append(ob, vprev) if SPEED_OBSV else ob
            on = torch.tensor(norm.nx(ob[None].astype(np.float32)), device=DEV)
            with torch.no_grad():
                chunk = norm.uy(net(on)[0].cpu().numpy()).reshape(H, ACT_D)
        for a in chunk[:AS]:
            ax, ay = float(a[0]), float(a[1])
            if NOISE_TYPE == "flip":
                g_cmd = float(a[2])
                if GRIP_LO < g_cmd < GRIP_HI:
                    ax *= 0.5  # partial grip degrades lateral authority
                if g_cmd >= 0.5 and not flipped:
                    flipped = True
                    if p[1] > Y_JAM:
                        return "jam"
                g_state = g_cmd
            elif NOISE_TYPE == "flip3":
                g_cmd = float(a[2])
                in_win = abs(p[1] - Y_OBJ) < OBJ_W
                at_ledge = abs(p[1] - Y_LEDGE) < LEDGE_W
                if in_win and g_cmd > 0.8:
                    grip_closed += CLOSE_RATE
                if grip_closed >= 1.0:
                    grasped = True
                if p[1] < Y_OBJ - OBJ_W and not grasped:
                    return "nogrip"
                if grasped and not at_ledge and p[1] < Y_OBJ - OBJ_W:
                    moving = abs(ay) > 0.005
                    if moving and GRIP_LO < g_cmd < GRIP_HI:
                        return "drop"  # moving with partial grip
                    if moving and g_cmd <= GRIP_LO:
                        return "drop"  # moving with open gripper below the object
            aa = np.clip(np.array([ax, ay]), -CLIP, CLIP)
            if expert and MOM > 0:
                aa = np.clip(aa - MOM * vel, -CLIP, CLIP)
            p_new = wall(p + aa + MOM * vel, sgn)
            if not np.isnan(p_new[0]):
                vel = p_new - p
            p = p_new
            vprev = float(aa[1])
            budget -= 1
            if np.isnan(p[0]):
                return "island"
            if np.linalg.norm(p - G) < DOCK_TOL:
                if NOISE_TYPE == "flip" and g_state < 0.5:
                    return "nogrip"
                return "success"
            if np.isnan(p[0]):
                return "island"
            if p[1] < -0.05:
                return "miss"
            if budget <= 0:
                break
    return "timeout"


def evaluate(net, norm, AS=8, n=100, expert=False):
    rng = np.random.RandomState(555 + AS)
    outs = []
    for _ in range(n):
        ep = np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.8, 1.0)])
        sgn = 1.0
        if NOISE_TYPE == "strat":
            sgn = 1.0 if rng.rand() < STRAT_P else -1.0
        outs.append(rollout(net, norm, ep, AS, expert, sgn))
    return {o: outs.count(o) / len(outs)
            for o in ("success", "miss", "timeout", "jam", "nogrip", "island", "drop")}


def main():
    results = {}
    if os.path.exists("toyhuman.json"):
        results.update(json.load(open("toyhuman.json")))
    drng = np.random.RandomState(0)
    X, Y, Yc, AL = build_dataset(drng)
    norm = Norm(X, Y)
    Xn, Yn, Ycn = norm.nx(X), norm.ny(Y), norm.ny(Yc)
    ii = np.where(~AL & (X[:, 1] < Y_ALIGN_LO) & (X[:, 1] > 0.02))[0]
    print(f"dataset {len(X)} samples, align frac {AL.mean():.2f}", flush=True)
    e = evaluate(None, None, expert=True)
    print(f"expert: {e}", flush=True)
    Xt = torch.tensor(Xn, device=DEV)
    Yt = torch.tensor(Yn, device=DEV)
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}_s{seed}"
            if key in results:
                continue
            w_all = (np.where(AL, 0.05, 1.0).astype(np.float32)
                     if arm == "l2ow" else None)
            if arm == "l2sw":  # transplant ht's learned sigma map as fixed weights
                htn = torch.load("human_ck_ht_med.pt", map_location=DEV,
                                 weights_only=False)
                with torch.no_grad():
                    sg = sigma_of(htn, torch.tensor(Xn, device=DEV)).cpu().numpy()
                w_all = (np.median(sg) / sg) ** 2
                w_all = np.clip(w_all / w_all.mean(), 0.02, 20.0).astype(np.float32)
            net, snap3k = train(arm, seed, Xn, Yn, w_all)
            r = {}
            ev = evaluate(net, norm, AS=8)
            r["sr"], r["exit"], r["to"] = ev["success"], ev["miss"], ev["timeout"]
            r["jam"], r["nogrip"] = ev.get("jam", 0.0), ev.get("nogrip", 0.0)
            r["island"] = ev.get("island", 0.0)
            r["drop"] = ev.get("drop", 0.0)
            w = weight_of(arm, net, Xt, Yt).cpu().numpy()
            if arm in ("l2ow", "l2sw") and 'w_all' in dir() and w_all is not None:
                w = w * w_all  # effective (weighted) ledger
            r["led_align"] = float(w[AL].sum() / (w.sum() + 1e-12))
            if snap3k is not None:
                w3 = weight_of(arm, snap3k, Xt, Yt).cpu().numpy()
                r["led_align_3k"] = float(w3[AL].sum() / (w3.sum() + 1e-12))
            if arm in ("hg", "ht"):
                with torch.no_grad():
                    sig = sigma_of(net, Xt).cpu().numpy()
                r["sig_ratio"] = float(sig[AL].mean() / (sig[~AL].mean() + 1e-12))
            if arm == "htd":
                with torch.no_grad():
                    out = net.full(Xt)
                    sd = torch.nn.functional.softplus(
                        out[:, ACT_D * H:]).mean(1).cpu().numpy()
                r["sig_ratio"] = float(sd[AL].mean() / (sd[~AL].mean() + 1e-12))
            if arm == "globalt":
                r["gsig"] = float(net.gsig2.sqrt().item())
            with torch.no_grad():
                pred = net(Xt[ii]).cpu().numpy() if arm != "mip" else net(Xt[ii]).cpu().numpy()
            r["ins_err"] = float(np.abs(pred - Ycn[ii]).mean())
            results[key] = r
            print(f"{arm} s{seed}: SR {r['sr']:.2f} (exit {r['exit']:.2f} "
                  f"to {r['to']:.2f}) | ledA {r['led_align']:.2f} | "
                  f"insErr {r['ins_err']:.4f}"
                  + (f" | sigR {r['sig_ratio']:.1f}" if "sig_ratio" in r else "")
                  + (f" | gsig {r['gsig']:.3f}" if "gsig" in r else ""), flush=True)
    print("\n=== TABLE (mean±sd over seeds) ===", flush=True)
    for arm in ARMS:
        vs = [results[f"{arm}_s{s}"]["sr"] for s in SEEDS if f"{arm}_s{s}" in results]
        le = [results[f"{arm}_s{s}"]["led_align"] for s in SEEDS if f"{arm}_s{s}" in results]
        ie = [results[f"{arm}_s{s}"]["ins_err"] for s in SEEDS if f"{arm}_s{s}" in results]
        print(f"{arm:>8}: SR {np.mean(vs):.2f}±{np.std(vs):.2f} | ledA "
              f"{np.mean(le):.2f} | insErr {np.mean(ie):.4f}", flush=True)
    json.dump(results, open("toyhuman.json", "w"), indent=1)
    print("wrote toyhuman.json", flush=True)


if __name__ == "__main__":
    main()
