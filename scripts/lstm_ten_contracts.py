"""Contract layer for the pure-history LSTM PINN/BSDE ten-contract study.

Same contracts as state_pinn/ten_contract_models.py (S0=K=1, r=.03, sigma=.4, T=1;
barrier 1.2 up / .8 down), but with 128-date DISCRETE monitoring, matching the
LSTM line. Six trained cases; the four knock-ins follow from the exact discrete
in-out parity V_in = vanilla - V_out (never trained).

History-free lifts H(t,S) (they never see the past; Y = H*sigmoid(A)):
  knock-outs   : discounted vanilla payoff restricted to the alive side, checked at
                 maturity only (no intermediate monitoring)
  lookback call: S        (0 < V < S)
  lookback put : cap*S    (V/S = G(log(M/S)) is unbounded in principle; cap=6
                 covers drawdowns to about 1/7 of the running maximum)
Exact references (evaluation only) use one-step Gaussian quadrature.
"""
import math

import torch

from .barrier_heat_kernel import interval_exponential, vanilla_tensor

CASES = ('lookback_call', 'lookback_put', 'up_call', 'up_put', 'down_call', 'down_put')
KNOCK_IN = {'up_call': 'up_in_call', 'up_put': 'up_in_put', 'down_call': 'down_in_call', 'down_put': 'down_in_put'}


def spec(case):
    kind, option = case.split('_')
    if kind == 'lookback':
        return dict(kind='lookback', option=option)
    return dict(kind='barrier', option=option, direction=kind, barrier=1.2 if kind == 'up' else .8)


def payoff(case, prices):
    """prices: (n, N+1) fine-grid prices including S0."""
    c = spec(case)
    sT = prices[:, -1]
    if c['kind'] == 'lookback':
        return sT-prices.min(1).values if c['option'] == 'call' else prices.max(1).values-sT
    vanilla = torch.relu(sT-1.) if c['option'] == 'call' else torch.relu(1.-sT)
    alive = prices.max(1).values < c['barrier'] if c['direction'] == 'up' else prices.min(1).values > c['barrier']
    return vanilla*alive


def lift(case, t, s, a):
    c = spec(case)
    if c['kind'] == 'lookback':
        return s if c['option'] == 'call' else a.put_cap*s
    tau = (a.T-t).clamp_min(0)
    safe = torch.where(tau > 0, tau, torch.ones_like(tau))
    mean, var = s.log()+(a.r-.5*a.sigma**2)*safe, a.sigma**2*safe
    lo, hi = (0., math.inf) if c['option'] == 'call' else (-math.inf, 0.)          # log K = 0
    b = math.log(c['barrier'])
    lo, hi = (lo, min(hi, b)) if c['direction'] == 'up' else (max(lo, b), hi)
    stock, mass = interval_exponential(1., mean, var, lo, hi), interval_exponential(0., mean, var, lo, hi)
    h = torch.exp(-a.r*safe)*((stock-mass) if c['option'] == 'call' else (mass-stock))
    local = (torch.relu(s-1.) if c['option'] == 'call' else torch.relu(1.-s))
    local = local*((s < c['barrier']) if c['direction'] == 'up' else (s > c['barrier']))
    return torch.where(tau > 0, h.clamp_min(0), local)


def vanilla(case, t, s, a):
    return vanilla_tensor(s, (a.T-t).clamp_min(0), 1., a.r, a.sigma, spec(case)['option'])


def _gauss_conv(f, mean, sd, h, pad_lo, pad_hi):
    half = int(math.ceil(12*sd/h))
    offs = h*torch.arange(-half, half+1, dtype=torch.float64)
    w = torch.exp(-.5*((offs-mean)/sd)**2)/(sd*math.sqrt(2*math.pi))*h
    lo = torch.zeros(half, dtype=torch.float64) if pad_lo == 'zero' else f[0].repeat(half)
    hi = torch.zeros(half, dtype=torch.float64) if pad_hi == 'zero' else f[-1].repeat(half)
    return torch.nn.functional.conv1d(torch.cat([lo, f, hi])[None, None], w[None, None])[0, 0]


class ExactReference:
    """node_values(prices) -> exact conditional prices at every monitoring node (float64, CPU)."""
    def __init__(self, case, a, h=2.5e-4):
        self.case, self.a, self.c, self.h = case, a, spec(case), h
        dt = a.T/a.steps
        sd = a.sigma*math.sqrt(dt)
        if self.c['kind'] == 'barrier':
            b = math.log(self.c['barrier'])
            if self.c['direction'] == 'up':     # alive grid ends exactly at log B
                n = int(round((b-math.log(.02))/h))+1
                x = b-h*torch.arange(n-1, -1, -1, dtype=torch.float64)
                edge, pads = -1, ('edge', 'zero')
            else:                               # alive grid starts exactly at log B
                n = int(round((math.log(20.)-b)/h))+1
                x = b+h*torch.arange(n, dtype=torch.float64)
                edge, pads = 0, ('zero', 'edge')
            s = torch.exp(x)
            U = torch.zeros(a.steps+1, n, dtype=torch.float64)
            U[-1] = torch.relu(s-1.) if self.c['option'] == 'call' else torch.relu(1.-s)
            for k in range(a.steps-1, -1, -1):
                f = U[k+1].clone(); f[edge] *= .5                       # one-sided trapezoid at the barrier
                U[k] = math.exp(-a.r*dt)*_gauss_conv(f, (a.r-.5*a.sigma**2)*dt, sd, h, *pads)
            self.x0, self.U = float(x[0]), U
        else:
            mu = (a.r+.5*a.sigma**2)*dt                                  # stock-numeraire log step
            n = int(round(5./h))+1
            y = h*torch.arange(n, dtype=torch.float64)
            G = torch.zeros(a.steps+1, n, dtype=torch.float64)
            G[0] = 1-torch.exp(-y) if self.c['option'] == 'call' else torch.exp(y)-1
            sign = 1. if self.c['option'] == 'call' else -1.             # call: x+Z ; put: y-Z
            reset = .5*(1+torch.erf(((-y-sign*mu)/sd)/math.sqrt(2)))     # P(state would cross below 0)
            for k in range(1, a.steps+1):
                f = G[k-1].clone(); f0 = float(f[0]); f[0] *= .5
                G[k] = _gauss_conv(f, sign*mu, sd, h, 'zero', 'edge')+f0*reset
            self.G = G

    def node_values(self, prices):
        p = prices.detach().double().cpu()
        a, N = self.a, self.a.steps
        if self.c['kind'] == 'barrier':
            B = self.c['barrier']
            alive = torch.cummax(p, 1).values < B if self.c['direction'] == 'up' else torch.cummin(p, 1).values > B
            pos = ((torch.log(p)-self.x0)/self.h).clamp(0, self.U.shape[1]-1-1e-9)
            i0 = pos.floor().long(); fr = pos-i0
            rows = torch.arange(N+1).expand(p.shape[0], -1)
            u = self.U[rows, i0]*(1-fr)+self.U[rows, (i0+1).clamp_max(self.U.shape[1]-1)]*fr
            u[:, -1] = torch.relu(p[:, -1]-1.) if self.c['option'] == 'call' else torch.relu(1.-p[:, -1])
            return torch.where(alive, u, torch.zeros_like(u))
        ext = torch.cummin(p, 1).values if self.c['option'] == 'call' else torch.cummax(p, 1).values
        state = torch.log(p/ext).abs()
        pos = (state/self.h).clamp(0, self.G.shape[1]-1-1e-9)
        i0 = pos.floor().long(); fr = pos-i0
        rem = (N-torch.arange(N+1)).expand(p.shape[0], -1)
        v = p*(self.G[rem, i0]*(1-fr)+self.G[rem, (i0+1).clamp_max(self.G.shape[1]-1)]*fr)
        v[:, -1] = (p[:, -1]-ext[:, -1]) if self.c['option'] == 'call' else (ext[:, -1]-p[:, -1])
        return v


def score(case, a, prices, pred, nodes):
    """pred: (n, len(nodes)) model prices at fine indices `nodes`; returns exact-reference metrics.
    Knock-outs also report the parity knock-in V_in = vanilla - V_out (prediction and reference)."""
    ref = ExactReference(case, a)
    p = prices.double().cpu()
    exact = ref.node_values(p)[:, nodes]
    pred = pred.double().cpu()
    ts = torch.tensor([a.T*k/a.steps for k in nodes], dtype=torch.float64)

    def block(P, E):
        rows = [dict(time=float(ts[j]), index=int(k), relative_l2=float((P[:, j]-E[:, j]).norm()/E[:, j].norm()) if E[:, j].norm() > 0 else None)
                for j, k in enumerate(nodes)]
        return dict(t0_prediction=float(P[0, 0]), t0_exact=float(E[0, 0]), by_date=rows)

    out = dict(case=case, histories=len(p), out=block(pred, exact))
    if case in KNOCK_IN:
        van = vanilla(case, ts.expand(len(p), -1), p[:, nodes], a)
        out['in'] = dict(contract=KNOCK_IN[case], **block(van-pred, van-exact))
    return out
