"""Noise-free reference prices for the discrete up-and-out call (evaluation only).

For an alive history at monitoring date k with price s,
    u_N(s) = (s-K)^+,
    u_k(s) = e^{-r dt} E[u_{k+1}(S_{k+1}) 1{S_{k+1}<B} | S_k=s],
computed by one-step Gaussian quadrature on a uniform log-price grid that ends
exactly at log(B/S0) (one-sided trapezoid at the barrier). The conditional price
of a history is V_k = 1{max_{j<=k} S_j < B} u_k(S_k).

This module is never a model input or a training label for the pure-history
methods; it replaces nested Monte Carlo only when scoring saved predictions.
"""
import argparse
import json
import math

import torch


class DiscreteUpOutReference:
    def __init__(self, a, h=2.5e-4, smin=.02):
        self.a = a
        dt = a.T/a.steps
        xb = math.log(a.barrier/a.s0)
        n = int(round((xb-math.log(smin/a.s0))/h))+1
        x = xb-h*torch.arange(n-1, -1, -1, dtype=torch.float64)
        m, sd = (a.r-.5*a.sigma**2)*dt, a.sigma*math.sqrt(dt)
        half = int(math.ceil(12*sd/h))
        offs = h*torch.arange(-half, half+1, dtype=torch.float64)
        w = torch.exp(-.5*((offs-m)/sd)**2)/(sd*math.sqrt(2*math.pi))*h
        U = torch.zeros(a.steps+1, n, dtype=torch.float64)
        U[-1] = torch.relu(a.s0*torch.exp(x)-a.strike)
        disc = math.exp(-a.r*dt)
        for k in range(a.steps-1, -1, -1):
            f = U[k+1].clone()
            f[-1] *= .5
            U[k] = disc*torch.nn.functional.conv1d(f[None, None], w[None, None], padding=half)[0, 0]
        self.x0, self.h, self.U = float(x[0]), h, U

    def node_values(self, prices):
        """prices: (n, steps+1) fine-grid prices on any device -> float64 CPU (n, steps+1)."""
        p = prices.detach().double().cpu()
        logs = torch.log(p/self.a.s0)
        alive = torch.cummax(p, 1).values < self.a.barrier
        pos = ((logs-self.x0)/self.h).clamp(0, self.U.shape[1]-1-1e-9)
        i0 = pos.floor().long()
        fr = pos-i0
        rows = torch.arange(p.shape[1]).expand(p.shape[0], -1)
        u = self.U[rows, i0]*(1-fr)+self.U[rows, (i0+1).clamp_max(self.U.shape[1]-1)]*fr
        u[:, -1] = torch.relu(p[:, -1]-self.a.strike)
        return torch.where(alive, u, torch.zeros_like(u))


def score(prediction, reference, prices, a, index):
    """Relative L2 and alive/dead split for one monitoring index (float64 CPU inputs)."""
    p, v = prediction.double().cpu(), reference.double().cpu()
    dead = prices[:, :index+1].detach().cpu().max(1).values >= a.barrier
    den = float(v.square().sum())
    return dict(time=a.T*index/a.steps, index=index, histories=len(v),
                relative_l2=math.sqrt(float((p-v).square().sum())/den) if den > 0 else None,
                dead_count=int(dead.sum()),
                dead_prediction_mean=float(p[dead].mean()) if dead.any() else None,
                alive_prediction_mean=float(p[~dead].mean()) if (~dead).any() else None,
                alive_reference_mean=float(v[~dead].mean()) if (~dead).any() else None,
                dead_share_of_squared_error=float((p[dead]-v[dead]).square().sum()/(p-v).square().sum().clamp_min(1e-300)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Validate the quadrature reference against plain Monte Carlo.')
    for key, value in [('s0', 1.), ('strike', 1.), ('barrier', 1.2), ('r', .03), ('sigma', .4), ('T', 1.)]:
        p.add_argument('--'+key, type=float, default=value)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--paths', type=int, default=1 << 21)
    p.add_argument('--seed', type=int, default=20261021)
    a = p.parse_args()
    ref = DiscreteUpOutReference(a)
    torch.manual_seed(a.seed)
    dt = a.T/a.steps
    total = square = 0.
    for start in range(0, a.paths, 1 << 16):
        n = min(1 << 16, a.paths-start)
        z = torch.randn(n, a.steps, dtype=torch.float64)
        x = a.s0*torch.exp(((a.r-.5*a.sigma**2)*dt+a.sigma*math.sqrt(dt)*z).cumsum(1))
        g = math.exp(-a.r*a.T)*torch.relu(x[:, -1]-a.strike)*(x.max(1).values < a.barrier)
        total += float(g.sum()); square += float(g.square().sum())
    mean = total/a.paths
    se = math.sqrt(max(0., (square-a.paths*mean**2)/(a.paths-1)/a.paths))
    quad = float(ref.U[0, int(round((0-ref.x0)/ref.h))])
    print(json.dumps(dict(quadrature_t0=quad, mc_t0=mean, mc_se=se, gap_in_se=(quad-mean)/se), indent=2))
