from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ..analytic_bs import floating_lookback_price
from ..artifacts import RunRecorder


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class Market:
    s0: float = 1.0
    r: float = 0.03
    sigma: float = 0.4
    T: float = 1.0


@dataclass(frozen=True)
class LookbackContract:
    option_type: OptionType = "call"


class MLP(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 4,
                 price_parameterization: str = "direct", option_type: OptionType = "call"):
        super().__init__()
        if price_parameterization not in ("direct", "terminal_linear", "hard_lift", "heat_lift"):
            raise ValueError("Unknown lookback price parameterization")
        if option_type not in ("call", "put"):
            raise ValueError("Unknown lookback option type")
        self.price_parameterization = price_parameterization
        self.option_type = option_type
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def reflected_heat_payoff(market: Market, t: torch.Tensor, s: torch.Tensor,
                          y: torch.Tensor, option_type: OptionType) -> torch.Tensor:
    """Diffusion-only Neumann heat lift, not the drifted lookback solution.

    E exp(k*|x+sigma*sqrt(tau)*Z|), k=-1 for call and +1 for put.
    The even heat kernel gives zero extremum partial derivative for tau>0.
    """
    tau = market.T - t
    alive = tau > 0
    safe_tau = torch.where(alive, tau, torch.full_like(tau, market.T))
    x = torch.log(s / y) if option_type == "call" else torch.log(y / s)
    a = market.sigma * torch.sqrt(safe_tau)
    k = -1.0 if option_type == "call" else 1.0
    normal_cdf = lambda z: 0.5 * torch.erfc(-z / math.sqrt(2.0))
    folded_exp = (torch.exp(k * x + 0.5 * a.square()) * normal_cdf(x / a + k * a)
                  + torch.exp(-k * x + 0.5 * a.square()) * normal_cdf(-x / a + k * a))
    lift = s * (1.0 - folded_exp) if option_type == "call" else s * (folded_exp - 1.0)
    payoff = s - y if option_type == "call" else y - s
    return torch.where(alive, lift, payoff)


def model_value(model: MLP, market: Market, t: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Evaluate V(t, S, y), where y is running min for call and running max for put."""
    mode = getattr(model, "price_parameterization", "direct")
    if mode in ("hard_lift", "heat_lift"):
        tau = torch.clamp(market.T - t, min=0.0)
        x = torch.log(s / y) if model.option_type == "call" else torch.log(y / s)
        # Resolves the sqrt(tau)-wide layer; even in x to preserve Neumann BC.
        safe_tau = torch.where(tau > 0, tau, torch.full_like(tau, market.T))
        ratio_coordinate = x.square() / (x.square() + market.sigma**2 * safe_tau)
        features = torch.stack([t / market.T, torch.log(s / market.s0), ratio_coordinate], dim=1)
        correction = s * (tau / market.T) * model(features)
        if mode == "hard_lift":
            # Algebraic smoothing, not a heat-equation solution. For tau>0,
            # radius is even in x and hence has zero extremum partial at x=0.
            # Its terminal limit is |x|=x on the physical state domain.
            radius = torch.sqrt(x.square() + market.sigma**2 * safe_tau)
            base = s * (-torch.expm1(-radius) if model.option_type == "call" else torch.expm1(radius))
            payoff = s-y if model.option_type == "call" else y-s
            base = torch.where(tau > 0, base, payoff)
        else:
            base = reflected_heat_payoff(market, t, s, y, model.option_type)
        return base + correction
    features = torch.stack(
        [t / market.T,
         torch.log(torch.clamp(s, min=1e-8) / market.s0),
         torch.log(torch.clamp(y, min=1e-8) / market.s0)], dim=1)
    raw = market.s0 * model(features)
    if mode == "terminal_linear":
        payoff = s - y if model.option_type == "call" else y - s
        return payoff + (1.0 - t / market.T) * raw
    return raw


def bs_residual_finite_difference(
    model: MLP,
    market: Market,
    t: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    ds_rel: float,
    dt: float,
    option_type: OptionType | None = None,
) -> torch.Tensor:
    """Centered FD; optional domain-safe one-sided stencils near the extremum."""
    h = torch.clamp(ds_rel * s, min=1e-5)
    if option_type is not None:
        if option_type not in ("call", "put"):
            raise ValueError("Unsupported lookback option type")
        # Bound the inward put stencil away from S=0, including tiny spot values.
        h = torch.minimum(h, 0.25 * s)
        near = (s - h < y) if option_type == "call" else (s + h > y)
        s_plus = s + h
        s_minus = s - h
        if option_type == "call":
            s_minus = torch.where(near, s, s_minus)
        else:
            s_plus = torch.where(near, s, s_plus)
    else:
        near = None
        s_plus, s_minus = s + h, s - h
    v = model_value(model, market, t, s, y)
    v_t_plus = model_value(model, market, t + dt, s, y)
    v_t_minus = model_value(model, market, t - dt, s, y)
    v_s_plus = model_value(model, market, t, s_plus, y)
    v_s_minus = model_value(model, market, t, s_minus, y)

    v_t = (v_t_plus - v_t_minus) / (2.0 * dt)
    v_s = (v_s_plus - v_s_minus) / (2.0 * h)
    v_ss = (v_s_plus - 2.0 * v + v_s_minus) / (h**2)
    if near is not None and near.any():
        # Second-order inward FD: three-point first and four-point second derivative.
        # Keep h unchanged near S=y: shrinking it to the gap would amplify float32 cancellation.
        sign = 1.0 if option_type == "call" else -1.0
        tn, sn, yn, hn, vn = t[near], s[near], y[near], h[near], v[near]
        v1 = model_value(model, market, tn, sn + sign * hn, yn)
        v2 = model_value(model, market, tn, sn + sign * 2 * hn, yn)
        v3 = model_value(model, market, tn, sn + sign * 3 * hn, yn)
        v_s = v_s.clone()
        v_ss = v_ss.clone()
        v_s[near] = sign * (-3 * vn + 4 * v1 - v2) / (2 * hn)
        v_ss[near] = (2 * vn - 5 * v1 + 4 * v2 - v3) / hn.square()
    return v_t + market.r * s * v_s + 0.5 * market.sigma**2 * s**2 * v_ss - market.r * v



def bs_residual_autograd(
    model: MLP,
    market: Market,
    t: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Differentiate at fixed running extremum; retain graphs for training weights."""
    with torch.enable_grad():
        t = t.detach().requires_grad_(True)
        s = s.detach().requires_grad_(True)
        y = y.detach()
        value = model_value(model, market, t, s, y)
        v_t, v_s = torch.autograd.grad(value.sum(), (t, s), create_graph=True)
        v_ss = torch.autograd.grad(v_s.sum(), s, create_graph=True)[0]
        return v_t + market.r * s * v_s + 0.5 * market.sigma**2 * s.square() * v_ss - market.r * value


def sample_spot(batch_size: int, market: Market, device: torch.device, width: float = 0.65) -> torch.Tensor:
    return torch.exp(torch.randn(batch_size, device=device) * width + math.log(market.s0))


def sample_time(batch_size: int, market: Market, device: torch.device, dt: float) -> torch.Tensor:
    """Layered time sampling: interior, near t=0, and near maturity."""
    n_left = batch_size // 4
    n_right = batch_size // 4
    n_mid = batch_size - n_left - n_right
    left = dt + (0.15 * market.T - dt) * torch.rand(n_left, device=device)
    mid = 0.15 * market.T + 0.7 * market.T * torch.rand(n_mid, device=device)
    right = 0.85 * market.T + (market.T - dt - 0.85 * market.T) * torch.rand(n_right, device=device)
    t = torch.cat([left, mid, right], dim=0)
    return t[torch.randperm(batch_size, device=device)]


def sample_interior(
    batch_size: int,
    market: Market,
    contract: LookbackContract,
    device: torch.device,
    ds_rel: float,
    dt: float,
    interior_sampling: str = "legacy",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = sample_time(batch_size, market, device, dt)
    s = sample_spot(batch_size, market, device)
    if interior_sampling not in ("legacy", "full"):
        raise ValueError("interior_sampling must be legacy or full")
    gap = (max(5.0 * ds_rel, 1e-3) if interior_sampling == "legacy"
           else 8 * torch.finfo(s.dtype).eps)

    if contract.option_type == "call":
        n_near = batch_size // 3
        n_wide = batch_size - n_near
        s_near = s[:n_near]
        s_wide = s[n_near:]
        near_gap = gap + (0.20 - gap) * torch.rand(n_near, device=device)
        y_near = s_near * (1.0 - near_gap)
        y_wide = torch.clamp(s_wide * torch.rand(n_wide, device=device) * (1.0 - gap), min=1e-5)
        y = torch.cat([y_near, y_wide], dim=0)
    else:
        n_near = batch_size // 3
        n_wide = batch_size - n_near
        s_near = s[:n_near]
        s_wide = s[n_near:]
        near_gap = gap + (0.20 - gap) * torch.rand(n_near, device=device)
        y_near = s_near * (1.0 + near_gap)
        y_wide = s_wide * (1.0 + gap + 2.0 * torch.rand(n_wide, device=device))
        y = torch.cat([y_near, y_wide], dim=0)
    order = torch.randperm(batch_size, device=device)
    s = s[order]
    y = y[order]
    return t, s, y


def terminal_loss(model: MLP, market: Market, contract: LookbackContract, batch_size: int, device: torch.device) -> torch.Tensor:
    s = sample_spot(batch_size, market, device)
    if contract.option_type == "call":
        y = torch.clamp(s * torch.rand(batch_size, device=device), min=1e-5)
        payoff = s - y
    else:
        y = s * (1.0 + 2.0 * torch.rand(batch_size, device=device))
        payoff = y - s
    t = torch.full_like(s, market.T)
    pred = model_value(model, market, t, s, y)
    return ((pred - payoff) ** 2).mean()


def extremum_boundary_loss(
    model: MLP,
    market: Market,
    contract: LookbackContract,
    batch_size: int,
    device: torch.device,
    dy_rel: float,
    derivatives: str = "finite_difference",
) -> torch.Tensor:
    """Smooth-pasting condition at S = running min/max."""
    t = torch.rand(batch_size, device=device) * market.T
    s = sample_spot(batch_size, market, device)
    if derivatives == "autograd":
        # y must be an independent leaf: d/dy at fixed S, not along S=y.
        with torch.enable_grad():
            y = s.detach().clone().requires_grad_(True)
            value = model_value(model, market, t, s, y)
            v_y = torch.autograd.grad(value.sum(), y, create_graph=True)[0]
            return v_y.square().mean()
    if derivatives != "finite_difference":
        raise ValueError("boundary derivatives must be finite_difference or autograd")
    dy = torch.clamp(dy_rel * s, min=1e-5)

    if contract.option_type == "call":
        v_at = model_value(model, market, t, s, s)
        v_inside = model_value(model, market, t, s, torch.clamp(s - dy, min=1e-5))
        v_y = (v_at - v_inside) / dy
    else:
        v_at = model_value(model, market, t, s, s)
        v_inside = model_value(model, market, t, s, s + dy)
        v_y = (v_inside - v_at) / dy
    return (v_y**2).mean()


def nonnegative_loss(value: torch.Tensor) -> torch.Tensor:
    return torch.relu(-value).pow(2).mean()


def lookback_loss(
    model: MLP,
    market: Market,
    contract: LookbackContract,
    batch_size: int,
    device: torch.device,
    ds_rel: float,
    dy_rel: float,
    dt: float,
    smooth_weight: float,
    nonnegative_weight: float,
    pde_weight: float,
    terminal_weight: float,
    interior_sampling: str = "legacy",
    pde_derivatives: str = "finite_difference",
    boundary_derivatives: str = "finite_difference",
) -> tuple[torch.Tensor, dict[str, float]]:
    t, s, y = sample_interior(batch_size, market, contract, device, ds_rel, dt, interior_sampling)
    value = model_value(model, market, t, s, y)
    if pde_derivatives == "finite_difference":
        residual = bs_residual_finite_difference(model, market, t, s, y, ds_rel, dt,
            option_type=contract.option_type if interior_sampling == "full" else None)
    elif pde_derivatives == "autograd":
        residual = bs_residual_autograd(model, market, t, s, y)
    else:
        raise ValueError("pde_derivatives must be finite_difference or autograd")

    loss_pde = (residual**2).mean()
    loss_terminal = terminal_loss(model, market, contract, batch_size, device)
    loss_smooth = extremum_boundary_loss(model, market, contract, batch_size, device, dy_rel, boundary_derivatives)
    loss_nonnegative = nonnegative_loss(value)
    loss = pde_weight * loss_pde + terminal_weight * loss_terminal + smooth_weight * loss_smooth + nonnegative_weight * loss_nonnegative
    parts = {
        "pde": loss_pde.item(),
        "terminal": loss_terminal.item(),
        "smooth": loss_smooth.item(),
        "nonnegative": loss_nonnegative.item(),
    }
    return loss, parts


@torch.no_grad()
def initial_prediction(model: MLP, market: Market) -> float:
    t = torch.tensor([0.0], device=next(model.parameters()).device)
    s = torch.tensor([market.s0], device=t.device)
    y = torch.tensor([market.s0], device=t.device)
    return model_value(model, market, t, s, y).item()


def analytic_initial_price(market: Market, contract: LookbackContract) -> float:
    kwargs = {"running_min": market.s0} if contract.option_type == "call" else {"running_max": market.s0}
    return floating_lookback_price(
        spot=market.s0,
        tau=market.T,
        r=market.r,
        sigma=market.sigma,
        option_type=contract.option_type,
        **kwargs,
    )


def train(args: argparse.Namespace) -> None:
    recorder = RunRecorder(args, __spec__.name if __spec__ else __name__)
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    market = Market(s0=args.s0, r=args.r, sigma=args.sigma, T=args.T)
    contract = LookbackContract(option_type=args.option_type)
    model = MLP(hidden=args.hidden, depth=args.depth,
                price_parameterization=args.price_parameterization, option_type=args.option_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    dt = min(args.fd_t, market.T / 20.0)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = lookback_loss(
            model=model,
            market=market,
            contract=contract,
            batch_size=args.batch_size,
            device=device,
            ds_rel=args.fd_s_rel,
            dy_rel=args.fd_y_rel,
            dt=dt,
            smooth_weight=args.smooth_weight,
            nonnegative_weight=args.nonnegative_weight,
            pde_weight=args.pde_weight,
            terminal_weight=args.terminal_weight,
            interior_sampling=args.interior_sampling,
            pde_derivatives=args.pde_derivatives,
            boundary_derivatives=args.boundary_derivatives,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        recorder.log(epoch, loss, parts)
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch={epoch:05d} loss={loss.item():.6f} "
                f"pde={parts['pde']:.6f} terminal={parts['terminal']:.6f} "
                f"smooth={parts['smooth']:.6f} nonneg={parts['nonnegative']:.6f}"
            )

    pred = initial_prediction(model, market)
    ref = analytic_initial_price(market, contract)
    abs_err = abs(pred - ref)
    rel_err = abs_err / max(abs(ref), 1e-12)
    print(f"predicted_initial_price={pred:.8f}")
    print(f"analytic_initial_price={ref:.8f}")
    print(f"absolute_error={abs_err:.8f}")
    print(f"relative_error={rel_err:.4%}")
    recorder.finish(model, market, contract, pred, ref, None)



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit-state PINN for BS floating lookback options.")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--s0", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--price-parameterization", choices=["direct", "terminal_linear", "hard_lift", "heat_lift"], default="direct",
                        help="direct, terminal-linear experiment, or diffusion heat lift with exact payoff and extremum boundary")
    parser.add_argument("--boundary-derivatives", choices=["finite_difference", "autograd"], default="finite_difference",
                        help="Extremum partial derivative at fixed spot; autograd avoids secant bias")
    parser.add_argument("--pde-derivatives", choices=["finite_difference", "autograd"], default="finite_difference",
                        help="PDE derivatives only; terminal and boundary penalties unchanged")
    parser.add_argument("--interior-sampling", choices=["legacy", "full"], default="legacy",
                        help="Full ratio coverage uses inward FD near the extremum")
    parser.add_argument("--fd-s-rel", type=float, default=1e-2)
    parser.add_argument("--fd-y-rel", type=float, default=1e-2)
    parser.add_argument("--fd-t", type=float, default=1e-3)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=2.0)
    parser.add_argument("--smooth-weight", type=float, default=0.1)
    parser.add_argument("--nonnegative-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--lr-step", type=int, default=1000)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
