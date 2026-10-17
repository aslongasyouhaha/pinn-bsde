from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ..analytic_bs import barrier_price, bsm_price
from ..artifacts import RunRecorder


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class Market:
    s0: float = 1.0
    r: float = 0.03
    sigma: float = 0.4
    T: float = 1.0


@dataclass(frozen=True)
class BarrierContract:
    option_type: OptionType = "call"
    strike: float = 1.0
    barrier: float = 0.8


class MLP(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        output = nn.Linear(in_dim, 1)
        nn.init.constant_(output.bias, 1.5)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def model_value(model: MLP, market: Market, contract: BarrierContract, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    features = torch.stack(
        [
            t / market.T,
            torch.log(torch.clamp(s, min=1e-8) / market.s0),
        ],
        dim=1,
    )
    raw = model(features)
    vanilla_bound = tensor_bsm_price(s, market.T - torch.clamp(t, 0.0, market.T), market, contract)
    return vanilla_bound * torch.sigmoid(raw)


def vanilla_payoff(s: torch.Tensor, contract: BarrierContract) -> torch.Tensor:
    if contract.option_type == "call":
        return torch.relu(s - contract.strike)
    return torch.relu(contract.strike - s)


def bridge_survival_prob(paths: torch.Tensor, market: Market, contract: BarrierContract) -> torch.Tensor:
    steps = paths.shape[1] - 1
    dt = market.T / steps
    log_paths = torch.log(torch.clamp(paths, min=1e-12))
    barrier_log = math.log(contract.barrier)
    x0 = log_paths[:, :-1]
    x1 = log_paths[:, 1:]
    above = (x0 > barrier_log) & (x1 > barrier_log)
    exponent = -2.0 * (x0 - barrier_log) * (x1 - barrier_log) / (market.sigma**2 * dt)
    hit_prob = torch.exp(torch.clamp(exponent, max=0.0))
    interval_survival = torch.where(above, torch.clamp(1.0 - hit_prob, 0.0, 1.0), torch.zeros_like(hit_prob))
    return interval_survival.prod(dim=1)


def bridge_downout_payoff(paths: torch.Tensor, market: Market, contract: BarrierContract) -> torch.Tensor:
    return bridge_survival_prob(paths, market, contract) * vanilla_payoff(paths[:, -1], contract)


@torch.no_grad()
def monte_carlo_level_anchor(
    market: Market,
    contract: BarrierContract,
    paths_count: int,
    steps: int,
    device: torch.device,
) -> float:
    dt = market.T / steps
    batch_size = min(paths_count, 10000)
    remaining = paths_count
    payoff_sum = 0.0
    seen = 0
    while remaining > 0:
        current = min(batch_size, remaining)
        z = torch.randn(current, steps, device=device)
        log_increments = (market.r - 0.5 * market.sigma**2) * dt + market.sigma * math.sqrt(dt) * z
        log_paths = torch.cat(
            [
                torch.full((current, 1), math.log(market.s0), device=device),
                math.log(market.s0) + torch.cumsum(log_increments, dim=1),
            ],
            dim=1,
        )
        paths = torch.exp(log_paths)
        payoff = bridge_downout_payoff(paths, market, contract)
        payoff_sum += payoff.sum().item()
        seen += current
        remaining -= current
    return math.exp(-market.r * market.T) * payoff_sum / seen


def sample_time(batch_size: int, market: Market, device: torch.device, dt: float) -> torch.Tensor:
    n_left = batch_size // 4
    n_right = batch_size // 4
    n_mid = batch_size - n_left - n_right
    left = dt + (0.15 * market.T - dt) * torch.rand(n_left, device=device)
    mid = 0.15 * market.T + 0.7 * market.T * torch.rand(n_mid, device=device)
    right = 0.85 * market.T + (market.T - dt - 0.85 * market.T) * torch.rand(n_right, device=device)
    t = torch.cat([left, mid, right], dim=0)
    return t[torch.randperm(batch_size, device=device)]


def sample_alive_spot(batch_size: int, market: Market, contract: BarrierContract, device: torch.device, ds_rel: float) -> torch.Tensor:
    gap = max(5.0 * ds_rel, 1e-3)
    n_near = batch_size // 3
    n_wide = batch_size - n_near
    lower = contract.barrier * (1.0 + gap)
    upper = max(3.0 * market.s0, lower * 1.5)
    near = contract.barrier * (1.0 + gap + 0.2 * torch.rand(n_near, device=device))
    wide = lower + (upper - lower) * torch.rand(n_wide, device=device)
    s = torch.cat([near, wide], dim=0)
    return s[torch.randperm(batch_size, device=device)]


def pde_residual(
    model: MLP,
    market: Market,
    contract: BarrierContract,
    t: torch.Tensor,
    s: torch.Tensor,
    ds_rel: float,
    fd_t: float,
) -> torch.Tensor:
    h = torch.clamp(ds_rel * s, min=1e-5)
    dt = min(fd_t, market.T / 20.0)
    v = model_value(model, market, contract, t, s)
    v_t_plus = model_value(model, market, contract, t + dt, s)
    v_t_minus = model_value(model, market, contract, t - dt, s)
    v_s_plus = model_value(model, market, contract, t, s + h)
    v_s_minus = model_value(model, market, contract, t, torch.clamp(s - h, min=1e-6))
    v_t = (v_t_plus - v_t_minus) / (2.0 * dt)
    v_s = (v_s_plus - v_s_minus) / (2.0 * h)
    v_ss = (v_s_plus - 2.0 * v + v_s_minus) / (h**2)
    return v_t + market.r * s * v_s + 0.5 * market.sigma**2 * s**2 * v_ss - market.r * v


def tensor_bsm_price(s: torch.Tensor, tau: torch.Tensor, market: Market, contract: BarrierContract) -> torch.Tensor:
    tau = torch.clamp(tau, min=1e-8)
    sqrt_tau = torch.sqrt(tau)
    k = torch.tensor(contract.strike, dtype=s.dtype, device=s.device)
    d1 = (torch.log(torch.clamp(s, min=1e-8) / k) + (market.r + 0.5 * market.sigma**2) * tau) / (market.sigma * sqrt_tau)
    d2 = d1 - market.sigma * sqrt_tau
    normal = torch.distributions.Normal(torch.tensor(0.0, device=s.device), torch.tensor(1.0, device=s.device))
    discount = torch.exp(-market.r * tau)
    if contract.option_type == "call":
        return s * normal.cdf(d1) - contract.strike * discount * normal.cdf(d2)
    return contract.strike * discount * normal.cdf(-d2) - s * normal.cdf(-d1)


def analytic_initial_price(market: Market, contract: BarrierContract) -> float:
    return barrier_price(
        spot=market.s0,
        strike=contract.strike,
        barrier=contract.barrier,
        tau=market.T,
        r=market.r,
        sigma=market.sigma,
        option_type=contract.option_type,
        knock_type="down-and-out",
        running_min=market.s0,
    )


def vanilla_initial_price(market: Market, contract: BarrierContract) -> float:
    return bsm_price(
        spot=market.s0,
        strike=contract.strike,
        tau=market.T,
        r=market.r,
        sigma=market.sigma,
        option_type=contract.option_type,
    )


def analytic_downin_initial_price(market: Market, contract: BarrierContract) -> float:
    return vanilla_initial_price(market, contract) - analytic_initial_price(market, contract)


def barrier_loss(
    model: MLP,
    market: Market,
    contract: BarrierContract,
    batch_size: int,
    device: torch.device,
    ds_rel: float,
    fd_t: float,
    pde_weight: float,
    terminal_weight: float,
    barrier_weight: float,
    mc_anchor_weight: float,
    mc_anchor_value: float,
    cap_weight: float,
    nonnegative_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    dt = min(fd_t, market.T / 20.0)
    t = sample_time(batch_size, market, device, dt)
    s = sample_alive_spot(batch_size, market, contract, device, ds_rel)
    value = model_value(model, market, contract, t, s)

    loss_pde = (pde_residual(model, market, contract, t, s, ds_rel, fd_t) ** 2).mean()

    s_terminal = sample_alive_spot(batch_size, market, contract, device, ds_rel)
    terminal = model_value(model, market, contract, torch.full_like(s_terminal, market.T), s_terminal)
    loss_terminal = ((terminal - vanilla_payoff(s_terminal, contract)) ** 2).mean()

    t_barrier = torch.rand(batch_size, device=device) * market.T
    s_barrier = torch.full((batch_size,), contract.barrier, device=device)
    loss_barrier = model_value(model, market, contract, t_barrier, s_barrier).pow(2).mean()

    t0 = torch.tensor([0.0], device=device)
    s0 = torch.tensor([market.s0], device=device)
    mc_anchor = torch.tensor([mc_anchor_value], device=device)
    loss_mc_anchor = (model_value(model, market, contract, t0, s0) - mc_anchor).pow(2).mean()

    vanilla = tensor_bsm_price(s, market.T - t, market, contract)
    loss_cap = torch.relu(value - vanilla).pow(2).mean()
    loss_nonnegative = torch.relu(-value).pow(2).mean()

    loss = (
        pde_weight * loss_pde
        + terminal_weight * loss_terminal
        + barrier_weight * loss_barrier
        + mc_anchor_weight * loss_mc_anchor
        + cap_weight * loss_cap
        + nonnegative_weight * loss_nonnegative
    )
    parts = {
        "pde": loss_pde.item(),
        "terminal": loss_terminal.item(),
        "barrier": loss_barrier.item(),
        "mc_anchor": loss_mc_anchor.item(),
        "cap": loss_cap.item(),
        "nonnegative": loss_nonnegative.item(),
    }
    return loss, parts


@torch.no_grad()
def initial_prediction(model: MLP, market: Market, contract: BarrierContract) -> float:
    device = next(model.parameters()).device
    return model_value(model, market, contract, torch.tensor([0.0], device=device), torch.tensor([market.s0], device=device)).item()


def train(args: argparse.Namespace) -> None:
    recorder = RunRecorder(args, __spec__.name if __spec__ else __name__)
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    market = Market(s0=args.s0, r=args.r, sigma=args.sigma, T=args.T)
    contract = BarrierContract(option_type=args.option_type, strike=args.strike, barrier=args.barrier)
    model = MLP(hidden=args.hidden, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    mc_anchor_value = monte_carlo_level_anchor(market, contract, args.mc_anchor_paths, args.mc_anchor_steps, device) if args.mc_anchor_weight > 0 else 0.0
    print(f"bridge_mc_level_anchor={mc_anchor_value:.8f}")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = barrier_loss(
            model,
            market,
            contract,
            args.batch_size,
            device,
            args.fd_s_rel,
            args.fd_t,
            args.pde_weight,
            args.terminal_weight,
            args.barrier_weight,
            args.mc_anchor_weight,
            mc_anchor_value,
            args.cap_weight,
            args.nonnegative_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        recorder.log(epoch, loss, parts)
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch={epoch:05d} loss={loss.item():.6f} pde={parts['pde']:.6f} "
                f"terminal={parts['terminal']:.6f} barrier={parts['barrier']:.6f} "
                f"mc_anchor={parts['mc_anchor']:.6f} cap={parts['cap']:.6f}"
            )

    pred = initial_prediction(model, market, contract)
    ref = analytic_initial_price(market, contract)
    discrete_ref = monte_carlo_level_anchor(market, contract, args.mc_paths, args.mc_steps, device)
    vanilla_ref = vanilla_initial_price(market, contract)
    inferred_downin = vanilla_ref - pred
    analytic_downin = analytic_downin_initial_price(market, contract)
    mc_downin = vanilla_ref - discrete_ref
    abs_err = abs(pred - ref)
    rel_err = abs_err / max(abs(ref), 1e-12)
    print(f"predicted_initial_price={pred:.8f}")
    print(f"continuous_analytic_initial_price={ref:.8f}")
    print(f"bridge_mc_initial_price={discrete_ref:.8f}")
    print(f"absolute_error={abs_err:.8f}")
    print(f"relative_error={rel_err:.4%}")
    print(f"relative_error_vs_bridge_mc={abs(pred - discrete_ref) / max(abs(discrete_ref), 1e-12):.4%}")
    print(f"vanilla_initial_price={vanilla_ref:.8f}")
    print(f"inferred_downin_initial_price={inferred_downin:.8f}")
    print(f"continuous_analytic_downin_initial_price={analytic_downin:.8f}")
    print(f"bridge_mc_downin_initial_price={mc_downin:.8f}")
    print(f"downin_relative_error_vs_continuous={abs(inferred_downin - analytic_downin) / max(abs(analytic_downin), 1e-12):.4%}")
    print(f"downin_relative_error_vs_bridge_mc={abs(inferred_downin - mc_downin) / max(abs(mc_downin), 1e-12):.4%}")
    recorder.finish(model, market, contract, pred, ref, discrete_ref)



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit-state PINN for BS down-and-out barrier options.")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--s0", type=float, default=1.0)
    parser.add_argument("--strike", type=float, default=1.0)
    parser.add_argument("--barrier", type=float, default=0.8)
    parser.add_argument("--r", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fd-s-rel", type=float, default=1e-2)
    parser.add_argument("--fd-t", type=float, default=1e-3)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=2.0)
    parser.add_argument("--barrier-weight", type=float, default=5.0)
    parser.add_argument("--mc-anchor-weight", type=float, default=100.0)
    parser.add_argument("--mc-anchor-paths", type=int, default=50000)
    parser.add_argument("--mc-anchor-steps", type=int, default=128)
    parser.add_argument("--mc-paths", type=int, default=50000)
    parser.add_argument("--mc-steps", type=int, default=128)
    parser.add_argument("--cap-weight", type=float, default=0.1)
    parser.add_argument("--nonnegative-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--lr-step", type=int, default=1000)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
