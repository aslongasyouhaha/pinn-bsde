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


class PathEncoder(nn.Module):
    def __init__(self, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, num_layers=layers, batch_first=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(features)
        return encoded


class PriceHead(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim + 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        output = nn.Linear(in_dim, 1)
        nn.init.constant_(output.bias, 1.5)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, s: torch.Tensor, z: torch.Tensor, market: Market, contract: BarrierContract) -> torch.Tensor:
        features = torch.cat(
            [
                (t / market.T).unsqueeze(-1),
                torch.log(torch.clamp(s, min=1e-8) / market.s0).unsqueeze(-1),
                z,
            ],
            dim=-1,
        )
        raw = self.net(features).squeeze(-1)
        vanilla_bound = tensor_bsm_price(s, market.T - torch.clamp(t, 0.0, market.T), market, contract)
        return vanilla_bound * torch.sigmoid(raw)


class LSTMBarrierPINN(nn.Module):
    def __init__(self, encoder_hidden: int = 32, encoder_layers: int = 1, head_hidden: int = 64, head_depth: int = 3):
        super().__init__()
        self.encoder = PathEncoder(hidden=encoder_hidden, layers=encoder_layers)
        self.head = PriceHead(state_dim=encoder_hidden, hidden=head_hidden, depth=head_depth)

    def forward(self, paths: torch.Tensor, market: Market, contract: BarrierContract) -> tuple[torch.Tensor, torch.Tensor]:
        features = make_path_features(paths, market)
        z = self.encoder(features)
        ts = make_time_grid(market, paths.shape[1] - 1, paths.device).expand_as(paths)
        values = self.head(ts, paths, z, market, contract)
        return values, z


def make_time_grid(market: Market, steps: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, market.T, steps + 1, device=device)


def simulate_gbm_paths(batch_size: int, steps: int, market: Market, device: torch.device) -> torch.Tensor:
    dt = market.T / steps
    z = torch.randn(batch_size, steps, device=device)
    log_increments = (market.r - 0.5 * market.sigma**2) * dt + market.sigma * math.sqrt(dt) * z
    log_paths = torch.cat(
        [
            torch.full((batch_size, 1), math.log(market.s0), device=device),
            math.log(market.s0) + torch.cumsum(log_increments, dim=1),
        ],
        dim=1,
    )
    return torch.exp(log_paths)


def make_path_features(paths: torch.Tensor, market: Market) -> torch.Tensor:
    batch_size, length = paths.shape
    ts = make_time_grid(market, length - 1, paths.device).expand(batch_size, -1)
    log_s = torch.log(torch.clamp(paths, min=1e-8) / market.s0)
    return torch.stack([ts / market.T, log_s], dim=-1)


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


def downout_payoff(paths: torch.Tensor, contract: BarrierContract) -> torch.Tensor:
    alive = (paths.min(dim=1).values > contract.barrier).to(paths.dtype)
    return alive * vanilla_payoff(paths[:, -1], contract)


def alive_mask(paths: torch.Tensor, contract: BarrierContract) -> torch.Tensor:
    running_min = -torch.cummax(-paths, dim=1).values
    return (running_min > contract.barrier) & (paths > contract.barrier)


def pde_residual(
    model: LSTMBarrierPINN,
    market: Market,
    paths: torch.Tensor,
    values: torch.Tensor,
    z: torch.Tensor,
    contract: BarrierContract,
    ds_rel: float,
) -> torch.Tensor:
    steps = paths.shape[1] - 1
    ts = make_time_grid(market, steps, paths.device).expand_as(paths)
    mask = alive_mask(paths, contract)[:, 1:-1]
    if not mask.any():
        return torch.zeros((), device=paths.device)

    t_all = ts[:, 1:-1][mask]
    s_all = paths[:, 1:-1][mask]
    z_all = z[:, 1:-1, :][mask]
    value = values[:, 1:-1][mask]
    value_t = ((values[:, 2:] - values[:, :-2]) / (2.0 * (market.T / steps)))[mask]

    h = torch.clamp(ds_rel * s_all, min=1e-5)
    v_s_plus = model.head(t_all, s_all + h, z_all, market, contract)
    v_s_minus = model.head(t_all, torch.clamp(s_all - h, min=1e-6), z_all, market, contract)
    v_s = (v_s_plus - v_s_minus) / (2.0 * h)
    v_ss = (v_s_plus - 2.0 * value + v_s_minus) / (h**2)
    return value_t + market.r * s_all * v_s + 0.5 * market.sigma**2 * s_all**2 * v_ss - market.r * value


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


@torch.no_grad()
def discrete_mc_reference(market: Market, contract: BarrierContract, paths_count: int, steps: int, device: torch.device) -> float:
    paths = simulate_gbm_paths(paths_count, steps, market, device)
    return (math.exp(-market.r * market.T) * bridge_downout_payoff(paths, market, contract).mean()).item()


def lstm_pinn_loss(
    model: LSTMBarrierPINN,
    market: Market,
    contract: BarrierContract,
    batch_size: int,
    steps: int,
    device: torch.device,
    ds_rel: float,
    pde_weight: float,
    terminal_weight: float,
    barrier_weight: float,
    mc_anchor_weight: float,
    mc_anchor_value: float,
    nonnegative_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    paths = simulate_gbm_paths(batch_size, steps, market, device)
    values, z = model(paths, market, contract)
    payoff = bridge_downout_payoff(paths, market, contract)

    residual = pde_residual(model, market, paths, values, z, contract, ds_rel)
    loss_pde = (residual**2).mean()
    loss_terminal = ((values[:, -1] - payoff) ** 2).mean()

    ts = make_time_grid(market, steps, device).expand_as(paths)
    z_flat = z.reshape(-1, z.shape[-1])
    boundary = model.head(ts.reshape(-1), torch.full_like(paths.reshape(-1), contract.barrier), z_flat, market, contract)
    loss_barrier = boundary.pow(2).mean()

    mc_anchor = torch.tensor([mc_anchor_value], device=device)
    loss_mc_anchor = (values[:, 0].mean() - mc_anchor).pow(2).mean()
    loss_nonnegative = torch.relu(-values).pow(2).mean()

    loss = (
        pde_weight * loss_pde
        + terminal_weight * loss_terminal
        + barrier_weight * loss_barrier
        + mc_anchor_weight * loss_mc_anchor
        + nonnegative_weight * loss_nonnegative
    )
    parts = {
        "pde": loss_pde.item(),
        "terminal": loss_terminal.item(),
        "barrier": loss_barrier.item(),
        "mc_anchor": loss_mc_anchor.item(),
    }
    return loss, parts


@torch.no_grad()
def initial_prediction(model: LSTMBarrierPINN, market: Market, contract: BarrierContract, eval_paths: int, steps: int, device: torch.device) -> float:
    paths = simulate_gbm_paths(eval_paths, steps, market, device)
    values, _ = model(paths, market, contract)
    return values[:, 0].mean().item()


def train(args: argparse.Namespace) -> None:
    recorder = RunRecorder(args, __spec__.name if __spec__ else __name__)
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    market = Market(s0=args.s0, r=args.r, sigma=args.sigma, T=args.T)
    contract = BarrierContract(option_type=args.option_type, strike=args.strike, barrier=args.barrier)
    model = LSTMBarrierPINN(args.encoder_hidden, args.encoder_layers, args.head_hidden, args.head_depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    mc_anchor_value = discrete_mc_reference(market, contract, args.mc_anchor_paths, args.steps, device) if args.mc_anchor_weight > 0 else 0.0
    print(f"bridge_mc_level_anchor={mc_anchor_value:.8f}")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = lstm_pinn_loss(
            model, market, contract, args.batch_size, args.steps, device, args.fd_s_rel,
            args.pde_weight, args.terminal_weight, args.barrier_weight,
            args.mc_anchor_weight, mc_anchor_value, args.nonnegative_weight
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
                f"mc_anchor={parts['mc_anchor']:.6f}"
            )

    pred = initial_prediction(model, market, contract, args.eval_paths, args.steps, device)
    analytic_ref = analytic_initial_price(market, contract)
    discrete_ref = discrete_mc_reference(market, contract, args.mc_paths, args.steps, device)
    vanilla_ref = vanilla_initial_price(market, contract)
    inferred_downin = vanilla_ref - pred
    analytic_downin = analytic_downin_initial_price(market, contract)
    mc_downin = vanilla_ref - discrete_ref
    print(f"predicted_initial_price={pred:.8f}")
    print(f"continuous_analytic_initial_price={analytic_ref:.8f}")
    print(f"bridge_mc_initial_price={discrete_ref:.8f}")
    print(f"relative_error_vs_continuous={abs(pred - analytic_ref) / max(abs(analytic_ref), 1e-12):.4%}")
    print(f"relative_error_vs_bridge_mc={abs(pred - discrete_ref) / max(abs(discrete_ref), 1e-12):.4%}")
    print(f"vanilla_initial_price={vanilla_ref:.8f}")
    print(f"inferred_downin_initial_price={inferred_downin:.8f}")
    print(f"continuous_analytic_downin_initial_price={analytic_downin:.8f}")
    print(f"bridge_mc_downin_initial_price={mc_downin:.8f}")
    print(f"downin_relative_error_vs_continuous={abs(inferred_downin - analytic_downin) / max(abs(analytic_downin), 1e-12):.4%}")
    print(f"downin_relative_error_vs_bridge_mc={abs(inferred_downin - mc_downin) / max(abs(mc_downin), 1e-12):.4%}")
    recorder.finish(model, market, contract, pred, analytic_ref, discrete_ref)



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM-augmented PINN for BS down-and-out barrier options.")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--s0", type=float, default=1.0)
    parser.add_argument("--strike", type=float, default=1.0)
    parser.add_argument("--barrier", type=float, default=0.8)
    parser.add_argument("--r", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encoder-hidden", type=int, default=32)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--head-depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fd-s-rel", type=float, default=1e-2)
    parser.add_argument("--pde-weight", type=float, default=0.2)
    parser.add_argument("--terminal-weight", type=float, default=5.0)
    parser.add_argument("--barrier-weight", type=float, default=2.0)
    parser.add_argument("--mc-anchor-weight", type=float, default=100.0)
    parser.add_argument("--mc-anchor-paths", type=int, default=50000)
    parser.add_argument("--nonnegative-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--lr-step", type=int, default=1000)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--eval-paths", type=int, default=4096)
    parser.add_argument("--mc-paths", type=int, default=50000)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
