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


class PathEncoder(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, num_layers=layers, batch_first=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(features)
        return encoded


class PriceHead(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim + 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, s: torch.Tensor, z: torch.Tensor, market: Market) -> torch.Tensor:
        if z.dim() == 3:
            t_feature = (t / market.T).unsqueeze(-1)
            s_feature = torch.log(torch.clamp(s, min=1e-8) / market.s0).unsqueeze(-1)
            features = torch.cat([t_feature, s_feature, z], dim=-1)
        else:
            t_feature = (t / market.T).unsqueeze(-1)
            s_feature = torch.log(torch.clamp(s, min=1e-8) / market.s0).unsqueeze(-1)
            features = torch.cat([t_feature, s_feature, z], dim=-1)
        return market.s0 * self.net(features).squeeze(-1)


class LSTMPINN(nn.Module):
    def __init__(self, encoder_hidden: int = 64, encoder_layers: int = 1, head_hidden: int = 128, head_depth: int = 3):
        super().__init__()
        self.encoder = PathEncoder(hidden=encoder_hidden, layers=encoder_layers)
        self.head = PriceHead(state_dim=encoder_hidden, hidden=head_hidden, depth=head_depth)

    def forward(self, paths: torch.Tensor, market: Market) -> tuple[torch.Tensor, torch.Tensor]:
        features = make_path_features(paths, market)
        z = self.encoder(features)
        ts = make_time_grid(market, paths.shape[1] - 1, paths.device).expand_as(paths)
        value = self.head(ts, paths, z, market)
        return value, z


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


def lookback_payoff(paths: torch.Tensor, contract: LookbackContract) -> torch.Tensor:
    terminal = paths[:, -1]
    if contract.option_type == "call":
        return terminal - paths.min(dim=1).values
    return paths.max(dim=1).values - terminal


def pde_residual(
    model: LSTMPINN,
    market: Market,
    paths: torch.Tensor,
    values: torch.Tensor,
    z: torch.Tensor,
    ds_rel: float,
    fd_t: float,
    time_mode: str,
) -> torch.Tensor:
    steps = paths.shape[1] - 1
    ts = make_time_grid(market, steps, paths.device).expand_as(paths)

    t = ts[:, 1:-1].reshape(-1)
    s = paths[:, 1:-1].reshape(-1)
    z_flat = z[:, 1:-1, :].reshape(-1, z.shape[-1])
    h = torch.clamp(ds_rel * s, min=1e-5)
    dt = min(fd_t, market.T / (2.0 * steps))
    path_dt = market.T / steps

    value = model.head(t, s, z_flat, market)
    if time_mode == "pathwise":
        value_t = ((values[:, 2:] - values[:, :-2]) / (2.0 * path_dt)).reshape(-1)
    elif time_mode == "frozen":
        value_t_plus = model.head(torch.clamp(t + dt, max=market.T), s, z_flat, market)
        value_t_minus = model.head(torch.clamp(t - dt, min=0.0), s, z_flat, market)
        value_t = (value_t_plus - value_t_minus) / (2.0 * dt)
    else:
        raise ValueError(f"unsupported time_mode: {time_mode}")
    value_s_plus = model.head(t, s + h, z_flat, market)
    value_s_minus = model.head(t, torch.clamp(s - h, min=1e-6), z_flat, market)

    value_s = (value_s_plus - value_s_minus) / (2.0 * h)
    value_ss = (value_s_plus - 2.0 * value + value_s_minus) / (h**2)
    return value_t + market.r * s * value_s + 0.5 * market.sigma**2 * s**2 * value_ss - market.r * value


def lstm_pinn_loss(
    model: LSTMPINN,
    market: Market,
    contract: LookbackContract,
    batch_size: int,
    steps: int,
    device: torch.device,
    ds_rel: float,
    fd_t: float,
    pde_weight: float,
    terminal_weight: float,
    nonnegative_weight: float,
    time_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    paths = simulate_gbm_paths(batch_size, steps, market, device)
    values, z = model(paths, market)
    payoff = lookback_payoff(paths, contract)

    residual = pde_residual(model, market, paths, values, z, ds_rel, fd_t, time_mode)
    loss_pde = (residual**2).mean()
    loss_terminal = ((values[:, -1] - payoff) ** 2).mean()
    loss_nonnegative = torch.relu(-values).pow(2).mean()
    loss = pde_weight * loss_pde + terminal_weight * loss_terminal + nonnegative_weight * loss_nonnegative
    parts = {
        "pde": loss_pde.item(),
        "terminal": loss_terminal.item(),
        "nonnegative": loss_nonnegative.item(),
    }
    return loss, parts


@torch.no_grad()
def initial_prediction(model: LSTMPINN, market: Market, eval_paths: int, steps: int, device: torch.device) -> float:
    paths = simulate_gbm_paths(eval_paths, steps, market, device)
    values, _ = model(paths, market)
    return values[:, 0].mean().item()


@torch.no_grad()
def discrete_mc_reference(
    market: Market,
    contract: LookbackContract,
    paths_count: int,
    steps: int,
    device: torch.device,
) -> float:
    paths = simulate_gbm_paths(paths_count, steps, market, device)
    payoff = lookback_payoff(paths, contract)
    return (math.exp(-market.r * market.T) * payoff.mean()).item()


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
    model = LSTMPINN(
        encoder_hidden=args.encoder_hidden,
        encoder_layers=args.encoder_layers,
        head_hidden=args.head_hidden,
        head_depth=args.head_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = lstm_pinn_loss(
            model=model,
            market=market,
            contract=contract,
            batch_size=args.batch_size,
            steps=args.steps,
            device=device,
            ds_rel=args.fd_s_rel,
            fd_t=args.fd_t,
            pde_weight=args.pde_weight,
            terminal_weight=args.terminal_weight,
            nonnegative_weight=args.nonnegative_weight,
            time_mode=args.time_mode,
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
                f"nonneg={parts['nonnegative']:.6f}"
            )

    pred = initial_prediction(model, market, args.eval_paths, args.steps, device)
    analytic_ref = analytic_initial_price(market, contract)
    discrete_ref = discrete_mc_reference(market, contract, args.mc_paths, args.steps, device)
    analytic_abs_err = abs(pred - analytic_ref)
    analytic_rel_err = analytic_abs_err / max(abs(analytic_ref), 1e-12)
    discrete_abs_err = abs(pred - discrete_ref)
    discrete_rel_err = discrete_abs_err / max(abs(discrete_ref), 1e-12)
    print(f"predicted_initial_price={pred:.8f}")
    print(f"continuous_analytic_initial_price={analytic_ref:.8f}")
    print(f"discrete_mc_initial_price={discrete_ref:.8f}")
    print(f"absolute_error_vs_continuous={analytic_abs_err:.8f}")
    print(f"relative_error_vs_continuous={analytic_rel_err:.4%}")
    print(f"absolute_error_vs_discrete_mc={discrete_abs_err:.8f}")
    print(f"relative_error_vs_discrete_mc={discrete_rel_err:.4%}")
    print(f"note=This is an LSTM-augmented PINN with {args.time_mode} time differencing, not a BSDE solver.")
    recorder.finish(model, market, contract, pred, analytic_ref, discrete_ref)



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM-augmented PINN for BS floating lookback options.")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--s0", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encoder-hidden", type=int, default=32)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--head-depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fd-s-rel", type=float, default=1e-2)
    parser.add_argument("--fd-t", type=float, default=1e-3)
    parser.add_argument("--pde-weight", type=float, default=0.2)
    parser.add_argument("--terminal-weight", type=float, default=5.0)
    parser.add_argument("--nonnegative-weight", type=float, default=0.01)
    parser.add_argument("--time-mode", choices=["frozen", "pathwise"], default="pathwise")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--lr-step", type=int, default=1000)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--eval-paths", type=int, default=2048)
    parser.add_argument("--mc-paths", type=int, default=50000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
