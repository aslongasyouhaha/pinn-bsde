from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

try:
    import signatory
except ImportError as exc:
    signatory = None
    SIGNATORY_IMPORT_ERROR = exc
except Exception as exc:
    signatory = None
    SIGNATORY_IMPORT_ERROR = exc

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


def require_signatory() -> None:
    if signatory is None:
        raise RuntimeError(
            "The paper-style signature BSDE requires the `signatory` package, "
            f"but it could not be imported: {SIGNATORY_IMPORT_ERROR}"
        )


class SignatureBSDE(nn.Module):
    def __init__(self, sig_dim: int, hidden: int = 64, layers: int = 1, head_hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=sig_dim, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
            nn.Softplus(),
        )
        self.z_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, signatures: torch.Tensor, market: Market) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.lstm(signatures)
        y = market.s0 * self.value_head(hidden).squeeze(-1)
        z = self.z_head(hidden).squeeze(-1)
        return y, z


def make_time_grid(market: Market, steps: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, market.T, steps + 1, device=device)


def simulate_gbm_paths(
    batch_size: int,
    steps: int,
    market: Market,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = market.T / steps
    d_w = math.sqrt(dt) * torch.randn(batch_size, steps, device=device)
    log_increments = (market.r - 0.5 * market.sigma**2) * dt + market.sigma * d_w
    log_paths = torch.cat(
        [
            torch.full((batch_size, 1), math.log(market.s0), device=device),
            math.log(market.s0) + torch.cumsum(log_increments, dim=1),
        ],
        dim=1,
    )
    return torch.exp(log_paths), d_w


def lead_lag_with_time(path: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    batch_size = path.shape[0]
    t = times.reshape(1, -1, 1).repeat(batch_size, 1, 1)
    t_rep = torch.repeat_interleave(t, repeats=3, dim=1)
    x_rep = torch.repeat_interleave(path.unsqueeze(-1), repeats=3, dim=1)
    return torch.cat([t_rep[:, :-2], x_rep[:, 1:-1], x_rep[:, 2:]], dim=2)


def signature_stream(
    paths: torch.Tensor,
    brownian_increments: torch.Tensor,
    market: Market,
    depth: int,
    lag: int,
    use_log_path: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    require_signatory()
    batch_size, length = paths.shape
    steps = length - 1
    if steps % lag != 0:
        raise ValueError(f"steps must be divisible by lag, got steps={steps}, lag={lag}")

    device = paths.device
    ts = make_time_grid(market, steps, device)
    coarse_ids = list(range(0, steps + 1, lag))
    sig_dim = signatory.logsignature_channels(in_channels=3, depth=depth)
    signatures = torch.zeros(batch_size, len(coarse_ids), sig_dim, device=device)
    coarse_d_w = torch.zeros(batch_size, len(coarse_ids), device=device)

    transformed_paths = torch.log(torch.clamp(paths, min=1e-8) / market.s0) if use_log_path else paths
    basepoint = torch.zeros(batch_size, 1, device=device)

    for idx, id_t in enumerate(coarse_ids):
        if idx == 0:
            interval_times = ts[:1].repeat(2)
            interval_path = torch.cat([basepoint, transformed_paths[:, :1]], dim=1)
        else:
            start = id_t - lag
            interval_times = ts[start : id_t + 1]
            interval_path = transformed_paths[:, start : id_t + 1]
            coarse_d_w[:, idx - 1] = brownian_increments[:, start:id_t].sum(dim=1)

        augmented = lead_lag_with_time(interval_path, interval_times)
        signatures[:, idx, :] = signatory.logsignature(augmented, depth)

    return signatures, coarse_d_w


def lookback_payoff(paths: torch.Tensor, contract: LookbackContract) -> torch.Tensor:
    terminal = paths[:, -1]
    if contract.option_type == "call":
        return terminal - paths.min(dim=1).values
    return paths.max(dim=1).values - terminal


def bsde_loss(
    model: SignatureBSDE,
    market: Market,
    contract: LookbackContract,
    batch_size: int,
    steps: int,
    depth: int,
    lag: int,
    device: torch.device,
    terminal_weight: float,
    nonnegative_weight: float,
    use_log_path: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    paths, d_w = simulate_gbm_paths(batch_size, steps, market, device)
    signatures, coarse_d_w = signature_stream(paths, d_w, market, depth, lag, use_log_path)
    y, z = model(signatures, market)
    payoff = lookback_payoff(paths, contract)

    coarse_dt = market.T / (signatures.shape[1] - 1)
    discount = math.exp(-market.r * coarse_dt)
    target = discount * y[:, 1:].detach()
    pred = y[:, :-1] + z[:, :-1] * coarse_d_w[:, :-1]

    bsde_step_loss = ((pred - target) ** 2).mean()
    terminal_loss = ((y[:, -1] - payoff) ** 2).mean()
    nonnegative_loss = torch.relu(-y).pow(2).mean()
    loss = bsde_step_loss + terminal_weight * terminal_loss + nonnegative_weight * nonnegative_loss
    parts = {
        "bsde": bsde_step_loss.item(),
        "terminal": terminal_loss.item(),
        "nonnegative": nonnegative_loss.item(),
    }
    return loss, parts


@torch.no_grad()
def initial_prediction(
    model: SignatureBSDE,
    market: Market,
    eval_paths: int,
    steps: int,
    depth: int,
    lag: int,
    device: torch.device,
    use_log_path: bool,
) -> float:
    paths, d_w = simulate_gbm_paths(eval_paths, steps, market, device)
    signatures, _ = signature_stream(paths, d_w, market, depth, lag, use_log_path)
    y, _ = model(signatures, market)
    return y[:, 0].mean().item()


@torch.no_grad()
def discrete_mc_reference(
    market: Market,
    contract: LookbackContract,
    paths_count: int,
    steps: int,
    device: torch.device,
) -> float:
    paths, _ = simulate_gbm_paths(paths_count, steps, market, device)
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
    require_signatory()
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.steps % args.lag != 0:
        raise ValueError("--steps must be divisible by --lag")

    market = Market(s0=args.s0, r=args.r, sigma=args.sigma, T=args.T)
    contract = LookbackContract(option_type=args.option_type)
    sig_dim = signatory.logsignature_channels(in_channels=3, depth=args.depth)
    model = SignatureBSDE(sig_dim=sig_dim, hidden=args.hidden, layers=args.layers, head_hidden=args.head_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = bsde_loss(
            model=model,
            market=market,
            contract=contract,
            batch_size=args.batch_size,
            steps=args.steps,
            depth=args.depth,
            lag=args.lag,
            device=device,
            terminal_weight=args.terminal_weight,
            nonnegative_weight=args.nonnegative_weight,
            use_log_path=args.use_log_path,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        recorder.log(epoch, loss, parts)
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch={epoch:05d} loss={loss.item():.6f} "
                f"bsde={parts['bsde']:.6f} terminal={parts['terminal']:.6f} "
                f"nonneg={parts['nonnegative']:.6f}"
            )

    pred = initial_prediction(model, market, args.eval_paths, args.steps, args.depth, args.lag, device, args.use_log_path)
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
    print("note=This is the paper-style LSTM-BSDE method using lead-lag logsignatures.")
    recorder.finish(model, market, contract, pred, analytic_ref, discrete_ref)



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style LSTM-BSDE solver with path logsignatures.")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--s0", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--lag", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--nonnegative-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--lr-step", type=int, default=1000)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--eval-paths", type=int, default=4096)
    parser.add_argument("--mc-paths", type=int, default=50000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-log-path", action="store_true", default=False)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
