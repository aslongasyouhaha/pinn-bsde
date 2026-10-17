from __future__ import annotations

import math
from typing import Literal


OptionType = Literal["call", "put"]
KnockType = Literal["down-and-out", "down-and-in", "up-and-out", "up-and-in"]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _check_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _d(z: float, sigma: float, tau: float, r: float, plus: bool) -> float:
    _check_positive("z", z)
    _check_positive("sigma", sigma)
    _check_positive("tau", tau)
    sign = 1.0 if plus else -1.0
    return (math.log(z) + (r + sign * 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))


def bsm_price(
    spot: float,
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Black-Scholes price for a European call or put with no dividend yield."""
    _check_positive("spot", spot)
    _check_positive("strike", strike)
    _check_positive("sigma", sigma)
    if tau <= 0.0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    d1 = _d(spot / strike, sigma, tau, r, True)
    d2 = d1 - sigma * math.sqrt(tau)
    discount = math.exp(-r * tau)
    if option_type == "call":
        return spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    if option_type == "put":
        return strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)
    raise ValueError(f"unsupported option_type: {option_type}")


def floating_lookback_price(
    spot: float,
    tau: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    running_min: float | None = None,
    running_max: float | None = None,
) -> float:
    """Black-Scholes closed form for a floating-strike lookback option.

    For a call the payoff is S_T - min(S), so running_min is required.
    For a put the payoff is max(S) - S_T, so running_max is required.
    """
    _check_positive("spot", spot)
    _check_positive("sigma", sigma)
    if tau <= 0.0:
        if option_type == "call":
            min_s = spot if running_min is None else running_min
            return max(spot - min_s, 0.0)
        max_s = spot if running_max is None else running_max
        return max(max_s - spot, 0.0)
    if abs(r) < 1e-12:
        raise ValueError("floating lookback formula uses sigma^2 / (2r); use r away from zero or add the r=0 limit.")

    discount = math.exp(-r * tau)
    v2r = sigma**2 / (2.0 * r)
    power = 2.0 * r / sigma**2

    if option_type == "call":
        min_s = spot if running_min is None else running_min
        _check_positive("running_min", min_s)
        if min_s > spot:
            raise ValueError("running_min cannot exceed spot for a lookback call state")
        d1 = norm_cdf(_d(spot / min_s, sigma, tau, r, True))
        d2 = norm_cdf(_d(spot / min_s, sigma, tau, r, False))
        d3 = norm_cdf(_d(min_s / spot, sigma, tau, r, False))
        d4 = norm_cdf(-_d(spot / min_s, sigma, tau, r, True))
        return spot * d1 - min_s * discount * d2 + discount * spot * v2r * (min_s / spot) ** power * d3 - spot * v2r * d4

    if option_type == "put":
        max_s = spot if running_max is None else running_max
        _check_positive("running_max", max_s)
        if max_s < spot:
            raise ValueError("running_max cannot be below spot for a lookback put state")
        d1 = norm_cdf(-_d(spot / max_s, sigma, tau, r, False))
        d2 = norm_cdf(_d(spot / max_s, sigma, tau, r, True))
        d3 = norm_cdf(-_d(max_s / spot, sigma, tau, r, False))
        return max_s * discount * d1 + spot * (1.0 + v2r) * d2 - spot * discount * v2r * (max_s / spot) ** power * d3 - spot

    raise ValueError(f"unsupported option_type: {option_type}")


def barrier_price(
    spot: float,
    strike: float,
    barrier: float,
    tau: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    knock_type: KnockType,
    running_min: float | None = None,
    running_max: float | None = None,
) -> float:
    """Black-Scholes closed form for continuously monitored barrier options.

    running_min/running_max represent the observed path extrema up to the
    pricing time. At t=0 they can be omitted and default to spot.
    """
    _check_positive("spot", spot)
    _check_positive("strike", strike)
    _check_positive("barrier", barrier)
    _check_positive("sigma", sigma)
    if tau <= 0.0:
        alive_or_hit = _barrier_active(spot, spot, barrier, knock_type)
        payoff = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return payoff if alive_or_hit else 0.0

    min_s = spot if running_min is None else running_min
    max_s = spot if running_max is None else running_max
    _check_positive("running_min", min_s)
    _check_positive("running_max", max_s)
    discount = math.exp(-r * tau)
    power = 2.0 * r / sigma**2
    vanilla = bsm_price(spot, strike, tau, r, sigma, option_type)

    pb_down = 1.0 if min_s > barrier else 0.0
    pb_up = 1.0 if max_s < barrier else 0.0

    if option_type == "call" and knock_type == "down-and-out":
        if barrier <= strike:
            image = spot * (barrier / spot) ** power * bsm_price(barrier / spot, strike / barrier, tau, r, sigma, option_type)
            return pb_down * (vanilla - image)
        d1 = norm_cdf(_d(spot / barrier, sigma, tau, r, True))
        d2 = norm_cdf(_d(spot / barrier, sigma, tau, r, False))
        d3 = norm_cdf(_d(barrier / spot, sigma, tau, r, True))
        d4 = norm_cdf(_d(barrier / spot, sigma, tau, r, False))
        return pb_down * (spot * d1 - discount * strike * d2 - barrier * (barrier / spot) ** power * d3 + discount * strike * (spot / barrier) ** (1.0 - power) * d4)

    if option_type == "call" and knock_type == "down-and-in":
        if barrier <= strike:
            d1 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, True))
            d2 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, False))
            return (1.0 - pb_down) * vanilla + pb_down * (spot * (barrier / spot) ** (1.0 + power) * d1 - discount * strike * (spot / barrier) ** (1.0 - power) * d2)
        d1 = norm_cdf(_d(spot / barrier, sigma, tau, r, True))
        d2 = norm_cdf(_d(spot / barrier, sigma, tau, r, False))
        d3 = norm_cdf(_d(barrier / spot, sigma, tau, r, True))
        d4 = norm_cdf(_d(barrier / spot, sigma, tau, r, False))
        return vanilla - pb_down * (spot * d1 - discount * strike * d2 - spot * (barrier / spot) ** (1.0 + power) * d3 + discount * strike * (spot / barrier) ** (1.0 - power) * d4)

    if option_type == "call" and knock_type == "up-and-out":
        if barrier <= strike:
            return 0.0
        d1 = norm_cdf(_d(spot / strike, sigma, tau, r, True))
        d2 = norm_cdf(_d(spot / barrier, sigma, tau, r, True))
        d3 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, True))
        d4 = norm_cdf(_d(barrier / spot, sigma, tau, r, True))
        d5 = norm_cdf(_d(spot / strike, sigma, tau, r, False))
        d6 = norm_cdf(_d(spot / barrier, sigma, tau, r, False))
        d7 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, False))
        d8 = norm_cdf(_d(barrier / spot, sigma, tau, r, False))
        return pb_up * (spot * (d1 - d2 - (barrier / spot) ** (1.0 + power) * (d3 - d4)) - discount * strike * (d5 - d6 - (spot / barrier) ** (1.0 - power) * (d7 - d8)))

    if option_type == "call" and knock_type == "up-and-in":
        if barrier <= strike:
            return vanilla
        d1 = norm_cdf(_d(spot / barrier, sigma, tau, r, True))
        d2 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, True))
        d3 = norm_cdf(_d(barrier / spot, sigma, tau, r, True))
        d4 = norm_cdf(_d(spot / barrier, sigma, tau, r, False))
        d5 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, False))
        d6 = norm_cdf(_d(barrier / spot, sigma, tau, r, False))
        return (1.0 - pb_up) * vanilla + pb_up * (spot * d1 + barrier * (barrier / spot) ** power * (d2 - d3) - discount * (strike * d4 + strike * (spot / barrier) ** (1.0 - power) * (d5 - d6)))

    if option_type == "put" and knock_type == "down-and-out":
        if barrier > strike:
            return 0.0
        d1 = norm_cdf(-_d(spot / barrier, sigma, tau, r, True))
        d2 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, True))
        d3 = norm_cdf(_d(barrier / spot, sigma, tau, r, True))
        d4 = norm_cdf(-_d(spot / barrier, sigma, tau, r, False))
        d5 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, False))
        d6 = norm_cdf(_d(barrier / spot, sigma, tau, r, False))
        return pb_down * (vanilla + spot * d1 - barrier * (barrier / spot) ** power * (d2 - d3) - discount * strike * d4 + discount * strike * (spot / barrier) ** (1.0 - power) * (d5 - d6))

    if option_type == "put" and knock_type == "down-and-in":
        if barrier > strike:
            return vanilla
        d1 = norm_cdf(-_d(spot / barrier, sigma, tau, r, True))
        d2 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, True))
        d3 = norm_cdf(_d(barrier / spot, sigma, tau, r, True))
        d4 = norm_cdf(-_d(spot / barrier, sigma, tau, r, False))
        d5 = norm_cdf(_d(barrier**2 / (strike * spot), sigma, tau, r, False))
        d6 = norm_cdf(_d(barrier / spot, sigma, tau, r, False))
        return (1.0 - pb_down) * vanilla - pb_down * (spot * d1 - barrier * (barrier / spot) ** power * (d2 - d3) - discount * strike * d4 + discount * strike * (spot / barrier) ** (1.0 - power) * (d5 - d6))

    if option_type == "put" and knock_type == "up-and-out":
        if barrier <= strike:
            d1 = norm_cdf(-_d(spot / barrier, sigma, tau, r, True))
            d2 = norm_cdf(-_d(barrier / spot, sigma, tau, r, True))
            d3 = norm_cdf(-_d(spot / barrier, sigma, tau, r, False))
            d4 = norm_cdf(-_d(barrier / spot, sigma, tau, r, False))
            return pb_up * (spot * (-d1 + (barrier / spot) ** (1.0 + power) * d2) - strike * discount * (-d3 + (spot / barrier) ** (1.0 - power) * d4))
        d1 = norm_cdf(-_d(barrier**2 / (strike * spot), sigma, tau, r, True))
        d2 = norm_cdf(-_d(barrier**2 / (strike * spot), sigma, tau, r, False))
        return pb_up * (vanilla + spot * (barrier / spot) ** (1.0 + power) * d1 - strike * discount * (spot / barrier) ** (1.0 - power) * d2)

    if option_type == "put" and knock_type == "up-and-in":
        if barrier <= strike:
            d1 = norm_cdf(-_d(barrier / spot, sigma, tau, r, True))
            d2 = norm_cdf(-_d(spot / barrier, sigma, tau, r, True))
            d3 = norm_cdf(-_d(barrier / spot, sigma, tau, r, False))
            d4 = norm_cdf(-_d(spot / barrier, sigma, tau, r, False))
            return vanilla - pb_up * (spot * ((barrier / spot) ** (1.0 + power) * d1 - d2) - strike * discount * ((spot / barrier) ** (1.0 - power) * d3 - d4))
        d1 = norm_cdf(-_d(barrier**2 / (strike * spot), sigma, tau, r, True))
        d2 = norm_cdf(-_d(barrier**2 / (strike * spot), sigma, tau, r, False))
        return (1.0 - pb_up) * vanilla - pb_up * (spot * (barrier / spot) ** (1.0 + power) * d1 - strike * discount * (spot / barrier) ** (1.0 - power) * d2)

    raise ValueError(f"unsupported barrier contract: {option_type} {knock_type}")


def _barrier_active(running_min: float, running_max: float, barrier: float, knock_type: KnockType) -> bool:
    if knock_type == "down-and-out":
        return running_min > barrier
    if knock_type == "down-and-in":
        return running_min <= barrier
    if knock_type == "up-and-out":
        return running_max < barrier
    if knock_type == "up-and-in":
        return running_max >= barrier
    raise ValueError(f"unsupported knock_type: {knock_type}")


def barrier_price_by_parity(
    spot: float,
    strike: float,
    barrier: float,
    tau: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    knock_type: KnockType,
    running_min: float | None = None,
    running_max: float | None = None,
) -> float:
    """Compute a barrier price using vanilla = knock-in + knock-out parity."""
    if knock_type.endswith("-out"):
        return barrier_price(spot, strike, barrier, tau, r, sigma, option_type, knock_type, running_min, running_max)
    out_type = knock_type.replace("-in", "-out")
    vanilla = bsm_price(spot, strike, tau, r, sigma, option_type)
    out = barrier_price(spot, strike, barrier, tau, r, sigma, option_type, out_type, running_min, running_max)  # type: ignore[arg-type]
    return vanilla - out
