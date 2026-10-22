"""Pure CRR binomial baselines: European floating lookback and zero-rebate knockouts.

Only tree dates are monitored. No bridge correction, fitting or neural libraries.
Lookback uses the exact stock-numeraire ratio reduction of the same CRR tree,
starting with running extremum equal to spot; it does not merge different histories blindly.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
from .analytic_bs import barrier_price, floating_lookback_price


def _parameters(spot, maturity, r, sigma, steps, option_type):
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be call or put")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not all(math.isfinite(x) for x in [spot, maturity, r, sigma]):
        raise ValueError("Market parameters must be finite")
    if spot <= 0 or sigma <= 0 or maturity <= 0:
        raise ValueError("spot, sigma and maturity must be positive")
    dt = maturity / steps
    h = sigma * math.sqrt(dt)
    u, d, growth = math.exp(h), math.exp(-h), math.exp(r*dt)
    p = (math.expm1(r*dt) - math.expm1(-h)) / (u-d)
    if not 0 <= p <= 1:
        raise ValueError("CRR probability outside [0,1]; increase steps")
    return h, u, d, growth, p


def floating_lookback_tree(spot, maturity, r, sigma, steps, option_type="call"):
    """Fresh floating lookback: initial min/max equals spot. O(N^2) time, O(N) memory."""
    h, u, d, growth, p = _parameters(spot, maturity, r, sigma, steps, option_type)
    q = p*u/growth  # Stock-numeraire probability; no cash discount for V/S.
    j = np.arange(steps+1, dtype=np.float64)
    values = -np.expm1(-h*j) if option_type == "call" else np.expm1(h*j)
    for n in range(steps-1, -1, -1):
        # j=log(S/min)/h for call; log(max/S)/h for put. At zero the
        # extremum updates, so the step toward negative j stays at zero.
        reflected = np.concatenate((values[:1], values[:n]))
        if option_type == "call":
            values = q*values[1:n+2] + (1-q)*reflected
        else:
            values = q*reflected + (1-q)*values[1:n+2]
    return float(spot*values[0])


def barrier_tree(spot, strike, barrier, maturity, r, sigma, steps,
                 option_type="call", knock_type="up-and-out"):
    """European, zero rebate, monitored at every CRR date including t=0 and expiry."""
    h, u, d, growth, p = _parameters(spot, maturity, r, sigma, steps, option_type)
    if not all(math.isfinite(x) and x > 0 for x in [strike, barrier]):
        raise ValueError("strike and barrier must be finite and positive")
    if knock_type not in ("up-and-out", "down-and-out"):
        raise ValueError("Only zero-rebate up/down-and-out contracts are supported")
    log_spot, log_barrier = math.log(spot), math.log(barrier)
    tol = 16*np.finfo(float).eps*(1+abs(log_barrier))
    def alive(log_s):
        # Numerical equality with the barrier counts as a touch.
        return log_s < log_barrier-tol if knock_type == "up-and-out" else log_s > log_barrier+tol
    if not alive(log_spot):
        return 0.0
    log_s = log_spot + h*(2*np.arange(steps+1)-steps)
    s = np.exp(log_s)
    payoff = np.maximum(s-strike,0) if option_type == "call" else np.maximum(strike-s,0)
    values = np.where(alive(log_s),payoff,0.)
    for n in range(steps-1,-1,-1):
        values = ((1-p)*values[:-1]+p*values[1:])/growth
        log_s = log_spot + h*(2*np.arange(n+1)-n)
        values = np.where(alive(log_s),values,0.)
    return float(values[0])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",nargs="+",type=int,default=[32,64,128,256,512,1024,2048,4096,8192,16384,32768])
    parser.add_argument("--products",nargs="+",choices=["lookback","barrier_upout","barrier_downout"],default=["lookback","barrier_upout","barrier_downout"])
    parser.add_argument("--option-type",choices=["call","put"],default="call")
    parser.add_argument("--s0",type=float,default=1.)
    parser.add_argument("--strike",type=float,default=1.)
    parser.add_argument("--up-barrier",type=float,default=1.2)
    parser.add_argument("--down-barrier",type=float,default=.8)
    parser.add_argument("--r",type=float,default=.03)
    parser.add_argument("--sigma",type=float,default=.4)
    parser.add_argument("--T",type=float,default=1.)
    parser.add_argument("--output-dir",default="outputs/binomial_baseline")
    parser.add_argument("--no-plots",action="store_true",help="Use only numpy and standard library")
    args=parser.parse_args(argv)
    for n in args.steps:
        _parameters(args.s0,args.T,args.r,args.sigma,n,args.option_type)
    if len(set(args.products))!=len(args.products):
        parser.error("Products must not repeat")
    if "lookback" in args.products and abs(args.r)<1e-12:
        parser.error("The analytic lookback reference currently requires nonzero r; tree API supports r=0")
    rows=[]
    for product in args.products:
        barrier=args.up_barrier if product=="barrier_upout" else args.down_barrier
        knock="up-and-out" if product=="barrier_upout" else "down-and-out"
        analytic=(floating_lookback_price(args.s0,args.T,args.r,args.sigma,args.option_type,running_min=args.s0,running_max=args.s0)
            if product=="lookback" else barrier_price(args.s0,args.strike,barrier,args.T,args.r,args.sigma,args.option_type,knock))
        for n in sorted(set(args.steps)):
            start=time.perf_counter()
            price=(floating_lookback_tree(args.s0,args.T,args.r,args.sigma,n,args.option_type) if product=="lookback" else
                barrier_tree(args.s0,args.strike,barrier,args.T,args.r,args.sigma,n,args.option_type,knock))
            if not math.isfinite(price):
                raise FloatingPointError("Nonfinite tree price")
            rows.append(dict(product=product,option_type=args.option_type,steps=n,tree_price=price,
                continuous_analytic=analytic,signed_error=price-analytic,absolute_error=abs(price-analytic),
                relative_error=abs(price-analytic)/abs(analytic) if analytic else None,
                elapsed_seconds=time.perf_counter()-start,monitoring="CRR dates; initial extremum equals spot"))
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    with (output/"convergence.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/"config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,len(args.products),figsize=(5*len(args.products),7),squeeze=False,constrained_layout=True)
        for col,product in enumerate(args.products):
            data=[row for row in rows if row["product"]==product]
            ns=[row["steps"] for row in data]
            axes[0,col].plot(ns,[row["tree_price"] for row in data],"o-",label="Plain CRR")
            axes[0,col].axhline(data[0]["continuous_analytic"],color="black",label="Continuous analytic")
            axes[0,col].set(xscale="log",title=product,ylabel="Initial price")
            axes[0,col].legend()
            axes[1,col].plot(ns,[row["absolute_error"] for row in data],"o-")
            axes[1,col].set(xscale="log",yscale="log",xlabel="Tree steps",ylabel="Absolute error to continuous analytic")
        fig.savefig(output/"convergence.png",dpi=160)
        plt.close(fig)
    for product in args.products:
        print(json.dumps([row for row in rows if row["product"]==product][-1]))
    print("Saved to",output.resolve())

if __name__=="__main__":
    main()
