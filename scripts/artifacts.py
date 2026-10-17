"""Shared, full-precision experiment output for all three solver families."""
from __future__ import annotations

import csv
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch


class RunRecorder:
    def __init__(self, args, module):
        self.args = args
        self.module = module
        self.started = time.perf_counter()
        self.history = []

    def log(self, epoch, loss, parts):
        row = {"epoch": epoch, "loss": float(loss.detach().cpu()), **parts}
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {row}")
        self.history.append(row)

    def finish(self, model, market, contract, prediction, analytic, reference=None):
        root = Path(self.args.output_dir) if self.args.output_dir else (
            Path(__file__).resolve().parents[1] / "outputs" / "standalone" /
            time.strftime("%Y%m%d-%H%M%S") / self.module.split(".")[-1]
        )
        root.mkdir(parents=True, exist_ok=True)
        if not math.isfinite(prediction):
            raise FloatingPointError("Non-finite initial price")
        with (root / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        metrics = {
            "module": self.module,
            "prediction": prediction,
            "continuous_analytic": analytic,
            "absolute_error": abs(prediction - analytic),
            "relative_error": abs(prediction - analytic) / abs(analytic) if analytic else None,
            "native_mc_reference": reference,
            "elapsed_seconds": time.perf_counter() - self.started,
            "market": asdict(market), "contract": asdict(contract),
            "args": vars(self.args),
            "python": platform.python_version(), "torch": torch.__version__,
        }
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        torch.save({"state_dict": model.state_dict(), "args": vars(self.args),
                    "market": asdict(market), "contract": asdict(contract)}, root / "model.pt")
        print(f"artifacts={root.resolve()}")
