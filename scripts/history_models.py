"""Shared containers and immutable-target utilities for history-based BSDEs."""
import hashlib

import torch
from torch import nn


class DateValues(nn.Module):
    """Evaluate each date-specific value model using its observed prefix only."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, features, ts, spots, args):
        return torch.stack([
            self.models[i](features[:, :i+1], ts[:i+1], spots[:, :i+1], args)[:, -1]
            for i in range(len(ts))
        ], 1)


class DateZ(nn.Module):
    """Evaluate adapted Brownian-integral coefficients; expiry has zero Z."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, features, ts, spots, args):
        values = [
            self.models[i](features[:, :i+1], ts[:i+1], spots[:, :i+1], args)[:, -1]
            if i < len(self.models) else torch.zeros_like(spots[:, i])
            for i in range(len(ts))
        ]
        return torch.stack(values, 1)


def tensor_hash(model):
    """Hash parameter and buffer bytes to audit frozen regression targets."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze(model):
    """Freeze a completed date model and discard any stale gradients."""
    model.train(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
