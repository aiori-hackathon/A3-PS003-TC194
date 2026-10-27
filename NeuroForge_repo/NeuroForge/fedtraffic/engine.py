"""Training and evaluation primitives shared by clients, the server and the baseline.

Keeping one train_epoch() and one evaluate() for every method is what makes the
centralised vs federated comparison fair (same loss, optimiser, batch size, metrics).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import WindowSet


def masked_l1(pred_z, y_z, mask):
    m = mask.float()
    return (torch.abs(pred_z - y_z) * m).sum() / m.sum().clamp_min(1.0)


def train_epoch(model: nn.Module, ws: WindowSet, lr: float, batch: int, rng: np.random.Generator,
                max_samples: int | None = None, clip: float = 1.0) -> float:
    """One local pass (E = 1) over a window set with Adam; returns mean training loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = rng.permutation(len(ws))
    if max_samples is not None:
        idx = idx[:max_samples]
    model.train()
    tot, nb = 0.0, 0
    for a in range(0, len(idx), batch):
        x, y, m = ws.batch(idx[a:a + batch])
        x = torch.from_numpy(x)
        y_z = torch.from_numpy((y - ws.mu) / ws.sd)
        m = torch.from_numpy(m)
        loss = masked_l1(model(x), y_z, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        tot += float(loss.detach())
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def error_sums(model: nn.Module, ws: WindowSet, batch: int = 32768) -> dict:
    """Return per-horizon sums of |e|, e^2 and counts in mph (masked)."""
    model.eval()
    sa = np.zeros(3); s2 = np.zeros(3); n = np.zeros(3)
    for a in range(0, len(ws), batch):
        x, y, m = ws.batch(np.arange(a, min(a + batch, len(ws))))
        pred = model(torch.from_numpy(x)).numpy() * ws.sd + ws.mu
        e = np.where(m, pred - y, 0.0)
        sa += np.abs(e).sum(0); s2 += (e ** 2).sum(0); n += m.sum(0)
    return {"abs": sa, "sq": s2, "n": n}


def combine(parts: list[dict]) -> dict:
    out = {"abs": np.zeros(3), "sq": np.zeros(3), "n": np.zeros(3)}
    for p in parts:
        for k in out:
            out[k] += p[k]
    return out


def metrics(sums: dict) -> dict:
    """MAE / RMSE pooled over all horizons, plus the +15 min horizon alone."""
    n = sums["n"].sum()
    return {
        "mae": float(sums["abs"].sum() / n),
        "rmse": float(np.sqrt(sums["sq"].sum() / n)),
        "mae_15": float(sums["abs"][2] / sums["n"][2]),
        "rmse_15": float(np.sqrt(sums["sq"][2] / sums["n"][2])),
    }
