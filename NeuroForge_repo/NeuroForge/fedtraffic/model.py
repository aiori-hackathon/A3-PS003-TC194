"""One compact forecaster shared by every method: GRU(h=32) + linear head.

Parameters: GRU 3*(4*32 + 32*32 + 32 + 32) = 3,648 ; head 32*3 + 3 = 99 ; total 3,747
-> 14,988 bytes in float32 (checked by tests/test_comm.py).
"""
import torch
from torch import nn


class GRUForecaster(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 32, horizons: int = 3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizons)

    def forward(self, x):                 # x: (B, 12, 4)
        _, h = self.gru(x)                # h: (1, B, hidden)
        return self.head(h[-1])           # (B, 3) in z-space


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_weights(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[torch.Tensor]) -> None:
    sd = model.state_dict()
    for k, w in zip(sd.keys(), weights):
        sd[k].copy_(w)
