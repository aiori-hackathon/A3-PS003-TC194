"""Module E (byte log): one payload-accounting function for every method.

Payload = tensor bytes + 8 B for the sample count n_k on uploads. Protocol headers,
TLS/TCP framing, retransmissions and compression by the transport are excluded for
every method: these are simulated payload sizes, not network measurements.

Innovation (extension beyond the mandated FedAvg scope): optional *update codecs* that
compress the client upload. The codec acts on the update delta (w_k - w_t), which is
what makes 8-bit quantisation accurate enough to be useful.
"""
from __future__ import annotations

import torch

COUNT_BYTES = 8  # n_k sent as int64 with every upload


def nbytes(weights: list[torch.Tensor], bytes_per_value: int = 4) -> int:
    """Float32 payload size of a list of tensors (numel x 4)."""
    return sum(w.numel() for w in weights) * bytes_per_value


class Codec:
    """Identity codec: float32 upload."""
    name = "fp32"

    def encode(self, delta: list[torch.Tensor]):
        return delta, nbytes(delta, 4)


class FP16Codec(Codec):
    name = "fp16"

    def encode(self, delta):
        out = [d.half().float() for d in delta]
        return out, nbytes(delta, 2)


class Int8Codec(Codec):
    """Symmetric per-tensor 8-bit quantisation of the update delta (1 B/value + 4 B scale)."""
    name = "int8"

    def encode(self, delta):
        out, size = [], 0
        for d in delta:
            scale = d.abs().max().clamp_min(1e-12) / 127.0
            q = torch.clamp(torch.round(d / scale), -127, 127)
            out.append(q * scale)
            size += d.numel() * 1 + 4
        return out, size


CODECS = {"fp32": Codec(), "fp16": FP16Codec(), "int8": Int8Codec()}


def round_bytes_analytic(n_values: int, K: int, codec: str = "fp32") -> int:
    """Expected bytes of one full-participation round (proposal Eq. B(t) = Bup + Bdown)."""
    down = K * n_values * 4
    if codec == "fp32":
        up = K * (n_values * 4 + COUNT_BYTES)
    elif codec == "fp16":
        up = K * (n_values * 2 + COUNT_BYTES)
    else:
        raise ValueError("analytic formula given for fp32/fp16 only")
    return down + up


class LinkModel:
    """Very simple network-condition model used to turn bytes into seconds.

    Each round: every participating client downloads |w| and uploads its payload over its own
    link; the round finishes when the slowest client finishes (synchronous FedAvg).
    """

    def __init__(self, up_mbps: float = 2.0, down_mbps: float = 10.0, rtt_ms: float = 60.0):
        self.up = up_mbps * 1e6 / 8
        self.down = down_mbps * 1e6 / 8
        self.rtt = rtt_ms / 1000

    def round_seconds(self, down_bytes: int, up_bytes: list[int]) -> float:
        if not up_bytes:
            return self.rtt
        return max(down_bytes / self.down + u / self.up + 2 * self.rtt for u in up_bytes)

    def stream_seconds(self, total_bytes: int) -> float:
        return total_bytes / self.up
