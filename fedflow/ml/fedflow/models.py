"""Pure-NumPy forecasting models with manual back-propagation.

Two models share one interface (params dict, forward, backward):
  * TemporalMLP  – compact edge model (also re-implemented in the browser, see web/src/engine)
  * GRUForecaster – recurrent model trained with back-propagation through time

Every model maps a window of the last 12 readings (+ time features) to flow
predictions at 4 horizons: +5, +15, +30 and +60 minutes.
"""
from __future__ import annotations
import numpy as np

HORIZONS = (1, 3, 6, 12)          # steps of 5 minutes
LAGS = 12                          # one hour of history


def _init(rng, fan_in, fan_out):
    lim = np.sqrt(6.0 / (fan_in + fan_out))   # Glorot uniform
    return rng.uniform(-lim, lim, size=(fan_in, fan_out)).astype(np.float64)


# ----------------------------------------------------------------------------------------------
class TemporalMLP:
    """x (B, LAGS + 3) -> 64 -> 32 -> 4.  Features: 12 normalised lags + sin/cos time-of-day + weekend."""

    name = "TemporalMLP"

    def __init__(self, n_in: int = LAGS + 3, h1: int = 64, h2: int = 32, n_out: int = len(HORIZONS)):
        self.shape = (n_in, h1, h2, n_out)

    def init(self, seed: int = 0) -> dict:
        n_in, h1, h2, n_out = self.shape
        rng = np.random.default_rng(seed)
        return {
            "W1": _init(rng, n_in, h1), "b1": np.zeros(h1),
            "W2": _init(rng, h1, h2), "b2": np.zeros(h2),
            "W3": _init(rng, h2, n_out), "b3": np.zeros(n_out),
        }

    @staticmethod
    def featurize(lags: np.ndarray, tod: np.ndarray, weekend: np.ndarray) -> np.ndarray:
        """lags (B, 12) normalised; tod (B,) in [0, 1); weekend (B,) 0/1."""
        ang = 2 * np.pi * tod
        return np.concatenate([lags, np.sin(ang)[:, None], np.cos(ang)[:, None], weekend[:, None]], axis=1)

    def forward(self, p: dict, x: np.ndarray):
        z1 = x @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0)
        y = a2 @ p["W3"] + p["b3"]
        return y, (x, z1, a1, z2, a2)

    def backward(self, p: dict, cache, dy: np.ndarray) -> dict:
        x, z1, a1, z2, a2 = cache
        g = {"W3": a2.T @ dy, "b3": dy.sum(0)}
        da2 = dy @ p["W3"].T; dz2 = da2 * (z2 > 0)
        g["W2"] = a1.T @ dz2; g["b2"] = dz2.sum(0)
        da1 = dz2 @ p["W2"].T; dz1 = da1 * (z1 > 0)
        g["W1"] = x.T @ dz1; g["b1"] = dz1.sum(0)
        return g


# ----------------------------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class GRUForecaster:
    """Sequence (B, T=12, 4) -> GRU(hidden) -> last hidden -> linear -> 4 horizons.
    Per-step input: [flow_norm, sin(tod), cos(tod), weekend]."""

    name = "GRU"

    def __init__(self, n_in: int = 4, hidden: int = 32, n_out: int = len(HORIZONS)):
        self.n_in, self.h, self.n_out = n_in, hidden, n_out

    def init(self, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        n, h = self.n_in, self.h
        p = {}
        for g in ("z", "r", "n"):
            p[f"W{g}"] = _init(rng, n, h)
            p[f"U{g}"] = np.linalg.qr(rng.normal(size=(h, h)))[0]   # orthogonal recurrent init
            p[f"b{g}"] = np.zeros(h)
        p["Wo"] = _init(rng, h, self.n_out); p["bo"] = np.zeros(self.n_out)
        return p

    @staticmethod
    def featurize(lags: np.ndarray, tod: np.ndarray, weekend: np.ndarray) -> np.ndarray:
        """Builds (B, 12, 4): time features of each lag step (5-minute spacing back from tod)."""
        B, T = lags.shape
        steps = (np.arange(T) - (T - 1)) / 288.0                 # -11/288 … 0
        t = (tod[:, None] + steps[None, :]) % 1.0
        ang = 2 * np.pi * t
        we = np.repeat(weekend[:, None], T, axis=1)
        return np.stack([lags, np.sin(ang), np.cos(ang), we], axis=2)

    def forward(self, p: dict, x: np.ndarray):
        B, T, _ = x.shape
        h = np.zeros((B, self.h))
        cache = []
        for t in range(T):
            xt = x[:, t, :]
            z = _sig(xt @ p["Wz"] + h @ p["Uz"] + p["bz"])
            r = _sig(xt @ p["Wr"] + h @ p["Ur"] + p["br"])
            hr = r * h
            n = np.tanh(xt @ p["Wn"] + hr @ p["Un"] + p["bn"])
            h_new = (1 - z) * n + z * h
            cache.append((xt, h, z, r, n, hr))
            h = h_new
        y = h @ p["Wo"] + p["bo"]
        return y, (cache, h)

    def backward(self, p: dict, cache_all, dy: np.ndarray) -> dict:
        cache, h_last = cache_all
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wo"] = h_last.T @ dy; g["bo"] = dy.sum(0)
        dh = dy @ p["Wo"].T
        for xt, h_prev, z, r, n, hr in reversed(cache):
            dn = dh * (1 - z)
            dz = dh * (h_prev - n)
            dh_prev = dh * z
            dan = dn * (1 - n ** 2)
            g["Wn"] += xt.T @ dan; g["Un"] += hr.T @ dan; g["bn"] += dan.sum(0)
            dhr = dan @ p["Un"].T
            dr = dhr * h_prev
            dh_prev += dhr * r
            daz = dz * z * (1 - z)
            dar = dr * r * (1 - r)
            g["Wz"] += xt.T @ daz; g["Uz"] += h_prev.T @ daz; g["bz"] += daz.sum(0)
            g["Wr"] += xt.T @ dar; g["Ur"] += h_prev.T @ dar; g["br"] += dar.sum(0)
            dh_prev += daz @ p["Uz"].T + dar @ p["Ur"].T
            dh = dh_prev
        return g


MODELS = {"mlp": TemporalMLP, "gru": GRUForecaster}


# ----------------------------------------------------------------------------------------------
class Adam:
    def __init__(self, params: dict, lr: float = 1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def huber_grad(pred: np.ndarray, target: np.ndarray, delta: float = 1.0):
    """Huber loss (robust to sensor spikes). Returns (loss, dL/dpred)."""
    e = pred - target
    a = np.abs(e)
    loss = np.where(a <= delta, 0.5 * e ** 2, delta * (a - 0.5 * delta)).mean()
    grad = np.where(a <= delta, e, delta * np.sign(e)) / e.size
    return float(loss), grad


# ---- helpers on parameter dicts ---------------------------------------------------------------
def flatten(p: dict) -> np.ndarray:
    return np.concatenate([p[k].ravel() for k in sorted(p)])


def unflatten(vec: np.ndarray, like: dict) -> dict:
    out, i = {}, 0
    for k in sorted(like):
        n = like[k].size
        out[k] = vec[i:i + n].reshape(like[k].shape).copy()
        i += n
    return out


def n_params(p: dict) -> int:
    return int(sum(v.size for v in p.values()))
