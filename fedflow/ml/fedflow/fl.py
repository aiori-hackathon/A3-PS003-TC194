"""Federated learning engine: clients (traffic zones), server, FedAvg / FedProx,
client-level differential privacy (clip + Gaussian noise), pairwise-mask secure aggregation,
random client dropout, and the non-federated baselines used for comparison."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
import numpy as np
from .models import Adam, huber_grad, flatten, unflatten, HORIZONS
from .data import PeMS, STEPS_PER_DAY


# ============================================================================== configuration
@dataclass
class FLConfig:
    model: str = "gru"            # "gru" | "mlp"
    strategy: str = "fedavg"      # "fedavg" | "fedprox"
    rounds: int = 40
    local_steps: int = 25         # mini-batches per client per round
    batch: int = 128
    lr: float = 3e-3
    mu: float = 0.01              # FedProx proximal weight
    clip: float = 0.0             # client-update L2 clip C (0 = off)
    noise: float = 0.0            # noise multiplier sigma (std = sigma*C on the aggregated sum)
    dropout: float = 0.0          # probability a client misses a round
    secure_agg: bool = True
    seed: int = 0
    eval_stride: int = 6


# ============================================================================== privacy accounting
def gaussian_rdp_epsilon(noise: float, rounds: int, delta: float = 1e-5) -> float:
    """(epsilon, delta) for `rounds` compositions of the Gaussian mechanism with noise multiplier
    `noise` (all clients participate, sensitivity = clip). RDP: eps_a = a / (2 sigma^2)."""
    if noise <= 0:
        return math.inf
    alphas = np.concatenate([np.linspace(1.01, 10, 400), np.arange(11, 512)])
    eps = rounds * alphas / (2 * noise ** 2) + np.log(1 / delta) / (alphas - 1)
    return float(eps.min())


# ============================================================================== secure aggregation
def pairwise_masks(ids: list[int], dim: int, round_seed: int) -> dict[int, np.ndarray]:
    """Each pair (i<j) agrees on a shared random vector m_ij (in practice via Diffie-Hellman key
    agreement); client i adds it, client j subtracts it. Masks cancel exactly in the sum, so the
    server only ever sees uniformly-noisy vectors of individual clients."""
    masks = {i: np.zeros(dim) for i in ids}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            m = np.random.default_rng((round_seed, i, j)).normal(scale=10.0, size=dim)
            masks[i] += m
            masks[j] -= m
    return masks


# ============================================================================== client
class Client:
    def __init__(self, cid: int, sensors: np.ndarray, data: PeMS, model, mu: float, sd: float, seed: int):
        self.cid, self.sensors, self.data, self.model = cid, sensors, data, model
        self.mu, self.sd = mu, sd
        self.s, self.t = data.windows(sensors, 0, data.t_train)       # local training windows (never shared)
        self.n = len(self.s)
        self.rng = np.random.default_rng(seed)

    def raw_bytes(self) -> int:
        return int(self.data.t_train * len(self.sensors) * 3 * 4)     # flow/occ/speed float32

    def sample(self, batch: int):
        i = self.rng.integers(0, self.n, batch)
        lags, y, tod, we = self.data.batch(self.s[i], self.t[i], self.mu, self.sd)
        return self.model.featurize(lags, tod, we), y

    def train(self, global_p: dict, cfg: FLConfig, steps: int | None = None):
        p = {k: v.copy() for k, v in global_p.items()}
        opt = Adam(p, cfg.lr)
        losses = []
        for _ in range(steps or cfg.local_steps):
            x, y = self.sample(cfg.batch)
            pred, cache = self.model.forward(p, x)
            loss, dy = huber_grad(pred, y)
            g = self.model.backward(p, cache, dy)
            if cfg.strategy == "fedprox":                              # proximal term mu/2 ||w - w_g||^2
                for k in g:
                    g[k] += cfg.mu * (p[k] - global_p[k])
            opt.step(p, g)
            losses.append(loss)
        return p, float(np.mean(losses))


# ============================================================================== evaluation
def evaluate(model, p: dict, data: PeMS, sensors: np.ndarray, mu: float, sd: float,
             t0: int, t1: int, stride: int = 6) -> dict:
    s, t = data.windows(sensors, t0, t1, stride)
    preds, trues = [], []
    for i in range(0, len(s), 8192):
        lags, y, tod, we = data.batch(s[i:i + 8192], t[i:i + 8192], mu, sd)
        yh, _ = model.forward(p, model.featurize(lags, tod, we))
        preds.append(yh * sd + mu); trues.append(y * sd + mu)
    return metrics(np.concatenate(preds), np.concatenate(trues))


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = np.maximum(pred, 0)
    out = {}
    for j, h in enumerate(HORIZONS):
        e = pred[:, j] - true[:, j]
        m = true[:, j] > 10                                             # MAPE ignores near-empty roads
        out[f"{h * 5}min"] = {
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "mape": float((np.abs(e[m]) / true[m, j]).mean() * 100),
        }
    out["avg"] = {k: float(np.mean([out[f"{h * 5}min"][k] for h in HORIZONS])) for k in ("mae", "rmse", "mape")}
    return out


# ============================================================================== federated training
@dataclass
class History:
    rounds: list = field(default_factory=list)


def federated_normalisation(clients: list[Client], data: PeMS):
    """Each client sends only (sum, sum of squares, count) of its training flow; the server derives
    the global mean / std. No raw readings leave the zone."""
    S = Q = C = 0.0
    for c in clients:
        x = data.flow[: data.t_train, c.sensors]
        S += x.sum(); Q += (x ** 2).sum(); C += x.size
    mu = S / C
    sd = math.sqrt(Q / C - mu ** 2)
    return mu, sd


def run_federated(data: PeMS, zones: np.ndarray, model, cfg: FLConfig, log=print):
    k = int(zones.max()) + 1
    rng = np.random.default_rng(cfg.seed)
    tmp = [Client(z, np.where(zones == z)[0], data, model, 0, 1, cfg.seed + z) for z in range(k)]
    mu, sd = federated_normalisation(tmp, data)
    clients = [Client(z, np.where(zones == z)[0], data, model, mu, sd, cfg.seed * 100 + z) for z in range(k)]
    gp = model.init(cfg.seed)
    dim = flatten(gp).size
    total_n = sum(c.n for c in clients)
    hist = []
    bytes_up = 0
    for r in range(1, cfg.rounds + 1):
        t_start = time.time()
        alive = [c for c in clients if rng.random() >= cfg.dropout] or [clients[int(rng.integers(k))]]
        gvec = flatten(gp)
        updates, weights, losses, norms = {}, {}, {}, {}
        for c in alive:
            lp, loss = c.train(gp, cfg)
            delta = flatten(lp) - gvec
            nrm = float(np.linalg.norm(delta))
            if cfg.clip > 0:                                   # clip the client update
                delta = delta * min(1.0, cfg.clip / (nrm + 1e-12))
            updates[c.cid] = delta; losses[c.cid] = loss; norms[c.cid] = nrm
            weights[c.cid] = c.n / total_n if cfg.clip == 0 else 1.0   # DP uses uniform weights (bounded sensitivity)
        ids = sorted(updates)
        if cfg.noise > 0 and cfg.clip > 0:                     # distributed noise: each adds N(0, (sigma C)^2 / m)
            for i in ids:
                updates[i] = updates[i] + rng.normal(scale=cfg.noise * cfg.clip / math.sqrt(len(ids)), size=dim)
        wsum = sum(weights[i] for i in ids)
        contrib = {i: updates[i] * (weights[i] / wsum) for i in ids}
        if cfg.secure_agg:
            masks = pairwise_masks(ids, dim, cfg.seed * 1000 + r)
            masked = {i: contrib[i] + masks[i] for i in ids}  # what the server actually receives
            agg = np.sum([masked[i] for i in ids], axis=0)
            assert np.allclose(agg, np.sum([contrib[i] for i in ids], axis=0), atol=1e-6), "secure-agg masks must cancel"
        else:
            agg = np.sum([contrib[i] for i in ids], axis=0)
        gp = unflatten(gvec + agg, gp)
        bytes_up += len(ids) * dim * 4
        val = evaluate(model, gp, data, np.arange(data.N), mu, sd, data.t_train, data.t_val, stride=12)
        rec = {"round": r, "clients": ids, "val_mae": val["avg"]["mae"], "val_rmse": val["avg"]["rmse"],
               "loss": float(np.mean(list(losses.values()))), "client_loss": losses, "update_norm": norms,
               "bytes_up": bytes_up, "secs": round(time.time() - t_start, 2)}
        hist.append(rec)
        if log and (r % 5 == 0 or r == 1):
            log(f"  [{cfg.model}/{cfg.strategy} σ={cfg.noise} drop={cfg.dropout}] round {r:3d}  val MAE {rec['val_mae']:.2f}  clients {len(ids)}")
    return gp, (mu, sd), clients, hist


# ============================================================================== baselines
def run_centralised(data: PeMS, zones, model, cfg: FLConfig, log=print):
    """Upper bound that ignores privacy: all raw data pooled in one place, same number of SGD steps."""
    k = int(zones.max()) + 1
    clients = [Client(z, np.where(zones == z)[0], data, model, 0, 1, z) for z in range(k)]
    mu, sd = federated_normalisation(clients, data)
    pooled = Client(-1, np.arange(data.N), data, model, mu, sd, cfg.seed)
    p = model.init(cfg.seed)
    opt = Adam(p, cfg.lr)
    for step in range(cfg.rounds * cfg.local_steps * 2):
        x, y = pooled.sample(cfg.batch)
        pred, cache = model.forward(p, x)
        _, dy = huber_grad(pred, y)
        opt.step(p, model.backward(p, cache, dy))
    return p, (mu, sd)


def run_local_only(data: PeMS, zones, model, cfg: FLConfig):
    """Each zone trains alone on its own sensors (no collaboration)."""
    k = int(zones.max()) + 1
    out = {}
    for z in range(k):
        sens = np.where(zones == z)[0]
        x = data.flow[: data.t_train, sens]
        mu, sd = x.mean(), x.std()
        c = Client(z, sens, data, model, mu, sd, cfg.seed + z)
        p = model.init(cfg.seed)
        p, _ = c.train(p, cfg, steps=cfg.rounds * cfg.local_steps)
        out[z] = (p, (mu, sd))
    return out


def historical_average(data: PeMS, sensors: np.ndarray, stride: int = 6) -> dict:
    """Mean flow for the same time-of-week in the training period."""
    week = 7 * STEPS_PER_DAY
    tr = data.flow[: data.t_train]
    idx = np.arange(data.t_train) % week
    prof = np.stack([tr[idx == w].mean(0) for w in range(week)])     # (2016, N)
    s, t = data.windows(sensors, data.t_val, data.T, stride)
    tgt = t[:, None] + (np.array(HORIZONS) - 1)[None, :]
    pred = prof[tgt % week, s[:, None]]
    true = data.flow[tgt, s[:, None]]
    return metrics(pred, true)


def last_value(data: PeMS, sensors: np.ndarray, stride: int = 6) -> dict:
    s, t = data.windows(sensors, data.t_val, data.T, stride)
    tgt = t[:, None] + (np.array(HORIZONS) - 1)[None, :]
    pred = np.repeat(data.flow[t - 1, s][:, None], len(HORIZONS), 1)
    true = data.flow[tgt, s[:, None]]
    return metrics(pred, true)
