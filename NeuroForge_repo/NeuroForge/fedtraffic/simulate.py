"""Modules D, E, F, G: clients, FedAvg coordinator, centralised baseline, evaluation.

Every run, centralised or federated, writes the same log schema
(seed, method, partition, round, bytes, validation metrics), so methods are compared on
accuracy *per byte*, not accuracy alone (proposal Section 5.3).
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .comm import CODECS, COUNT_BYTES, LinkModel, nbytes
from .data import Prepared, WindowSet
from .engine import combine, error_sums, metrics, train_epoch
from .model import GRUForecaster, get_weights, n_params, set_weights
from .partition import client_stats, heterogeneity_index, partition


@dataclass
class RunConfig:
    seed: int = 11
    method: str = "fedavg"        # fedavg | central | local
    partition: str = "iid"        # iid | noniid | speed   (ignored by central)
    K: int = 4
    rounds: int = 50              # R ; central / local train for R*E epochs
    local_epochs: int = 1         # E
    lr: float = 3e-3
    batch: int = 512
    train_stride: int = 4         # anchor stride for training windows (compute budget)
    val_stride: int = 3           # anchor stride for per-round validation
    codec: str = "fp32"           # fp32 | fp16 | int8   (upload compression, extension)
    dropout: float = 0.0          # probability a selected client fails to upload (network sim)
    up_mbps: float = 2.0
    down_mbps: float = 10.0
    rtt_ms: float = 60.0
    max_rounds_eval_test: bool = True
    save_weights: bool = False    # write the selected checkpoint to results/models/

    @property
    def run_id(self) -> str:
        if self.method == "central":
            return f"central_s{self.seed}"
        tag = f"{self.method}_{self.partition}_E{self.local_epochs}_{self.codec}"
        if self.dropout:
            tag += f"_drop{int(self.dropout * 100)}"
        return f"{tag}_s{self.seed}"


def save_model(model, run_id: str, out_dir: str = "results/models") -> str:
    """Save the validation-selected weights as JSON (portable: Python, browser, anything)."""
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, run_id + ".json")
    with open(path, "w") as f:
        json.dump({k: v.tolist() for k, v in model.state_dict().items()}, f)
    return path


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


class Client:
    """A simulated intersection: holds its own windows; raw data never leaves it."""

    def __init__(self, cid: int, sensors: np.ndarray, prep: Prepared, cfg: RunConfig):
        self.cid = cid
        self.sensors = sensors
        self.train = WindowSet(prep.train, sensors, prep.mu, prep.sd, cfg.train_stride)
        self.val = WindowSet(prep.val, sensors, prep.mu, prep.sd, cfg.val_stride)
        self.test = WindowSet(prep.test, sensors, prep.mu, prep.sd, 1)
        self.n = len(self.train)

    def local_update(self, model, global_w, cfg: RunConfig, rng):
        set_weights(model, global_w)
        for _ in range(cfg.local_epochs):
            train_epoch(model, self.train, cfg.lr, cfg.batch, rng)
        return get_weights(model)


def _stats_setup_bytes(K: int) -> int:
    """Clients share (count, sum, sum of squares) once for global z-score: 24 B each."""
    return 24 * K


def run_federated(prep: Prepared, cfg: RunConfig, log=print):
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    groups = partition(cfg.partition, prep.coords, prep.train.speed, cfg.K, cfg.seed)
    stats = client_stats(groups, prep.train.speed)
    clients = [Client(i + 1, g, prep, cfg) for i, g in enumerate(groups)]
    model = GRUForecaster()
    w = get_weights(model)
    codec = CODECS[cfg.codec]
    link = LinkModel(cfg.up_mbps, cfg.down_mbps, cfg.rtt_ms)
    w_bytes = nbytes(w)
    rows, ckpt = [], {}
    B = _stats_setup_bytes(cfg.K)
    sim_seconds = 0.0
    t0 = time.time()
    for t in range(1, cfg.rounds + 1):
        ups, up_sizes, down = [], [], 0
        for c in clients:
            down += w_bytes                                   # broadcast w_t
            if cfg.dropout and rng.random() < cfg.dropout:    # client drops after download
                continue
            wk = c.local_update(model, w, cfg, rng)
            delta = [a - b for a, b in zip(wk, w)]
            dec, size = codec.encode(delta)
            ups.append(([b + d for b, d in zip(w, dec)], c.n))
            up_sizes.append(size + COUNT_BYTES)
        if ups:                                               # FedAvg, Eq. (1)
            n = sum(k for _, k in ups)
            w = [sum((k / n) * wk[i] for wk, k in ups) for i in range(len(w))]
        b_up = sum(up_sizes)
        B += down + b_up
        sim_seconds += link.round_seconds(w_bytes, up_sizes)
        set_weights(model, w)
        ckpt[t] = copy.deepcopy(w)
        val = metrics(combine([error_sums(model, c.val) for c in clients]))
        rows.append({"round": t, "B_up": b_up, "B_down": down, "B_cum": B, "participants": len(ups),
                     "sim_comm_s": round(sim_seconds, 3), "val_mae": val["mae"], "val_rmse": val["rmse"],
                     "wall_s": round(time.time() - t0, 1)})
        log(f"[{cfg.run_id}] round {t:3d}  val MAE {val['mae']:.3f}  RMSE {val['rmse']:.3f}  Bcum {B/1e6:.2f} MB")
    best = min(rows, key=lambda r: r["val_mae"])
    set_weights(model, ckpt[best["round"]])
    if cfg.save_weights:
        save_model(model, cfg.run_id)
    per_client = []
    parts = []
    for c in clients:
        s = error_sums(model, c.test)
        parts.append(s)
        per_client.append({"client": c.cid, **metrics(s)})
    test = metrics(combine(parts))
    summary = {
        "run_id": cfg.run_id, **asdict(cfg), "params": n_params(model), "bytes_per_param_set": w_bytes,
        "best_round": best["round"], "B_at_best": best["B_cum"], "B_total": B,
        "sim_comm_s_total": round(sim_seconds, 2), "test": test, "per_client_test": per_client,
        "client_stats": stats, "heterogeneity_mph": heterogeneity_index(stats),
        "train_windows": sum(c.n for c in clients), "wall_s": round(time.time() - t0, 1),
    }
    return rows, summary


def run_central(prep: Prepared, cfg: RunConfig, log=print):
    """Same GRU on pooled training windows; epochs = R * E (equal data passes)."""
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    allsens = np.arange(prep.train.speed.shape[1])
    tr = WindowSet(prep.train, allsens, prep.mu, prep.sd, cfg.train_stride)
    va = WindowSet(prep.val, allsens, prep.mu, prep.sd, cfg.val_stride)
    te = WindowSet(prep.test, allsens, prep.mu, prep.sd, 1)
    model = GRUForecaster()
    raw_bytes = prep.train.speed.size * 4      # every training reading of every sensor, float32
    link = LinkModel(cfg.up_mbps, cfg.down_mbps, cfg.rtt_ms)
    rows, ckpt = [], {}
    t0 = time.time()
    for ep in range(1, cfg.rounds * cfg.local_epochs + 1):
        train_epoch(model, tr, cfg.lr, cfg.batch, rng)
        ckpt[ep] = get_weights(model)
        val = metrics(error_sums(model, va))
        rows.append({"round": ep, "B_up": raw_bytes if ep == 1 else 0, "B_down": 0, "B_cum": raw_bytes,
                     "participants": 1, "sim_comm_s": round(link.stream_seconds(raw_bytes), 3),
                     "val_mae": val["mae"], "val_rmse": val["rmse"], "wall_s": round(time.time() - t0, 1)})
        log(f"[{cfg.run_id}] epoch {ep:3d}  val MAE {val['mae']:.3f}  RMSE {val['rmse']:.3f}")
    best = min(rows, key=lambda r: r["val_mae"])
    set_weights(model, ckpt[best["round"]])
    if cfg.save_weights:
        save_model(model, cfg.run_id)
    test = metrics(error_sums(model, te))
    summary = {"run_id": cfg.run_id, **asdict(cfg), "params": n_params(model), "best_round": best["round"],
               "B_at_best": raw_bytes, "B_total": raw_bytes, "sim_comm_s_total": round(link.stream_seconds(raw_bytes), 2),
               "test": test, "train_windows": len(tr), "wall_s": round(time.time() - t0, 1)}
    return rows, summary


def run_local(prep: Prepared, cfg: RunConfig, log=print):
    """E4: every client trains alone for R*E epochs; zero communication floor."""
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    groups = partition(cfg.partition, prep.coords, prep.train.speed, cfg.K, cfg.seed)
    clients = [Client(i + 1, g, prep, cfg) for i, g in enumerate(groups)]
    models = [GRUForecaster() for _ in clients]
    best = [(np.inf, None) for _ in clients]
    rows = []
    t0 = time.time()
    for ep in range(1, cfg.rounds * cfg.local_epochs + 1):
        parts = []
        for i, (c, m) in enumerate(zip(clients, models)):
            train_epoch(m, c.train, cfg.lr, cfg.batch, rng)
            s = error_sums(m, c.val)
            parts.append(s)
            mae = metrics(s)["mae"]
            if mae < best[i][0]:
                best[i] = (mae, get_weights(m))
        val = metrics(combine(parts))
        rows.append({"round": ep, "B_up": 0, "B_down": 0, "B_cum": 0, "participants": len(clients),
                     "sim_comm_s": 0.0, "val_mae": val["mae"], "val_rmse": val["rmse"], "wall_s": round(time.time() - t0, 1)})
        log(f"[{cfg.run_id}] epoch {ep:3d}  val MAE {val['mae']:.3f}")
    parts, per_client = [], []
    for (mae, w), c, m in zip(best, clients, models):
        set_weights(m, w)
        s = error_sums(m, c.test)
        parts.append(s)
        per_client.append({"client": c.cid, **metrics(s)})
    summary = {"run_id": cfg.run_id, **asdict(cfg), "best_round": None, "B_at_best": 0, "B_total": 0,
               "sim_comm_s_total": 0.0, "test": metrics(combine(parts)), "per_client_test": per_client,
               "wall_s": round(time.time() - t0, 1)}
    return rows, summary


def run(prep: Prepared, cfg: RunConfig, log=print):
    return {"fedavg": run_federated, "central": run_central, "local": run_local}[cfg.method](prep, cfg, log)
