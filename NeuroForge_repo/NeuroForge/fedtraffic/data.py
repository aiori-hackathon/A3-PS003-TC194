"""Module A + B: load, validate, split and window the METR-LA dataset.

Design rules (proposal Section 4.4):
* chronological 70/10/20 split of *time*, never random rows;
* z-score statistics fitted on the training period only;
* windows are built inside each period, so no input or target crosses a boundary;
* a reading of 0 mph in METR-LA means "missing" and is masked everywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd

# Public mirrors of the DCRNN files (Li et al., ICLR 2018). The h5 file is the one linked
# from github.com/liyaguang/DCRNN; the mirror is used because Google Drive is not scriptable.
H5_URLS = [
    "https://raw.githubusercontent.com/deepkashiwa20/DL-Traff-Graph/main/METRLA/metr-la.h5",
    "https://raw.githubusercontent.com/deepkashiwa20/MegaCRN/main/METRLA/metr-la.h5",
]
LOC_URL = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv"
H5_SHA256 = "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"  # SHA-256 of the file we used

IN_STEPS = 12          # 60 minutes of history
HORIZONS = (1, 2, 3)   # +5, +10, +15 minutes
STEPS_PER_DAY = 288


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir: str = "data") -> tuple[str, str]:
    """Download metr-la.h5 and the sensor-location CSV if not already cached."""
    os.makedirs(data_dir, exist_ok=True)
    h5 = os.path.join(data_dir, "metr-la.h5")
    loc = os.path.join(data_dir, "graph_sensor_locations.csv")
    if not os.path.exists(h5):
        for url in H5_URLS:
            try:
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, h5)
                break
            except Exception as e:  # pragma: no cover - network dependent
                print(f"  failed: {e}")
    if not os.path.exists(loc):
        urllib.request.urlretrieve(LOC_URL, loc)
    return h5, loc


def load_metr_la(h5_path: str, loc_path: str):
    """Return speeds (T, N) float32, timestamps (T,), sensor ids (N,), coords (N, 2)."""
    with h5py.File(h5_path, "r") as f:
        values = f["df/block0_values"][:].astype(np.float32)
        ids = [s.decode() for s in f["df/block0_items"][:]]
        ts = pd.to_datetime(f["df/axis1"][:])
    loc = pd.read_csv(loc_path)
    loc["sensor_id"] = loc["sensor_id"].astype(str)
    loc = loc.set_index("sensor_id").loc[ids]
    coords = loc[["latitude", "longitude"]].to_numpy(np.float64)
    # validation (proposal risk table: assert shape and timestamps)
    assert values.shape == (34272, 207), values.shape
    assert ts.is_monotonic_increasing, "timestamps not sorted"
    step = np.diff(ts.values).astype("timedelta64[m]").astype(int)
    assert (step == 5).all(), "irregular 5-minute grid"
    return values, ts, np.array(ids), coords


@dataclass
class Split:
    """One chronological period of the series."""
    name: str
    speed: np.ndarray        # (T_p, N) raw mph, 0 = missing
    tod: np.ndarray          # (T_p,) time-of-day in [0, 1)
    start: int               # index of first step in the full series


@dataclass
class Prepared:
    train: Split
    val: Split
    test: Split
    mu: float
    sd: float
    sensor_ids: np.ndarray
    coords: np.ndarray
    meta: dict = field(default_factory=dict)


def chrono_split(values, ts, ratios=(0.7, 0.1, 0.2)):
    T = len(values)
    n_tr = int(round(T * ratios[0]))
    n_va = int(round(T * ratios[1]))
    bounds = [(0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, T)]
    tod = ((ts.hour * 60 + ts.minute) / (24 * 60)).to_numpy(np.float32)
    out = []
    for name, (a, b) in zip(("train", "val", "test"), bounds):
        out.append(Split(name, values[a:b], tod[a:b], a))
    return out


def fit_zscore(train: Split):
    """Mean / std over observed (non-zero) training readings only."""
    obs = train.speed[train.speed > 0]
    return float(obs.mean()), float(obs.std())


def prepare(h5_path: str, loc_path: str) -> Prepared:
    values, ts, ids, coords = load_metr_la(h5_path, loc_path)
    tr, va, te = chrono_split(values, ts)
    mu, sd = fit_zscore(tr)
    meta = {
        "shape": list(values.shape),
        "missing_rate": float((values == 0).mean()),
        "first": str(ts[0]), "last": str(ts[-1]),
        "train_steps": len(tr.speed), "val_steps": len(va.speed), "test_steps": len(te.speed),
        "train_end": str(ts[tr.start + len(tr.speed) - 1]), "val_end": str(ts[va.start + len(va.speed) - 1]),
        "mu": mu, "sd": sd, "sha256": sha256(h5_path),
    }
    return Prepared(tr, va, te, mu, sd, ids, coords, meta)


class WindowSet:
    """Lazy sliding windows for a subset of sensors inside one period.

    A sample is (sensor j, anchor t): input steps t-11..t, targets t+1..t+3.
    Windows are materialised per batch, so memory stays small even for 5M samples.
    """

    def __init__(self, split: Split, sensors: np.ndarray, mu: float, sd: float, stride: int = 1):
        self.speed = split.speed[:, sensors]            # (T_p, n)
        self.tod = split.tod
        self.mu, self.sd = mu, sd
        T = self.speed.shape[0]
        anchors = np.arange(IN_STEPS - 1, T - max(HORIZONS), stride)
        jj, tt = np.meshgrid(np.arange(self.speed.shape[1]), anchors, indexing="ij")
        self.j = jj.ravel().astype(np.int64)
        self.t = tt.ravel().astype(np.int64)
        # drop samples whose three targets are all missing (they carry no loss signal)
        tgt = self._targets(self.j, self.t)
        keep = (tgt > 0).any(axis=1)
        self.j, self.t = self.j[keep], self.t[keep]
        self.n_sensors = self.speed.shape[1]

    def __len__(self):
        return len(self.j)

    def _targets(self, j, t):
        return np.stack([self.speed[t + h, j] for h in HORIZONS], axis=1)

    def batch(self, idx):
        """Return x (B, 12, 4), y_mph (B, 3), y_mask (B, 3) as numpy arrays."""
        j, t = self.j[idx], self.t[idx]
        steps = t[:, None] + np.arange(-IN_STEPS + 1, 1)[None, :]       # (B, 12)
        raw = self.speed[steps, j[:, None]]                               # (B, 12)
        obs = raw > 0
        # in-window forward fill of missing inputs; leading gaps -> train mean (z = 0)
        pos = np.where(obs, np.arange(IN_STEPS)[None, :], -1)
        pos = np.maximum.accumulate(pos, axis=1)
        filled = np.take_along_axis(raw, np.clip(pos, 0, None), axis=1)
        z = np.where(pos >= 0, (filled - self.mu) / self.sd, 0.0)
        ang = 2 * np.pi * self.tod[steps]
        x = np.stack([z, (~obs).astype(np.float32), np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)
        y = self._targets(j, t)
        return x, y.astype(np.float32), (y > 0)


def describe(prep: Prepared) -> str:
    return json.dumps(prep.meta, indent=2)
