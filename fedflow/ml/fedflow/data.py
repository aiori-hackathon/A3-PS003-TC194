"""PeMS08 loading, zone partitioning (one zone = one federated client) and sample windows.

PeMS08: Caltrans Performance Measurement System, San Bernardino district, 170 loop-detector
sensors, 1 Jul – 31 Aug 2016, 5-minute aggregates of flow / occupancy / speed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from .models import LAGS, HORIZONS

STEPS_PER_DAY = 288
FIRST_WEEKDAY = 4          # 1 July 2016 was a Friday (Mon = 0)
TRAIN_DAYS, VAL_DAYS = 43, 7   # remaining 12 days are the test period


class PeMS:
    def __init__(self, npz_path: str, dist_path: str):
        raw = np.load(npz_path)["data"]
        self.flow = raw[..., 0].astype(np.float64)        # (T, N)
        self.speed = raw[..., 2].astype(np.float64)
        self.T, self.N = self.flow.shape
        self.dist = pd.read_csv(dist_path)
        self.t_train = TRAIN_DAYS * STEPS_PER_DAY
        self.t_val = (TRAIN_DAYS + VAL_DAYS) * STEPS_PER_DAY

    # ---------------------------------------------------------------- graph and zones
    def adjacency(self) -> np.ndarray:
        A = np.zeros((self.N, self.N))
        sigma = self.dist["cost"].std()
        for f, t, c in self.dist[["from", "to", "cost"]].itertuples(index=False):
            w = np.exp(-(c / sigma) ** 2)
            A[int(f), int(t)] = A[int(t), int(f)] = max(w, 1e-3)
        return A

    def zones(self, k: int = 8, seed: int = 7) -> np.ndarray:
        """Spectral clustering of the road graph; sensors with no edge are assigned to the zone
        whose average training-period profile they correlate with best."""
        from sklearn.cluster import SpectralClustering
        A = self.adjacency()
        connected = A.sum(1) > 0
        sub = A[np.ix_(connected, connected)] + 1e-6
        lab = SpectralClustering(k, affinity="precomputed", random_state=seed, assign_labels="discretize").fit_predict(sub)
        labels = -np.ones(self.N, dtype=int)
        labels[connected] = lab
        prof = self.flow[: self.t_train]
        centers = np.stack([prof[:, labels == z].mean(1) for z in range(k)], 1)
        for i in np.where(~connected)[0]:
            c = [np.corrcoef(prof[:, i], centers[:, z])[0, 1] for z in range(k)]
            labels[i] = int(np.nanargmax(c))
        # relabel zones by size (largest = 0) for stable display
        order = np.argsort(-np.bincount(labels, minlength=k))
        remap = {old: new for new, old in enumerate(order)}
        return np.array([remap[l] for l in labels])

    def layout(self, seed: int = 3) -> np.ndarray:
        """2-D positions from classical MDS on road-network shortest-path distances (no GPS in PeMS08)."""
        rows, cols, vals = [], [], []
        for f, t, c in self.dist[["from", "to", "cost"]].itertuples(index=False):
            rows += [int(f), int(t)]; cols += [int(t), int(f)]; vals += [c, c]
        G = csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        D = shortest_path(G, directed=False)
        finite = D[np.isfinite(D)]
        D[~np.isfinite(D)] = finite.max() * 1.2
        n = self.N
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        w, v = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1][:2]
        X = v[:, idx] * np.sqrt(np.maximum(w[idx], 1e-9))
        X += np.random.default_rng(seed).normal(scale=finite.max() * 0.01, size=X.shape)
        X -= X.min(0); X /= X.max(0)
        return X

    # ---------------------------------------------------------------- time features
    @staticmethod
    def time_feats(t_idx: np.ndarray):
        """t_idx = index of the LAST lag step."""
        tod = (t_idx % STEPS_PER_DAY) / STEPS_PER_DAY
        dow = (FIRST_WEEKDAY + t_idx // STEPS_PER_DAY) % 7
        weekend = (dow >= 5).astype(np.float64)
        return tod, weekend

    def windows(self, sensors: np.ndarray, t0: int, t1: int, stride: int = 1):
        """All (sensor, t) windows whose lags start at >= t0 and whose farthest target is < t1.
        Returns sensor ids, t (first target index) arrays."""
        H = max(HORIZONS)
        ts = np.arange(t0 + LAGS, t1 - H + 1, stride)
        S, Tt = np.meshgrid(sensors, ts, indexing="ij")
        return S.ravel(), Tt.ravel()

    def batch(self, s: np.ndarray, t: np.ndarray, mu: float, sd: float):
        """lags (B,12), targets (B,4) normalised; tod & weekend of the last lag."""
        lag_idx = t[:, None] + np.arange(-LAGS, 0)[None, :]
        lags = (self.flow[lag_idx, s[:, None]] - mu) / sd
        tgt_idx = t[:, None] + (np.array(HORIZONS) - 1)[None, :]
        y = (self.flow[tgt_idx, s[:, None]] - mu) / sd
        tod, we = self.time_feats(t - 1)
        return lags, y, tod, we
