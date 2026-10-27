"""Module C: assign sensors to K simulated intersection clients.

* iid      - sensors shuffled with the run seed into K near-equal groups (52/52/52/51).
* noniid   - k-means (k = K, seeded) on sensor latitude/longitude: spatially coherent,
             unequal-size clients, so the n_k weights in FedAvg differ.
* speed    - documented fallback / stress test: sensors ranked by training-period mean
             speed and cut into K quantile blocks (strong level skew).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def partition(mode: str, coords: np.ndarray, train_speed: np.ndarray, K: int, seed: int) -> list[np.ndarray]:
    N = coords.shape[0]
    rng = np.random.default_rng(seed)
    if mode == "iid":
        perm = rng.permutation(N)
        groups = np.array_split(perm, K)
    elif mode == "noniid":
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(coords)
        groups = [np.where(km.labels_ == k)[0] for k in range(K)]
        groups.sort(key=lambda g: -len(g))
    elif mode == "speed":
        mean_speed = np.array([s[s > 0].mean() for s in train_speed.T])
        order = np.argsort(mean_speed)
        groups = np.array_split(order, K)
    else:
        raise ValueError(mode)
    return [np.sort(g) for g in groups]


def client_stats(groups, train_speed: np.ndarray) -> list[dict]:
    """Per-client heterogeneity report (proposal validity check)."""
    out = []
    for k, g in enumerate(groups):
        s = train_speed[:, g]
        obs = s[s > 0]
        out.append({
            "client": k + 1, "sensors": int(len(g)),
            "mean_speed": round(float(obs.mean()), 2), "std_speed": round(float(obs.std()), 2),
            "missing_rate": round(float((s == 0).mean()), 4),
            "share_congested": round(float((obs < 40).mean()), 4),
        })
    return out


def heterogeneity_index(stats: list[dict]) -> float:
    """Spread of client mean speeds (mph): 0 means identical clients."""
    m = np.array([s["mean_speed"] for s in stats])
    return float(m.max() - m.min())
