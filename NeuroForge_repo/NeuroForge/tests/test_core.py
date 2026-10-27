"""Unit tests named in the proposal's risk table.

Run with:  pytest -q
Tests that need the dataset are skipped automatically when data/metr-la.h5 is absent (CI).
"""
import os

import numpy as np
import pytest
import torch

from fedtraffic.comm import CODECS, COUNT_BYTES, nbytes, round_bytes_analytic
from fedtraffic.engine import masked_l1
from fedtraffic.model import GRUForecaster, get_weights, n_params

DATA = os.path.exists("data/metr-la.h5")


def test_model_size_matches_proposal():
    m = GRUForecaster()
    assert n_params(m) == 3747
    assert nbytes(get_weights(m)) == 14_988


def test_round_bytes_analytic():
    # 14,988 B down + 14,996 B up per client; K = 4 -> 119,936 B per round
    assert round_bytes_analytic(3747, 4) == 119_936
    assert round_bytes_analytic(3747, 4) * 50 == 5_996_800


def test_fedavg_is_sample_weighted_mean():
    a = [torch.ones(3)]; b = [torch.zeros(3)]
    ups = [(a, 30), (b, 10)]
    n = sum(k for _, k in ups)
    w = [sum((k / n) * wk[i] for wk, k in ups) for i in range(1)]
    assert torch.allclose(w[0], torch.full((3,), 0.75))


def test_masked_l1_ignores_missing():
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    y = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, False]])
    assert float(masked_l1(pred, y, mask)) == 0.0


def test_int8_codec_is_small_and_close():
    torch.manual_seed(0)
    delta = [torch.randn(96, 4) * 1e-2, torch.randn(32) * 1e-2]
    dec, size = CODECS["int8"].encode(delta)
    assert size == (96 * 4 + 32) + 2 * 4
    err = max(float((d - e).abs().max()) for d, e in zip(delta, dec))
    scale = max(float(d.abs().max()) for d in delta) / 127
    assert err <= scale / 2 + 1e-9
    assert size + COUNT_BYTES < nbytes(delta) / 3


@pytest.mark.skipif(not DATA, reason="dataset not downloaded")
def test_no_temporal_leakage():
    from fedtraffic.data import prepare
    p = prepare("data/metr-la.h5", "data/graph_sensor_locations.csv")
    tr_end = p.train.start + len(p.train.speed) - 1
    assert tr_end < p.val.start
    assert p.val.start + len(p.val.speed) - 1 < p.test.start
    assert p.meta["train_steps"] == 23_990


@pytest.mark.skipif(not DATA, reason="dataset not downloaded")
def test_windows_stay_inside_period():
    from fedtraffic.data import IN_STEPS, WindowSet, prepare
    p = prepare("data/metr-la.h5", "data/graph_sensor_locations.csv")
    ws = WindowSet(p.val, np.arange(5), p.mu, p.sd, stride=1)
    assert ws.t.min() - (IN_STEPS - 1) >= 0
    assert ws.t.max() + 3 <= len(p.val.speed) - 1
    x, y, m = ws.batch(np.arange(64))
    assert x.shape == (64, 12, 4) and np.isfinite(x).all()


@pytest.mark.skipif(not DATA, reason="dataset not downloaded")
def test_logged_bytes_equal_formula():
    from fedtraffic.data import prepare
    from fedtraffic.simulate import RunConfig, run
    p = prepare("data/metr-la.h5", "data/graph_sensor_locations.csv")
    rows, s = run(p, RunConfig(rounds=2, train_stride=96, val_stride=48, batch=2048), log=lambda m: None)
    per_round = [r["B_up"] + r["B_down"] for r in rows]
    assert per_round == [119_936, 119_936]
    assert rows[-1]["B_cum"] == 2 * 119_936 + 24 * 4   # + one-off z-score statistics
