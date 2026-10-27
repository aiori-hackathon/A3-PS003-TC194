"""Mentor demo: a 3-round FedAvg smoke test on seed 11 that checks bytes against the formula.

    python scripts/smoke.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedtraffic.comm import round_bytes_analytic  # noqa: E402
from fedtraffic.data import download, prepare  # noqa: E402
from fedtraffic.simulate import RunConfig, run  # noqa: E402

if __name__ == "__main__":
    h5, loc = download("data")
    prep = prepare(h5, loc)
    cfg = RunConfig(seed=11, partition="noniid", rounds=3, lr=1e-2, train_stride=48, val_stride=12, batch=1024)
    t0 = time.time()
    rows, s = run(prep, cfg)
    expected = round_bytes_analytic(3747, 4)
    print(f"\n{'round':>5} {'B_up':>8} {'B_down':>8} {'B_round':>8} {'expected':>8}")
    for r in rows:
        b = r["B_up"] + r["B_down"]
        print(f"{r['round']:>5} {r['B_up']:>8} {r['B_down']:>8} {b:>8} {expected:>8}  {'OK' if b == expected else 'MISMATCH'}")
    print(f"\nclients: {[c['sensors'] for c in s['client_stats']]} sensors; test MAE {s['test']['mae']:.3f} mph; {time.time()-t0:.0f}s")
    assert all(r["B_up"] + r["B_down"] == expected for r in rows)
