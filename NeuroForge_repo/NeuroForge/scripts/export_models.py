"""Re-train the seed-11 centralised and FedAvg non-IID runs and save their selected weights.

The weights power the in-browser forecast explorer in site/index.html. The runs use the same
configuration as the logged experiments, so the test MAE must match results/runs/*.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedtraffic.data import prepare  # noqa: E402
from fedtraffic.simulate import RunConfig, run  # noqa: E402


def job(kw):
    import torch
    torch.set_num_threads(1)
    prep = prepare("data/metr-la.h5", "data/graph_sensor_locations.csv")
    lr = json.load(open("results/lr_selection.json"))
    cfg = RunConfig(rounds=50, train_stride=24, val_stride=6, batch=1024, save_weights=True,
                    lr=lr["central" if kw["method"] == "central" else "fedavg"], **kw)
    _, s = run(prep, cfg, log=lambda m: None)
    logged = json.load(open(f"results/runs/{cfg.run_id}.json"))["test"]["mae"]
    return cfg.run_id, s["test"]["mae"], logged


if __name__ == "__main__":
    jobs = [dict(seed=11, method="central"), dict(seed=11, method="fedavg", partition="noniid")]
    with ProcessPoolExecutor(2) as ex:
        for rid, mae, logged in ex.map(job, jobs):
            print(f"{rid}: test MAE {mae:.4f} (logged {logged:.4f}) {'MATCH' if abs(mae - logged) < 1e-3 else 'DIFFERS'}")
