"""Learning-rate selection on validation (proposal 5.1: <=3-value grid, identical for all methods).

To keep CPU cost low, each candidate is trained for 8 rounds/epochs on seed 11 and the
value with the lowest best-validation MAE is kept for every seed. The test period is never used.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedtraffic.data import prepare  # noqa: E402
from fedtraffic.simulate import RunConfig, run  # noqa: E402

GRID = (1e-3, 3e-3, 1e-2)


def trial(args):
    import torch
    torch.set_num_threads(1)
    method, lr = args
    prep = prepare("data/metr-la.h5", "data/graph_sensor_locations.csv")
    cfg = RunConfig(seed=11, method=method, partition="iid", rounds=8, lr=lr, batch=1024, train_stride=24, val_stride=6)
    rows, _ = run(prep, cfg, log=lambda m: None)
    return method, lr, min(r["val_mae"] for r in rows)


if __name__ == "__main__":
    jobs = [(m, lr) for m in ("central", "fedavg") for lr in GRID]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(trial, jobs))
    table = [{"method": m, "lr": lr, "best_val_mae_8": round(v, 4)} for m, lr, v in res]
    choice = {m: min((r for r in table if r["method"] == m), key=lambda r: r["best_val_mae_8"])["lr"]
              for m in ("central", "fedavg")}
    os.makedirs("results", exist_ok=True)
    with open("results/lr_selection.json", "w") as f:
        json.dump(choice, f, indent=2)
    with open("results/lr_grid.json", "w") as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2), "\nselected:", choice)
