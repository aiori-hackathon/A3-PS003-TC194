"""Experiment manager (module G): runs the proposal's matrix across seeds 11, 22, 33.

    python scripts/run_experiments.py --suite core        # E1-E3 (mandatory)
    python scripts/run_experiments.py --suite all         # + E4, E5 and the extensions
    python scripts/run_experiments.py --suite smoke       # 3-round demo for the mentor

Runs already present in results/runs/ are skipped, so the script can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedtraffic.data import download, prepare  # noqa: E402
from fedtraffic.simulate import RunConfig, run  # noqa: E402

SEEDS = (11, 22, 33)
OUT = "results/runs"


def load_lr():
    path = "results/lr_selection.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"central": 3e-3, "fedavg": 3e-3}


def suite(name: str, base: RunConfig) -> list[RunConfig]:
    lr = load_lr()
    c = lambda **kw: replace(base, **kw)  # noqa: E731
    out = []
    if name == "smoke":
        return [c(seed=11, method="fedavg", partition="iid", rounds=3, lr=lr["fedavg"])]
    for s in SEEDS:
        out += [c(seed=s, method="central", lr=lr["central"]),                              # E1
                c(seed=s, method="fedavg", partition="iid", lr=lr["fedavg"]),               # E2
                c(seed=s, method="fedavg", partition="noniid", lr=lr["fedavg"])]            # E3
    if name == "all":
        for s in SEEDS:
            out += [c(seed=s, method="fedavg", partition="noniid", codec="int8", lr=lr["fedavg"]),   # X1 compression
                    c(seed=s, method="fedavg", partition="speed", lr=lr["fedavg"]),                  # X3 strong non-IID
                    c(seed=s, method="fedavg", partition="noniid", dropout=0.25, lr=lr["fedavg"]),   # X2 unreliable links
                    c(seed=s, method="local", partition="noniid", lr=lr["fedavg"]),                  # E4 zero-communication floor
                    c(seed=s, method="fedavg", partition="noniid", local_epochs=5, rounds=base.rounds // 5, lr=lr["fedavg"])]  # E5
    if name == "extra":   # optional, not run for the submitted report
        for s in SEEDS:
            out += [c(seed=s, method="local", partition="iid", lr=lr["fedavg"]),
                    c(seed=s, method="fedavg", partition="iid", local_epochs=5, rounds=base.rounds // 5, lr=lr["fedavg"]),
                    c(seed=s, method="fedavg", partition="noniid", codec="fp16", lr=lr["fedavg"])]
    return out


def _worker(cfg: RunConfig, data_dir: str):
    import torch
    torch.set_num_threads(1)
    prep = prepare(os.path.join(data_dir, "metr-la.h5"), os.path.join(data_dir, "graph_sensor_locations.csv"))
    logf = open(os.path.join(OUT, cfg.run_id + ".log"), "w")
    rows, summary = run(prep, cfg, log=lambda m: (logf.write(m + "\n"), logf.flush()))
    df = pd.DataFrame(rows)
    for k in ("seed", "method", "partition", "local_epochs", "codec", "dropout"):
        df.insert(0, k, getattr(cfg, k))
    df.insert(0, "run_id", cfg.run_id)
    df.to_csv(os.path.join(OUT, cfg.run_id + ".csv"), index=False)
    with open(os.path.join(OUT, cfg.run_id + ".json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return cfg.run_id, summary["test"]["mae"], summary["wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="core", choices=["smoke", "core", "all", "extra"])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--data", default="data")
    ap.add_argument("--train-stride", type=int, default=24)
    ap.add_argument("--val-stride", type=int, default=6)
    args = ap.parse_args()
    download(args.data)
    os.makedirs(OUT, exist_ok=True)
    base = RunConfig(rounds=args.rounds, train_stride=args.train_stride, val_stride=args.val_stride, batch=1024)
    cfgs = suite(args.suite, base)
    todo = [c for c in cfgs if not os.path.exists(os.path.join(OUT, c.run_id + ".json"))]
    if args.suite == "smoke":
        todo = suite("smoke", base)
    print(f"{len(todo)} runs to do")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, c, args.data): c for c in todo}
        for f in as_completed(futs):
            rid, mae, wall = f.result()
            print(f"done {rid}: test MAE {mae:.3f} mph  ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
