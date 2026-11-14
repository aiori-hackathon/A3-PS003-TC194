"""Runs every experiment reported in the dashboard and exports assets for the web app.

    python run_experiments.py            # full run (~10 min on a laptop CPU)
    python run_experiments.py --quick    # 8 rounds, for a smoke test

Outputs
    results/results.json        all metrics, curves and per-zone comparisons
    results/*.png               figures used in the report
    ../web/public/data/*.json   network, sensor data, trained weights for the browser demo
"""
from __future__ import annotations
import argparse, base64, json, math, os, time
import numpy as np
from fedflow.data import PeMS, STEPS_PER_DAY
from fedflow.models import MODELS, n_params, HORIZONS
from fedflow.fl import (FLConfig, run_federated, run_centralised, run_local_only, evaluate,
                        historical_average, last_value, gaussian_rdp_epsilon)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
WEB = os.path.join(HERE, "..", "web", "public", "data")


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


def b64_u16(a: np.ndarray) -> str:
    return base64.b64encode(np.clip(np.round(a), 0, 65535).astype("<u2").tobytes()).decode()


def weights_json(p: dict) -> dict:
    return {k: {"shape": list(v.shape), "data": [round(float(x), 6) for x in v.ravel()]} for k, v in p.items()}


def main(quick: bool):
    os.makedirs(OUT, exist_ok=True); os.makedirs(WEB, exist_ok=True)
    d = PeMS(os.path.join(HERE, "data/pems08.npz"), os.path.join(HERE, "data/distance.csv"))
    zones = d.zones(8)
    K = int(zones.max()) + 1
    R = 8 if quick else 40
    allS = np.arange(d.N)
    res = {"dataset": {"name": "PeMS08 (Caltrans PeMS, District 8 – San Bernardino)", "sensors": int(d.N),
                       "days": int(d.T // STEPS_PER_DAY), "interval_min": 5, "train_days": 43, "val_days": 7,
                       "test_days": int(d.T // STEPS_PER_DAY) - 50, "zones": K,
                       "zone_sizes": np.bincount(zones).tolist()},
           "horizons_min": [h * 5 for h in HORIZONS], "runs": {}, "curves": {}, "per_zone": {}, "privacy": [], "dropout": []}
    t_all = time.time()

    def test_metrics(M, p, ms):
        return evaluate(M, p, d, allS, ms[0], ms[1], d.t_val, d.T)

    res["runs"]["Historical average"] = {"model": "-", "metrics": historical_average(d, allS), "private": True}
    res["runs"]["Last value (persistence)"] = {"model": "-", "metrics": last_value(d, allS), "private": True}

    trained = {}
    for m in ("mlp", "gru"):
        M = MODELS[m]()
        base = FLConfig(model=m, rounds=R)
        print(f"\n== {m.upper()} ({n_params(M.init())} parameters)")
        # federated: FedAvg & FedProx
        for strat in ("fedavg", "fedprox"):
            cfg = FLConfig(model=m, strategy=strat, rounds=R)
            gp, ms, clients, hist = run_federated(d, zones, M, cfg)
            key = f"{'FedAvg' if strat == 'fedavg' else 'FedProx'} · {M.name}"
            res["runs"][key] = {"model": M.name, "metrics": test_metrics(M, gp, ms), "private": True,
                                "rounds": R, "bytes_up": hist[-1]["bytes_up"]}
            res["curves"][key] = [{"round": h["round"], "val_mae": round(h["val_mae"], 3), "loss": round(h["loss"], 5)} for h in hist]
            trained[(m, strat)] = (gp, ms, clients, hist)
        # centralised & local-only
        cp, cms = run_centralised(d, zones, M, base)
        res["runs"][f"Centralised · {M.name}"] = {"model": M.name, "metrics": test_metrics(M, cp, cms), "private": False}
        local = run_local_only(d, zones, M, base)
        # local-only overall metric: each zone's model on its own sensors, pooled
        from fedflow.fl import metrics as _metrics
        P, Tt = [], []
        for z, (lp, lms) in local.items():
            sens = np.where(zones == z)[0]
            s, t = d.windows(sens, d.t_val, d.T, 6)
            lags, y, tod, we = d.batch(s, t, *lms)
            yh, _ = M.forward(lp, M.featurize(lags, tod, we))
            P.append(yh * lms[1] + lms[0]); Tt.append(y * lms[1] + lms[0])
        res["runs"][f"Local-only · {M.name}"] = {"model": M.name, "metrics": _metrics(np.concatenate(P), np.concatenate(Tt)), "private": True}
        trained[(m, "local")] = local
        trained[(m, "central")] = (cp, cms)
        # per-zone: local-only vs federated
        gp, ms, clients, hist = trained[(m, "fedavg")]
        rows = []
        for z in range(K):
            sens = np.where(zones == z)[0]
            fl = evaluate(M, gp, d, sens, ms[0], ms[1], d.t_val, d.T)["avg"]["mae"]
            lp, lms = local[z]
            lo = evaluate(M, lp, d, sens, lms[0], lms[1], d.t_val, d.T)["avg"]["mae"]
            rows.append({"zone": z, "sensors": int(len(sens)), "samples": int(clients[z].n),
                         "local_mae": round(lo, 3), "fed_mae": round(fl, 3), "gain_pct": round((lo - fl) / lo * 100, 2)})
        res["per_zone"][M.name] = rows
        print("  per-zone gains:", [r["gain_pct"] for r in rows])

    # ---------------------------------------------------------------- privacy sweep (GRU, FedAvg)
    M = MODELS["gru"]()
    norms = [v for h in trained[("gru", "fedavg")][3][5:] for v in h["update_norm"].values()]
    clip = float(np.median(norms))
    res["privacy_clip"] = round(clip, 4)
    for sigma in ([0.5, 1.0] if quick else [0.3, 0.5, 1.0, 2.0]):
        cfg = FLConfig(model="gru", rounds=R, clip=clip, noise=sigma)
        gp, ms, _, hist = run_federated(d, zones, M, cfg)
        res["privacy"].append({"sigma": sigma, "epsilon": round(gaussian_rdp_epsilon(sigma, R), 2), "delta": 1e-5,
                               "mae": round(test_metrics(M, gp, ms)["avg"]["mae"], 3)})
    cfg = FLConfig(model="gru", rounds=R, clip=clip, noise=0.0)
    gp, ms, _, _ = run_federated(d, zones, M, cfg)
    res["privacy"].insert(0, {"sigma": 0, "epsilon": None, "delta": None, "mae": round(test_metrics(M, gp, ms)["avg"]["mae"], 3),
                              "note": "clipping only"})
    # ---------------------------------------------------------------- dropout robustness
    for p_drop in ([0.3] if quick else [0.2, 0.4, 0.6]):
        cfg = FLConfig(model="gru", rounds=R, dropout=p_drop)
        gp, ms, _, hist = run_federated(d, zones, M, cfg)
        res["dropout"].append({"dropout": p_drop, "mae": round(test_metrics(M, gp, ms)["avg"]["mae"], 3),
                               "avg_clients": round(float(np.mean([len(h["clients"]) for h in hist])), 2)})
    res["dropout"].insert(0, {"dropout": 0.0, "mae": round(res["runs"]["FedAvg · GRU"]["metrics"]["avg"]["mae"], 3), "avg_clients": 8})

    # ---------------------------------------------------------------- communication
    gp, ms, clients, hist = trained[("gru", "fedavg")]
    raw = sum(c.raw_bytes() for c in clients)
    res["communication"] = {"params_gru": n_params(gp), "params_mlp": n_params(trained[("mlp", "fedavg")][0]),
                            "model_bytes": n_params(gp) * 4, "total_upload_bytes": hist[-1]["bytes_up"],
                            "raw_data_bytes": raw, "rounds": R}
    res["runtime_sec"] = round(time.time() - t_all, 1)
    jdump(res, os.path.join(OUT, "results.json"))
    jdump(res, os.path.join(WEB, "results.json"))
    print(json.dumps({k: v["metrics"]["avg"] for k, v in res["runs"].items()}, indent=1))
    print("privacy", res["privacy"]); print("dropout", res["dropout"]); print("runtime", res["runtime_sec"])

    # ---------------------------------------------------------------- web assets
    X = d.layout()
    zstats = []
    for z in range(K):
        sens = np.where(zones == z)[0]
        zstats.append({"zone": z, "sensors": sens.tolist(), "cx": float(X[sens, 0].mean()), "cy": float(X[sens, 1].mean())})
    edges = [[int(a), int(b)] for a, b in d.dist[["from", "to"]].itertuples(index=False)]
    p95 = np.percentile(d.flow[: d.t_train], 95, axis=0)
    jdump({"sensors": [{"id": i, "x": round(float(X[i, 0]), 4), "y": round(float(X[i, 1]), 4), "zone": int(zones[i]),
                        "p95": round(float(p95[i]), 1)} for i in range(d.N)],
           "zones": zstats, "edges": edges, "start": "2016-07-01", "test_start_step": int(d.t_val)}, os.path.join(WEB, "network.json"))
    # test-period flow for all sensors (+1 h of context before)
    ctx = 12
    test = d.flow[d.t_val - ctx:, :]
    jdump({"t0": int(d.t_val - ctx), "steps": int(test.shape[0]), "sensors": int(d.N), "flow_u16": b64_u16(test.T)},
          os.path.join(WEB, "test_flow.json"))
    # training subset for the in-browser federated demo: up to 6 sensors per zone, last 14 training days
    pick = []
    for z in range(K):
        sens = np.where(zones == z)[0]
        pick += sorted(sens[np.argsort(-d.flow[: d.t_train, sens].mean(0))][:6].tolist())
    t0 = d.t_train - 14 * STEPS_PER_DAY
    jdump({"t0": int(t0), "steps": int(14 * STEPS_PER_DAY), "sensors": pick, "zones": [int(zones[s]) for s in pick],
           "flow_u16": b64_u16(d.flow[t0: d.t_train, pick].T)}, os.path.join(WEB, "train_subset.json"))
    # trained weights
    gp, ms, _, _ = trained[("gru", "fedavg")]
    lp = {str(z): {"norm": list(map(float, v[1])), "weights": weights_json(v[0])} for z, v in trained[("gru", "local")].items()}
    cp, cms = trained[("gru", "central")]
    jdump({"model": "GRU", "hidden": 32, "lags": 12, "horizons": list(HORIZONS),
           "federated": {"norm": [float(ms[0]), float(ms[1])], "weights": weights_json(gp)},
           "local": lp}, os.path.join(WEB, "gru_weights.json"))
    print("web assets written:", sorted(os.listdir(WEB)))

    # ---------------------------------------------------------------- figures for the report
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for key, col in (("FedAvg · GRU", "#4f46e5"), ("FedProx · GRU", "#0ea5e9"), ("FedAvg · TemporalMLP", "#f59e0b")):
        c = res["curves"][key]; ax.plot([r["round"] for r in c], [r["val_mae"] for r in c], label=key, color=col, lw=2)
    ax.axhline(res["runs"]["Centralised · GRU"]["metrics"]["avg"]["mae"], ls="--", color="#64748b", label="Centralised GRU (no privacy)")
    ax.axhline(res["runs"]["Historical average"]["metrics"]["avg"]["mae"], ls=":", color="#dc2626", label="Historical average")
    ax.set_xlabel("Federated round"); ax.set_ylabel("Validation MAE (veh / 5 min)"); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "convergence.png"), dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    rows = res["per_zone"]["GRU"]; xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["local_mae"] for r in rows], 0.4, label="Local-only", color="#cbd5e1")
    ax.bar(xs + 0.2, [r["fed_mae"] for r in rows], 0.4, label="Federated (FedAvg)", color="#4f46e5")
    ax.set_xticks(xs, [f"Z{r['zone'] + 1}\n({r['sensors']})" for r in rows]); ax.set_ylabel("Test MAE"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "per_zone.png"), dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    pv = [p for p in res["privacy"] if p["epsilon"]]
    ax.plot([p["epsilon"] for p in pv], [p["mae"] for p in pv], "o-", color="#be123c")
    for p in pv: ax.annotate(f"σ={p['sigma']}", (p["epsilon"], p["mae"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(res["runs"]["FedAvg · GRU"]["metrics"]["avg"]["mae"], ls="--", color="#64748b", label="No DP")
    ax.set_xscale("log"); ax.set_xlabel("Privacy budget ε (δ = 1e-5), lower = more private"); ax.set_ylabel("Test MAE"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "privacy.png"), dpi=180); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
