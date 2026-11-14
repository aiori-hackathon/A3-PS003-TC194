"""IID vs non-IID partitioning (the two settings in our proposal), 3 random seeds each, FedAvg GRU.
Non-IID = 8 road-graph zones (spectral clustering); IID = the same 8 client sizes, sensors assigned at random.
Also logs communication cost per round. Appends `iid` to results.json."""
import json, os
import numpy as np
from fedflow.data import PeMS
from fedflow.models import MODELS
from fedflow import fl as F

HERE = os.path.dirname(os.path.abspath(__file__))
d = PeMS(os.path.join(HERE, "data/pems08.npz"), os.path.join(HERE, "data/distance.csv"))
zones = d.zones(8)
M = MODELS["gru"]()
out = {"noniid": [], "iid": [], "curves": {}}
for seed in (0, 1, 2):
    rng = np.random.default_rng(100 + seed)
    iid = zones[rng.permutation(d.N)]          # same client sizes, random sensors
    for name, part in (("noniid", zones), ("iid", iid)):
        cfg = F.FLConfig(model="gru", rounds=40, seed=seed)
        gp, ms, clients, hist = F.run_federated(d, part, M, cfg, log=None)
        m = F.evaluate(M, gp, d, np.arange(d.N), ms[0], ms[1], d.t_val, d.T)
        out[name].append({"seed": seed, "mae": round(m["avg"]["mae"], 3), "rmse": round(m["avg"]["rmse"], 3)})
        if seed == 0:
            out["curves"][name] = [{"round": h["round"], "val_mae": round(h["val_mae"], 3), "bytes_up": h["bytes_up"]} for h in hist]
        print(name, seed, out[name][-1], flush=True)
for k in ("noniid", "iid"):
    v = np.array([r["mae"] for r in out[k]]); out[k + "_mean"] = round(float(v.mean()), 3); out[k + "_std"] = round(float(v.std()), 3)
print(out["noniid_mean"], out["noniid_std"], out["iid_mean"], out["iid_std"])
p = os.path.join(HERE, "results", "results.json")
res = json.load(open(p)); res["iid"] = out
json.dump(res, open(p, "w"), separators=(",", ":"))
json.dump(res, open(os.path.join(HERE, "..", "web", "public", "data", "results.json"), "w"), separators=(",", ":"))
