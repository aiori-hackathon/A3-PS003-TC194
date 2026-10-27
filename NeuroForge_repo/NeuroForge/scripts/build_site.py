"""Build the static demo site (site/index.html) from the logged results.

The page is self-contained (data embedded as JSON), so it can be hosted on GitHub Pages,
Netlify, or opened from disk. Every number on the page comes from results/.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedtraffic.data import prepare  # noqa: E402
from fedtraffic.partition import partition  # noqa: E402

if __name__ == "__main__":
    with open("results/summary.json") as f:
        S = json.load(f)
    prep = prepare("data/metr-la.h5", "data/graph_sensor_locations.csv")
    parts = {m: [g.tolist() for g in partition(m, prep.coords, prep.train.speed, 4, 11)] for m in ("iid", "noniid", "speed")}
    curves = pd.concat([pd.read_csv(f) for f in glob.glob("results/runs/*.csv")], ignore_index=True)
    curves["group"] = curves.run_id.str.rsplit("_s", n=1).str[0]
    per_seed = {}
    for (g, s), d in curves.groupby(["group", "seed"]):
        d = d.sort_values("round")
        per_seed.setdefault(g, {})[str(s)] = {"round": d["round"].tolist(), "val_mae": d.val_mae.round(4).tolist(),
                                              "B_cum": d.B_cum.tolist(), "B_round": (d.B_up + d.B_down).tolist(),
                                              "participants": d.participants.tolist()}
    data = {
        "summary": {k: S[k] for k in ("dataset", "table", "heterogeneity", "curves", "analytic_round_bytes",
                                      "raw_upload_bytes", "raw_stream_bytes_per_day", "lr", "env", "link_model")},
        "per_seed": per_seed,
        "coords": np.round(prep.coords, 5).tolist(),
        "partitions": parts,
        "sensor_ids": [str(x) for x in prep.sensor_ids],
    }
    # ---- forecast explorer: two test days (Tue 12 - Wed 13 June 2012) + the trained weights
    import base64
    import h5py
    with h5py.File("data/metr-la.h5", "r") as f:
        ts = pd.to_datetime(f["df/axis1"][:])
        vals = f["df/block0_values"][:]
    a = int(np.where(ts == pd.Timestamp("2012-06-12"))[0][0])
    s0, s1 = a - 14, a + 576
    assert s0 >= prep.test.start, "explorer window must lie inside the test period"
    seg = np.round(vals[s0:s1] * 10).astype("<i2")
    models = {}
    for key, rid in (("fedavg", "fedavg_noniid_E1_fp32_s11"), ("central", "central_s11")):
        path = f"results/models/{rid}.json"
        if os.path.exists(path):
            w = json.load(open(path))
            models[key] = {k: np.round(np.array(v), 6).tolist() for k, v in w.items()}
    explorer = {
        "start": str(ts[s0]), "first_shown": 14, "n_steps": int(s1 - s0), "n_sensors": int(vals.shape[1]),
        "speed_x10_b64": base64.b64encode(seg.tobytes()).decode(), "mu": prep.mu, "sd": prep.sd,
        "models": models, "default_sensor": 29,
        "test_mae_s11": {k: json.load(open(f"results/runs/{r}.json"))["test"]["mae"]
                         for k, r in (("fedavg", "fedavg_noniid_E1_fp32_s11"), ("central", "central_s11"))},
    }
    data["explorer"] = explorer
    tpl = open("site/template.html").read()
    os.makedirs("site", exist_ok=True)
    with open("site/index.html", "w") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print("site/index.html", os.path.getsize("site/index.html") // 1024, "kB")
