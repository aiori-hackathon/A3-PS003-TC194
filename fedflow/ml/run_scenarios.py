"""Two deployment scenarios where collaboration matters most (appended to results.json):

1. New deployment / scarce history: every zone has only the last D days of data.
   Local-only models vs one federated model trained on the same scarce data.
2. Unseen zone (cold start): train the federated model on 7 zones, test on the 8th zone,
   which never shared anything. Compared with that zone's own local model (trained on 7 days).
"""
from __future__ import annotations
import json, os
import numpy as np
from fedflow.data import PeMS, STEPS_PER_DAY
from fedflow.models import MODELS
from fedflow import fl as F

HERE = os.path.dirname(os.path.abspath(__file__))


class ScarceClient(F.Client):
    days = 7

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        t0 = self.data.t_train - self.days * STEPS_PER_DAY
        self.s, self.t = self.data.windows(self.sensors, t0, self.data.t_train)
        self.n = len(self.s)


def main():
    d = PeMS(os.path.join(HERE, "data/pems08.npz"), os.path.join(HERE, "data/distance.csv"))
    zones = d.zones(8)
    K = int(zones.max()) + 1
    M = MODELS["gru"]()
    path = os.path.join(HERE, "results", "results.json")
    res = json.load(open(path))
    F.Client = ScarceClient                     # every client below sees only its scarce window
    out = []
    for days in (3, 7):
        ScarceClient.days = days
        cfg = F.FLConfig(model="gru", rounds=30)
        gp, ms, clients, _ = F.run_federated(d, zones, M, cfg, log=None)
        rows = []
        for z in range(K):
            sens = np.where(zones == z)[0]
            fed = F.evaluate(M, gp, d, sens, ms[0], ms[1], d.t_val, d.T)["avg"]["mae"]
            x = d.flow[d.t_train - days * STEPS_PER_DAY: d.t_train, sens]
            lms = (x.mean(), x.std())
            c = ScarceClient(z, sens, d, M, lms[0], lms[1], z)
            lp, _ = c.train(M.init(0), cfg, steps=cfg.rounds * cfg.local_steps)
            loc = F.evaluate(M, lp, d, sens, lms[0], lms[1], d.t_val, d.T)["avg"]["mae"]
            rows.append({"zone": z, "local_mae": round(loc, 3), "fed_mae": round(fed, 3), "gain_pct": round((loc - fed) / loc * 100, 2)})
        w = np.bincount(zones)
        avg_loc = float(np.average([r["local_mae"] for r in rows], weights=w))
        avg_fed = float(np.average([r["fed_mae"] for r in rows], weights=w))
        out.append({"days": days, "local_mae": round(avg_loc, 3), "fed_mae": round(avg_fed, 3),
                    "gain_pct": round((avg_loc - avg_fed) / avg_loc * 100, 2), "zones": rows})
        print(f"scarce {days}d: local {avg_loc:.2f}  fed {avg_fed:.2f}")
    res["scarce"] = out

    # unseen zone: leave each of the 4 largest zones out in turn
    ScarceClient.days = 43
    unseen = []
    for hold in range(4):
        mask = zones != hold
        remap = {old: i for i, old in enumerate(sorted(set(zones[mask])))}
        zl = np.array([remap.get(z, 99) for z in zones])      # held-out zone gets label 99 = not a client
        cfg = F.FLConfig(model="gru", rounds=30)
        gp, ms, _, _ = run_partial(d, zl, M, cfg)
        sens = np.where(zones == hold)[0]
        fed = F.evaluate(M, gp, d, sens, ms[0], ms[1], d.t_val, d.T)["avg"]["mae"]
        x = d.flow[d.t_train - 7 * STEPS_PER_DAY: d.t_train, sens]
        lms = (x.mean(), x.std())
        ScarceClient.days = 7
        c = ScarceClient(hold, sens, d, M, lms[0], lms[1], hold)
        lp, _ = c.train(M.init(0), cfg, steps=cfg.rounds * cfg.local_steps)
        ScarceClient.days = 43
        loc = F.evaluate(M, lp, d, sens, lms[0], lms[1], d.t_val, d.T)["avg"]["mae"]
        unseen.append({"zone": hold, "sensors": int(len(sens)), "fed_unseen_mae": round(fed, 3), "local_7day_mae": round(loc, 3)})
        print("unseen", unseen[-1])
    res["unseen_zone"] = unseen

    # DP at scale: the same network split into ~34 junction-level clients (≈5 sensors each)
    F.Client = F.Client.__mro__[1] if F.Client is ScarceClient else F.Client
    labels = np.zeros(d.N, dtype=int); nxt = 0
    for z in range(K):
        sens = np.where(zones == z)[0]
        for chunk in np.array_split(sens, max(1, round(len(sens) / 5))):
            labels[chunk] = nxt; nxt += 1
    base_cfg = F.FLConfig(model="gru", rounds=30)
    gp, ms, _, hist = F.run_federated(d, labels, M, base_cfg, log=None)
    no_dp = F.evaluate(M, gp, d, np.arange(d.N), ms[0], ms[1], d.t_val, d.T)["avg"]["mae"]
    clip = float(np.median([v for h in hist[5:] for v in h["update_norm"].values()]))
    scale = [{"clients": int(nxt), "sigma": 0, "epsilon": None, "mae": round(no_dp, 3)}]
    for sigma in (0.5, 1.0, 2.0):
        cfg = F.FLConfig(model="gru", rounds=30, clip=clip, noise=sigma)
        gp, ms, _, _ = F.run_federated(d, labels, M, cfg, log=None)
        mae = F.evaluate(M, gp, d, np.arange(d.N), ms[0], ms[1], d.t_val, d.T)["avg"]["mae"]
        scale.append({"clients": int(nxt), "sigma": sigma, "epsilon": round(F.gaussian_rdp_epsilon(sigma, 30), 2), "mae": round(mae, 3)})
        print("dp-scale", scale[-1])
    res["privacy_scale"] = scale
    json.dump(res, open(path, "w"), separators=(",", ":"))
    json.dump(res, open(os.path.join(HERE, "..", "web", "public", "data", "results.json"), "w"), separators=(",", ":"))


def run_partial(d, zl, M, cfg):
    """run_federated over the zones whose label != 99"""
    labels = zl.copy()
    keep = labels != 99
    k = int(labels[keep].max()) + 1
    clients0 = [F.Client(z, np.where(labels == z)[0], d, M, 0, 1, z) for z in range(k)]
    mu, sd = F.federated_normalisation(clients0, d)
    clients = [F.Client(z, np.where(labels == z)[0], d, M, mu, sd, z) for z in range(k)]
    gp = M.init(cfg.seed)
    total = sum(c.n for c in clients)
    for r in range(cfg.rounds):
        gvec = F.flatten(gp)
        agg = np.zeros_like(gvec)
        for c in clients:
            lp, _ = c.train(gp, cfg)
            agg += (F.flatten(lp) - gvec) * c.n / total
        gp = F.unflatten(gvec + agg, gp)
    return gp, (mu, sd), clients, None


if __name__ == "__main__":
    main()
