"""FedFlow zone client - holds one zone's private sensor data, trains locally, uploads masked updates.

    python client.py --server http://192.168.1.10:8000 --zone 0

Secure aggregation: every pair of zones derives a shared secret with Diffie-Hellman
(RFC 3526 2048-bit MODP group). From that secret both expand the same random mask each round;
the lower-numbered zone adds it, the other subtracts it. The coordinator only ever sees masked
vectors, and the masks cancel exactly when it sums them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import secrets
import time
import urllib.request

import numpy as np
from fedflow.data import PeMS
from fedflow.fl import Client, FLConfig, evaluate
from fedflow.models import MODELS, flatten, unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
# RFC 3526 group 14 (2048-bit MODP), generator 2
P = int("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
        "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
        "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
        "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
        "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
        "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
G = 2


def http(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def mask_for(my_id: int, others: dict[int, int], secret: int, rnd: int, dim: int) -> np.ndarray:
    m = np.zeros(dim)
    for oid, pub in others.items():
        shared = pow(pub, secret, P)                                 # DH: same value on both sides
        seed = int.from_bytes(hashlib.sha256(f"{shared}:{rnd}".encode()).digest()[:8], "big")
        r = np.random.default_rng(seed).normal(scale=10.0, size=dim)
        m += r if my_id < oid else -r
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--zone", type=int, required=True)
    ap.add_argument("--zones", type=int, default=8, help="how the city is partitioned (data side)")
    ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()

    d = PeMS(os.path.join(HERE, "data/pems08.npz"), os.path.join(HERE, "data/distance.csv"))
    zones = d.zones(a.zones)
    sensors = np.where(zones == a.zone)[0]
    x = d.flow[: d.t_train, sensors].astype(float)
    stats = [float(x.sum()), float((x ** 2).sum()), float(x.size)]

    secret = secrets.randbelow(P - 3) + 2
    pub = pow(G, secret, P)
    st = http(a.server + "/state")
    M = MODELS[st["model"]]()
    probe = Client(a.zone, sensors, d, M, 0, 1, a.zone)
    http(a.server + "/register", {"cid": a.zone, "n": probe.n, "pub": str(pub), "stats": stats})
    print(f"[zone {a.zone}] {len(sensors)} sensors, {probe.n:,} training windows. Raw data stays here. Registered.")

    done_round = 0
    client = None
    while True:
        st = http(a.server + "/state")
        if st["done"]:
            break
        if st["round"] == 0 or st["round"] == done_round:
            time.sleep(0.5)
            continue
        if client is None:
            client = Client(a.zone, sensors, d, M, st["mu"], st["sd"], a.zone)
            cfg = FLConfig(model=st["model"], local_steps=a.steps)
        rnd = st["round"]
        gp = unflatten(np.asarray(st["global"]), M.init(0))
        lp, loss = client.train(gp, cfg)
        delta = (flatten(lp) - flatten(gp)) * st["clients"][str(a.zone)]["weight"]   # FedAvg weight n_k / n
        others = {int(k): int(v["pub"]) for k, v in st["clients"].items() if int(k) != a.zone}
        masked = delta + mask_for(a.zone, others, secret, rnd, len(delta))
        http(a.server + "/update", {"cid": a.zone, "round": rnd, "masked": masked.tolist(), "loss": loss})
        print(f"[zone {a.zone}] round {rnd}: trained {cfg.local_steps} steps, loss {loss:.4f}, sent masked update "
              f"(|true| {np.linalg.norm(delta):.3f}, |sent| {np.linalg.norm(masked):.1f})")
        done_round = rnd

    st = http(a.server + "/state")
    gp = unflatten(np.asarray(st["global"]), M.init(0))
    m = evaluate(M, gp, d, sensors, st["mu"], st["sd"], d.t_val, d.T)
    print(f"[zone {a.zone}] final global model on this zone's unseen test days: MAE {m['avg']['mae']:.2f} veh/5min, "
          f"MAPE {m['avg']['mape']:.1f}%")


if __name__ == "__main__":
    main()
