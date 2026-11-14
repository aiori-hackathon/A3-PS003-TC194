"""FedFlow coordinator - real networked federated learning over HTTP (Python standard library only).

Run on one laptop:      python server.py --clients 3 --rounds 15
Then on each zone laptop (same Wi-Fi):  python client.py --server http://<server-ip>:8000 --zone 0
                                         python client.py --server http://<server-ip>:8000 --zone 1  ...

What the coordinator ever receives from a zone:
  * at registration: its sample count, a Diffie-Hellman public key, and 3 aggregate numbers
    (sum, sum of squares, count) for federated normalisation
  * every round: a MASKED model update. Pairwise masks are derived from DH shared secrets the
    server never learns, so a single update looks like random noise; only the sum is meaningful.
Raw traffic readings never leave the zone laptops.
"""
from __future__ import annotations
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from fedflow.models import MODELS, flatten, unflatten, n_params


class Coordinator:
    def __init__(self, n_clients: int, rounds: int, model: str):
        self.K, self.R = n_clients, rounds
        self.M = MODELS[model]()
        self.model_name = model
        self.global_p = self.M.init(0)
        self.dim = len(flatten(self.global_p))
        self.clients: dict[int, dict] = {}       # cid -> {n, pub, stats}
        self.round = 0                           # 0 = waiting for registrations
        self.updates: dict[int, list] = {}
        self.losses: dict[int, float] = {}
        self.history: list[dict] = []
        self.mu = self.sd = None
        self.lock = threading.Lock()
        self.done = False

    # ------------------------------------------------------------------ protocol
    def register(self, body: dict) -> dict:
        with self.lock:
            cid = int(body["cid"])
            self.clients[cid] = {"n": int(body["n"]), "pub": body["pub"], "stats": body["stats"]}
            print(f"[server] zone {cid} registered ({body['n']:,} private samples). {len(self.clients)}/{self.K}")
            if len(self.clients) == self.K and self.round == 0:
                s = np.array([c["stats"] for c in self.clients.values()], dtype=float).sum(0)
                self.mu = s[0] / s[2]
                self.sd = float(np.sqrt(s[1] / s[2] - self.mu ** 2))
                self.round = 1
                self.t0 = time.time()
                print(f"[server] all zones in. federated normalisation mu={self.mu:.1f} sd={self.sd:.1f}. Round 1 starts.")
        return {"ok": True}

    def state(self) -> dict:
        with self.lock:
            total = sum(c["n"] for c in self.clients.values())
            return {
                "round": self.round, "rounds": self.R, "done": self.done, "model": self.model_name,
                "clients": {str(k): {"pub": v["pub"], "weight": v["n"] / total if total else 0} for k, v in self.clients.items()},
                "mu": self.mu, "sd": self.sd,
                "global": flatten(self.global_p).tolist() if self.round > 0 else None,
                "history": self.history[-5:],
            }

    def update(self, body: dict) -> dict:
        with self.lock:
            if int(body["round"]) != self.round:
                return {"ok": False, "why": "stale round"}
            cid = int(body["cid"])
            self.updates[cid] = body["masked"]
            self.losses[cid] = float(body["loss"])
            preview = ", ".join(f"{v:+.3f}" for v in body["masked"][:4])
            print(f"[server] round {self.round}: masked update from zone {cid}  [{preview}, ...]  local loss {body['loss']:.4f}")
            if len(self.updates) == self.K:
                agg = np.sum([np.asarray(u) for u in self.updates.values()], axis=0)   # masks cancel here
                g = flatten(self.global_p) + agg
                self.global_p = unflatten(g, self.global_p)
                rec = {"round": self.round, "update_norm": float(np.linalg.norm(agg)),
                       "mean_client_loss": float(np.mean(list(self.losses.values()))),
                       "secs": round(time.time() - self.t0, 1)}
                self.history.append(rec)
                print(f"[server] round {self.round} aggregated: |sum of updates| = {rec['update_norm']:.4f}  mean zone loss {rec['mean_client_loss']:.4f}  ({rec['secs']} s)")
                self.updates = {}
                if self.round >= self.R:
                    self.done = True
                    np.savez("results/global_model_networked.npz", **self.global_p)
                    print("[server] training finished -> results/global_model_networked.npz")
                else:
                    self.round += 1
        return {"ok": True}


def make_handler(co: Coordinator):
    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/state"):
                self._send(co.state())
            else:
                self._send({"service": "FedFlow coordinator", "round": co.round, "zones": len(co.clients), "expected": co.K})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/register":
                self._send(co.register(body))
            elif self.path == "/update":
                self._send(co.update(body))
            else:
                self._send({"error": "unknown"}, 404)

        def log_message(self, *a):
            pass
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--model", default="gru", choices=list(MODELS))
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    co = Coordinator(a.clients, a.rounds, a.model)
    print(f"[server] FedFlow coordinator on :{a.port}  model={a.model} ({n_params(co.global_p):,} params)  "
          f"waiting for {a.clients} zones")
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(co)).serve_forever()
