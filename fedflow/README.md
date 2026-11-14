# FedFlow — traffic intelligence, learned privately

Federated traffic-flow forecasting: city zones (agencies) train one shared model without sharing raw sensor data.
Real PeMS08 data (170 sensors, 62 days), NumPy GRU/MLP with hand-written backprop, FedAvg/FedProx,
pairwise-mask secure aggregation (Diffie–Hellman in network mode), client-level DP with an RDP accountant,
and a 7-screen web app with live in-browser federated training.

## Web app
```
cd web
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in web/dist (deploy to Vercel / Netlify / GitHub Pages)
```

## Experiments (Python 3.10+)
```
cd ml
pip install -r requirements.txt
python3 run_experiments.py   # benchmarks + web assets (~10 min, CPU)
python3 run_scenarios.py     # scarce data, cold start, DP at scale
./demo_network.sh            # networked FL: coordinator + 3 zone clients over HTTP
```
Multi-laptop: `python3 server.py --clients 2` on one machine, `python3 client.py --server http://<ip>:8000 --zone <k>` on each other.

## Key results (12 unseen test days)
| Method | Avg MAE (veh/5 min) |
|---|---|
| FedAvg · GRU (private) | 17.99 |
| Centralised · GRU (pools data) | 18.22 |
| Historical average | 21.99 |
| Persistence | 23.50 |

See `docs/FedFlow_Regionals_Prep.pdf` for the full write-up.

Team NeuroForge · Data: Caltrans PeMS District 8 (PeMS08).
