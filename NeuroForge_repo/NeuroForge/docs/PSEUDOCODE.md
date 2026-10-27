# Pseudocode — FedAvg traffic forecasting with byte accounting

This is the snippet for the mentor demonstration. Every line maps to real code (file in brackets).

```text
for seed in (11, 22, 33):                                    # run_experiments.py
    set_seed(seed)
    tr, va, te = chrono_split(load_metr_la(), 0.7, 0.1, 0.2)  # data.py  (time order, no shuffling)
    mu, sd     = fit_zscore(tr)                               # data.py  (training readings only)

    # E1 - centralised baseline: same GRU, same optimiser, R*E epochs on pooled windows
    B_raw = 207 * len(tr) * 4                                 # every training reading uploaded once
    w_c   = train_central(pool(tr), epochs=R*E)               # simulate.run_central

    for mode in ("iid", "noniid"):                            # E2, E3
        clients = partition(sensors, K=4, mode, seed)         # partition.py (random / k-means on lat-lon)
        w = init_gru(seed);  B = 24 * K                       # one-off z-score stats (count,sum,sumsq)
        for t in 1..R:                                        # simulate.run_federated
            ups = []
            for c in clients:                                 # all K take part (sync FedAvg)
                B += nbytes(w)                                # download w_t        (14,988 B)
                w_k = local_train(c.train_windows, w, E)      # raw windows never leave c
                ups.append((w_k, c.n));  B += nbytes(w_k) + 8 # upload w_k and n_k  (14,996 B)
            n = sum(n_k for _, n_k in ups)
            w = sum(n_k / n * w_k for w_k, n_k in ups)        # FedAvg, Eq. (1)
            log(seed, mode, t, B, val_mae(w), val_rmse(w))    # same schema for every method
        t_best = argmin_t val_mae                             # validation only
        report(test_mae(w[t_best]), test_rmse(w[t_best]))     # test period touched once
```

## Byte accounting check

| quantity | formula | value |
|---|---|---|
| model size \|w\| | 3,747 params × 4 B | 14,988 B |
| download per round | K · \|w\| | 59,952 B |
| upload per round | K · (\|w\| + 8) | 59,984 B |
| **per round** | 2·K·\|w\| + 8K | **119,936 B** |
| 50 rounds | × 50 | 5,996,800 B |
| centralised raw upload | 207 × 23,990 × 4 B | 19,863,720 B |

`python scripts/smoke.py` runs 3 rounds and prints the logged bytes next to the formula.

## Extension codecs (X1)

```text
delta = w_k - w_t                      # what the client actually changed this round
q     = round(delta / s), s = max|delta| / 127      # 1 byte per value + 4-byte scale per tensor
server reconstructs  w_k ≈ w_t + q * s
```
