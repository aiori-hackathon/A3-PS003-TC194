#!/usr/bin/env bash
# One-laptop demo of networked FL: 1 coordinator + 3 zone clients as separate processes over HTTP.
# For the multi-laptop demo run server.py on one machine and client.py --server http://<ip>:8000 on the others.
cd "$(dirname "$0")"
mkdir -p results
python3 server.py --clients 3 --rounds 10 --port 8000 &
SERVER=$!
sleep 1
PIDS=()
for z in 0 1 2; do python3 client.py --server http://127.0.0.1:8000 --zone $z & PIDS+=($!); done
wait "${PIDS[@]}"
kill $SERVER
