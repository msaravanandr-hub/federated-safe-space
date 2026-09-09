"""Simulated federated clients exercising the LAG-PSA loop over REST.

Protocol-loop demo: updates are synthetic (a planted shared signal plus
per-client noise), NOT real model gradients. Real on-device training plugs in
with the mobile client (next batch). Each simulated client:

  1. registers with the API and gets a JWT
  2. per round: compresses a toy delta (top-k + 8-bit quantization),
     goes offline with probability DROP_RATE, applies a bandwidth delay,
     uploads (staleness grows while it is offline)
  3. the server aggregates once a quorum (default 5) of pending updates exists
"""
import argparse
import os
import random
import time

import numpy as np
import requests


def wait_for_api(base: str, timeout_s: int = 90) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{base}/health", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise SystemExit(f"API not reachable at {base}")


def register(base: str, username: str) -> str:
    r = requests.post(f"{base}/auth/register", json={
        "username": username, "password": "sim-pass-123", "consent_federated": True,
    }, timeout=10)
    if r.status_code == 409:  # username exists from a previous run -> log in
        r = requests.post(f"{base}/auth/login", data={
            "username": username, "password": "sim-pass-123",
        }, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("token") or data["access_token"]


def compress(delta: np.ndarray, topk_frac: float, bits: int = 8):
    k = max(1, int(topk_frac * delta.size))
    idx = np.argpartition(np.abs(delta), -k)[-k:]
    vals = delta[idx]
    scale = float(np.max(np.abs(vals))) or 1.0
    q = np.round(vals / scale * (2 ** (bits - 1) - 1))
    deq = q / (2 ** (bits - 1) - 1) * scale
    return idx.astype(int).tolist(), deq.astype(float).tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000"))
    p.add_argument("--clients", type=int, default=int(os.getenv("N_CLIENTS", "8")))
    p.add_argument("--rounds", type=int, default=int(os.getenv("ROUNDS", "5")))
    p.add_argument("--drop-rate", type=float, default=float(os.getenv("DROP_RATE", "0.3")))
    p.add_argument("--n-params", type=int, default=100_000)
    p.add_argument("--topk-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    wait_for_api(args.api)
    print(f"[sim] API up at {args.api}")

    tokens = [register(args.api, f"sim_{random.randrange(10**6)}_{i}")
              for i in range(args.clients)]
    print(f"[sim] registered {args.clients} clients")

    signal = rng.normal(0, 0.05, args.n_params)   # shared "global" direction
    staleness = [0] * args.clients

    for rnd in range(1, args.rounds + 1):
        uploads = 0
        for i in range(args.clients):
            if random.random() < args.drop_rate:
                staleness[i] += 1
                print(f"[sim] round {rnd}: client {i} offline (stale={staleness[i]})")
                continue
            delta = signal + rng.normal(0, 0.02, args.n_params)  # toy local update
            idx, vals = compress(delta, args.topk_frac)
            r = requests.post(f"{args.api}/federation/upload-update", json={
                "client_id": f"sim-{i}", "round_id": rnd,
                "staleness": staleness[i], "n_samples": random.randint(20, 200),
                "indices": idx, "values": vals,
            }, headers={"Authorization": f"Bearer {tokens[i]}"}, timeout=30)
            r.raise_for_status()
            uploads += 1
            staleness[i] = 0
            time.sleep(0.3)  # bandwidth-cap stand-in
        gm = requests.get(f"{args.api}/federation/global-model", timeout=10).json()
        print(f"[sim] round {rnd}: uploads={uploads}/{args.clients} | server: {gm}")

    print("[sim] done - inspect GET /metrics and GET /federation/global-model")


if __name__ == "__main__":
    main()
