"""Simulated federated clients running REAL local training.

Each simulated client:
  1. registers with the API and gets a JWT
  2. holds a private, non-IID (label-skewed) partition of a real labeled
     text corpus — this text NEVER leaves the client
  3. every round: trains the logistic-regression toxicity model locally on its
     partition (numpy SGD), computes the weight delta vs the global model,
     compresses it (top-k + 8-bit quantization), and uploads through the
     LAG-PSA pipeline (async staleness window, clipping, DP noise, quorum)
  4. goes offline with probability DROP_RATE, growing its staleness

The server evaluates AUC / F1 on a held-out test set after each aggregation —
watch those numbers converge for real.
"""
import argparse
import base64
import os
import random
import time

import numpy as np
import requests

from train import MODEL_DIM, augment, featurize, load_corpus, partition_non_iid, train_local


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
        "username": username, "password": "***", "consent_federated": True,
    }, timeout=10)
    if r.status_code == 409:
        r = requests.post(f"{base}/auth/login", data={
            "username": username, "password": "***",
        }, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("token") or data["access_token"]


def get_global_weights(base: str):
    j = requests.get(f"{base}/federation/global-weights", timeout=15).json()
    return np.frombuffer(base64.b64decode(j["weights_b64"]), dtype=np.float32).copy()


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
    p.add_argument("--topk-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    wait_for_api(args.api)
    print(f"[sim] API up at {args.api}")

    tokens = [register(args.api, f"sim_{random.randrange(10**6)}_{i}")
              for i in range(args.clients)]
    print(f"[sim] registered {args.clients} clients")

    corpus = augment(load_corpus(), np.random.default_rng(args.seed))
    blocks = partition_non_iid(corpus, args.clients)
    for i, b in enumerate(blocks):
        n_pos = sum(e["label"] for e in b)
        print(f"[sim] client {i}: {len(b)} local samples ({n_pos} toxic, {len(b) - n_pos} clean) — text stays on device")
    texts = [[e["text"] for e in b] for b in blocks]
    labels = [[e["label"] for e in b] for b in blocks]

    staleness = [0] * args.clients
    w_global = get_global_weights(args.api)
    server_round = 0

    for rnd in range(1, args.rounds + 1):
        uploads = 0
        for i in range(args.clients):
            if random.random() < args.drop_rate:
                staleness[i] += 1
                print(f"[sim] round {rnd}: client {i} offline (stale={staleness[i]})")
                continue
            w_local = train_local(w_global, texts[i], labels[i])
            delta = w_local - w_global
            idx, vals = compress(delta, args.topk_frac)
            r = requests.post(f"{args.api}/federation/upload-update", json={
                "client_id": f"sim-{i}", "round_id": rnd,
                "staleness": staleness[i], "n_samples": len(labels[i]),
                "indices": idx, "values": vals,
            }, headers={"Authorization": f"Bearer {tokens[i]}"}, timeout=30)
            r.raise_for_status()
            uploads += 1
            staleness[i] = 0
            time.sleep(0.3)  # bandwidth-cap stand-in

        mt = requests.get(f"{args.api}/metrics", timeout=10).json()
        gm = requests.get(f"{args.api}/federation/global-model", timeout=10).json()
        if gm["round"] > server_round:
            server_round = gm["round"]
            w_global = get_global_weights(args.api)
        ev = mt.get("eval") or {}
        print(f"[sim] round {rnd}: uploads={uploads}/{args.clients} | server round {gm['round']} | "
              f"eval: auc={ev.get('auc')} f1={ev.get('f1')} (n={ev.get('n')})")

    print("[sim] done — the global model now scores /posts; see GET /metrics")


if __name__ == "__main__":
    main()
