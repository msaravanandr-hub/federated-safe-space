"""Server-side global-model utilities.

The featurizer here MUST mirror client-sim/train.py (deterministic crc32
hashing) so that client weight deltas align with server inference.
Includes: test-set loading and per-round federated evaluation (AUC / F1).
"""
import json
import os
import re
import zlib

import numpy as np

DIM = 4096
MODEL_DIM = DIM + 1  # last slot = bias
_token_re = re.compile(r"[a-z0-9']+")


def _tokens(text):
    t = _token_re.findall(text.lower())
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]  # unigrams + bigrams


def featurize(text):
    # MUST stay identical to client-sim/train.py
    x = np.zeros(MODEL_DIM, dtype=np.float32)
    toks = _tokens(text)
    for tok in toks:
        x[zlib.crc32(tok.encode("utf-8")) % DIM] += 1.0
    if toks:
        n = float(np.linalg.norm(x[:DIM])) or 1.0
        x[:DIM] /= n
    x[DIM] = 1.0
    return x


def load_test_set():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "test_set.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["examples"]


def evaluate_global(w, examples, round_id):
    """AUC (Mann-Whitney) and F1 @ threshold 0.5 on the held-out test set."""
    y = np.asarray([e["label"] for e in examples], dtype=np.float32)
    s = np.asarray([float(w @ featurize(e["text"])) for e in examples], dtype=np.float32)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = float(y.sum()), float((1 - y).sum())
    auc = float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else 0.5
    p = (s > 0).astype(np.float32)
    tp = float(((p == 1) & (y == 1)).sum())
    fp = float(((p == 1) & (y == 0)).sum())
    fn = float(((p == 0) & (y == 1)).sum())
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    return {"round": round_id, "auc": round(auc, 4), "f1": round(f1, 4), "n": int(len(y))}
