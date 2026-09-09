"""Real local training for simulated clients.

Model: binary logistic regression over a deterministic hashing featurizer.
- features: unigrams + bigrams hashed into 4096 buckets (crc32 — stable across
  processes), L2-normalized counts, plus a bias term in the last slot.
- training: SGD on binary cross-entropy, seeded and reproducible.
- augmentation: template-generated examples expand the handwritten seed corpus.

Deliberately numpy-only (no torch/sklearn): the point is REAL gradients on REAL
text through the LAG-PSA pipeline, with a tiny footprint. Raw text never leaves
the client — only compressed weight deltas travel.
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
    # MUST stay identical to backend/app/federation/model.py
    x = np.zeros(MODEL_DIM, dtype=np.float32)
    toks = _tokens(text)
    for tok in toks:
        x[zlib.crc32(tok.encode("utf-8")) % DIM] += 1.0
    if toks:
        n = float(np.linalg.norm(x[:DIM])) or 1.0
        x[:DIM] /= n
    x[DIM] = 1.0
    return x


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_local(w_global, texts, labels, epochs=15, lr=0.1):
    w = w_global.copy()
    X = np.stack([featurize(t) for t in texts])
    y = np.asarray(labels, dtype=np.float32)
    for _ in range(epochs):
        for i in np.random.permutation(len(y)):
            p = sigmoid(float(w @ X[i]))
            w += lr * (y[i] - p) * X[i]
    return w


_ADJ = ["stupid", "idiotic", "useless", "pathetic", "worthless", "dumb", "trash", "lame", "clueless", "annoying"]
_NOUN = ["idiot", "loser", "moron", "clown", "fool", "jerk", "fraud", "disgrace"]
_TOX_T = [
    "you are such a {a} {n}",
    "what a {a} {n} move",
    "only a {n} would post this",
    "shut up you {a} {n}",
    "your take is {a} garbage",
    "people like you are {a}",
    "get lost you {n}",
    "this is {a} nonsense from a {n}",
    "nobody asked, {n}",
    "you keep embarrassing yourself, you {n}",
]
_THING = ["the project", "my homework", "this game", "the recipe", "the tutorial", "my code", "the meetup"]
_PADJ = ["helpful", "solid", "friendly", "awesome", "thoughtful", "great"]
_CLEAN_T = [
    "has anyone tried {thing} yet?",
    "thanks for sharing, really {adj} post",
    "love how {adj} this community is",
    "working on {thing} today",
    "great question about {thing}",
    "any tips for {thing}?",
    "just finished {thing}, felt {adj}",
    "hello everyone, happy to be here",
    "what do you all think about {thing}?",
    "appreciate the {adj} feedback",
]


def load_corpus():
    path = os.path.join(os.path.dirname(__file__), "data", "train_corpus.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["examples"]


def augment(seed_examples, rng, target=600):
    """Seeded template expansion — real labeled text, reproducible."""
    out = list(seed_examples)
    while len(out) < target:
        if rng.random() < 0.5:
            text = rng.choice(_TOX_T).format(a=rng.choice(_ADJ), n=rng.choice(_NOUN))
            out.append({"text": text, "label": 1})
        else:
            text = rng.choice(_CLEAN_T).format(thing=rng.choice(_THING), adj=rng.choice(_PADJ))
            out.append({"text": text, "label": 0})
    return out


def partition_non_iid(examples, n_clients):
    """Extreme label-skew partition: sort by label, contiguous blocks.
    Some clients hold (almost) only clean or only toxic text — the hard case."""
    ex = sorted(examples, key=lambda e: e["label"])
    blocks, size = [], len(ex) // n_clients
    for i in range(n_clients):
        lo, hi = i * size, (len(ex) if i == n_clients - 1 else (i + 1) * size)
        blocks.append(ex[lo:hi])
    return blocks
