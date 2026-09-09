"""LAG-PSA: Low-bandwidth Asynchronous Gradient-compressed
Privacy-preserving Secure Aggregation (server-side core)."""
import numpy as np
from dataclasses import dataclass


@dataclass
class ClientUpdate:
    client_id: str
    round_id: int
    staleness: int            # server_round - client_round
    n_samples: int
    indices: np.ndarray       # top-k parameter indices
    values: np.ndarray        # dequantized float32 deltas on a sparse grid
    # NOTE: in deployment, `values` arrive pairwise-masked; the server can # only recover the SUM across a quorum, never an individual update.


class LagPsaAggregator:
    def __init__(self, n_params: int, staleness_window: int = 3,
                 clip_norm: float = 1.0, dp_sigma: float = 1e-3,
                 min_quorum: int = 5, lr: float = 0.01):
        self.n_params, self.W = n_params, np.zeros(n_params, dtype=np.float32)
        self.staleness_window, self.clip_norm = staleness_window, clip_norm
        self.dp_sigma, self.min_quorum, self.lr = dp_sigma, min_quorum, lr self.pending: list[ClientUpdate] = []
        self.round_id = 0

    def accept(self, u: ClientUpdate) -> bool:
        """Asynchronous acceptance: tolerate intermittent clients."""
        if u.staleness > self.staleness_window:
            return False                      # too stale -> residual stays client-side
        self.pending.append(u)
        return True

    def maybe_aggregate(self) -> dict | None:
        """Aggregate when quorum reached; otherwise keep waiting (no round failure)."""
        if len(self.pending) < self.min_quorum:
            return None
        us = self.pending[: self.max_batch()]
        self.pending = self.pending[len(us):]
        total_w = sum(u.n_samples for u in us) or 1
        agg = np.zeros(self.n_params, dtype=np.float32)
        for u in us:
            agg[u.indices] += (u.n_samples / total_w) * u.values
        norm = float(np.linalg.norm(agg)) or 1.0
        if norm > self.clip_norm:
            agg *= self.clip_norm / norm
        agg += np.random.normal(0, self.dp_sigma, self.n_params).astype(np.float32)
        self.W += self.lr * agg
        self.round_id += 1
        return {"round": self.round_id, "updates": len(us),
                "bytes_in": int(sum(u.indices.nbytes + u.values.nbytes for u in us))}

    def max_batch(self) -> int:
        return len(self.pending)


class LagPsaClient:
    """Top-k + quantization + residual carry-over across disconnects."""

    def __init__(self, topk_frac: float = 0.05, bits: int = 8):
        self.topk_frac, self.bits = topk_frac, bits
        self.residual = None                  # carries unsent signal across drops

    def prepare_update(self, local_weights, global_weights, n_samples):
        delta = _flat(local_weights) - _flat(global_weights)
        if self.residual is not None:
            delta = delta + self.residual     # re-apply what never got sent
        k = max(1, int(self.topk_frac * delta.size))
        idx = np.argpartition(np.abs(delta), -k)[-k:]
        vals = delta[idx]
        # symmetric 8-bit quantization of the selected values
        scale = float(np.max(np.abs(vals))) or 1.0
        q = np.round(vals / scale * (2 ** (self.bits - 1) - 1)).astype(np.int8)
        sent = q.astype(np.float32) / (2 ** (self.bits - 1) - 1) * scale
        self.residual = delta.copy()
        self.residual[idx] -= sent            # carry-over only the unsent error
        return idx, q, scale, n_samples       # -> mask -> queue for upload


def _flat(ws):
    return np.concatenate([w.astype(np.float32).ravel() for w in ws])
