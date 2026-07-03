"""Exact k-nearest-neighbor search — the ground truth.

Every approximate index in this project (IVF, HNSW) is judged by how well its
results agree with this module. It is intentionally simple: compute the
distance to *every* point and take the smallest ``k``. That is O(n·d) per query
and does not scale, which is precisely the problem the other indexes exist to
solve — but it is exact by construction, so it defines "correct".
"""

from __future__ import annotations

import numpy as np

from .vectors import get_metric


class BruteForceIndex:
    """Linear-scan exact k-NN.

    Parameters
    ----------
    vectors:
        Corpus matrix, shape (n, d).
    metric:
        ``"l2"`` (default), ``"cosine"``, ... — see :mod:`vectors`.
    """

    def __init__(self, vectors: np.ndarray, metric: str = "l2") -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.metric = metric
        self._dist = get_metric(metric)

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, distances)`` of the ``k`` nearest points to ``query``.

        Results are sorted by increasing distance. ``argpartition`` gets us the
        top-k in O(n) before a final O(k log k) sort of just those k, which is
        meaningfully faster than sorting all n distances.
        """
        query = np.asarray(query, dtype=np.float32)
        dists = self._dist(query, self.vectors)
        k = min(k, self.n)
        # Partial selection: cheapest k, then sort only those k.
        part = np.argpartition(dists, k - 1)[:k]
        order = np.argsort(dists[part])
        idx = part[order]
        return idx, dists[idx]

    def search_batch(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized search over many queries; returns (m, k) index/dist arrays."""
        queries = np.asarray(queries, dtype=np.float32)
        out_idx = np.empty((len(queries), min(k, self.n)), dtype=np.int64)
        out_dist = np.empty((len(queries), min(k, self.n)), dtype=np.float32)
        for i, q in enumerate(queries):
            out_idx[i], out_dist[i] = self.search(q, k)
        return out_idx, out_dist
