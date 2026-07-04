"""Product Quantization (PQ) — vector compression, from scratch.

PQ is how vector databases fit billions of vectors in RAM. The idea: split each
d-dim vector into ``m`` contiguous sub-vectors, run k-means *independently* in
each subspace to learn a small codebook (``ksub`` centroids, typically 256), and
store each vector as ``m`` centroid ids — one byte per subspace when
``ksub <= 256``. A 128-dim float32 vector (512 bytes) becomes ``m`` bytes; at
m=16 that is a **32× compression**.

Search never decompresses. **Asymmetric Distance Computation (ADC)**: for a
query, precompute the squared distance from each query sub-vector to all ``ksub``
centroids in its subspace — an ``(m, ksub)`` lookup table — then a database
vector's approximate distance is just ``m`` table lookups summed. The query stays
full-precision (asymmetric), which is more accurate than also quantizing it.

This module is the quantizer; :mod:`ivfpq_index` combines it with IVF coarse
quantization over residuals.
"""

from __future__ import annotations

import numpy as np

from .kmeans import KMeans


def _subspace_bounds(d: int, m: int) -> list[tuple[int, int]]:
    """Split ``d`` dimensions into ``m`` contiguous subspaces as evenly as possible.

    Works even when ``d`` isn't divisible by ``m`` (e.g. d=9, m=2 -> sizes 5,4).
    """
    sizes = np.full(m, d // m, dtype=int)
    sizes[: d % m] += 1  # distribute the remainder across the first subspaces
    bounds, start = [], 0
    for s in sizes:
        bounds.append((start, start + int(s)))
        start += int(s)
    return bounds


class ProductQuantizer:
    """Learns per-subspace codebooks and encodes vectors to compact codes.

    Parameters
    ----------
    m:
        Number of subspaces (bytes per code when ``ksub <= 256``).
    ksub:
        Centroids per subspace. 256 uses a full byte; smaller trains faster.
    max_iter, n_init, random_state:
        Passed through to the per-subspace k-means.
    """

    def __init__(
        self,
        m: int = 8,
        ksub: int = 256,
        max_iter: int = 25,
        n_init: int = 1,
        random_state: int | None = None,
    ) -> None:
        if ksub > 256:
            raise ValueError("ksub must be <= 256 so a code fits in one uint8")
        self.m = m
        self.ksub = ksub
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.bounds_: list[tuple[int, int]] = []
        # codebooks_[s] has shape (ksub, subdim_s)
        self.codebooks_: list[np.ndarray] = []
        self.d_: int | None = None

    def fit(self, X: np.ndarray) -> "ProductQuantizer":
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.d_ = X.shape[1]
        self.bounds_ = _subspace_bounds(self.d_, self.m)
        self.codebooks_ = []
        for s, (lo, hi) in enumerate(self.bounds_):
            sub = X[:, lo:hi]
            ksub = min(self.ksub, len(np.unique(sub, axis=0)))
            km = KMeans(
                n_clusters=ksub,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=None if self.random_state is None else self.random_state + s,
            ).fit(sub)
            self.codebooks_.append(km.centers_.astype(np.float32))
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode vectors to codes of shape (n, m), dtype uint8."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        codes = np.empty((len(X), self.m), dtype=np.uint8)
        for s, (lo, hi) in enumerate(self.bounds_):
            sub = X[:, lo:hi]
            cb = self.codebooks_[s]
            # nearest centroid per sub-vector via the ||a-b||^2 expansion
            d2 = (
                np.einsum("ij,ij->i", sub, sub)[:, None]
                + np.einsum("ij,ij->i", cb, cb)[None, :]
                - 2.0 * sub @ cb.T
            )
            codes[:, s] = d2.argmin(axis=1).astype(np.uint8)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Approximately reconstruct vectors from their codes."""
        codes = np.asarray(codes)
        out = np.empty((len(codes), self.d_), dtype=np.float32)
        for s, (lo, hi) in enumerate(self.bounds_):
            out[:, lo:hi] = self.codebooks_[s][codes[:, s]]
        return out

    def distance_tables(self, query: np.ndarray) -> np.ndarray:
        """Squared-L2 lookup table of shape (m, ksub) for one query (ADC).

        ``table[s, c]`` is the squared distance from the query's sub-vector ``s``
        to centroid ``c`` of subspace ``s``.
        """
        query = np.asarray(query, dtype=np.float32)
        table = np.empty((self.m, self.ksub), dtype=np.float32)
        for s, (lo, hi) in enumerate(self.bounds_):
            cb = self.codebooks_[s]                 # (ksub_s, subdim)
            diff = cb - query[lo:hi]                # (ksub_s, subdim)
            d = np.einsum("ij,ij->i", diff, diff)   # (ksub_s,)
            table[s, : len(d)] = d
            if len(d) < self.ksub:                  # ragged codebook (rare)
                table[s, len(d):] = np.inf
        return table

    def adc_distances(self, codes: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Approximate squared distances for coded vectors via table lookups.

        Sums ``table[s, codes[:, s]]`` over the ``m`` subspaces — the whole point
        of PQ: a distance estimate with no decompression and no full-precision
        arithmetic over the database.
        """
        codes = np.asarray(codes)
        total = np.zeros(len(codes), dtype=np.float32)
        for s in range(self.m):
            total += table[s, codes[:, s]]
        return total

    def memory_bytes(self, n: int) -> int:
        """Bytes to store ``n`` coded vectors plus the codebooks."""
        codes = n * self.m                                  # 1 byte per subspace
        books = sum(cb.nbytes for cb in self.codebooks_)
        return codes + books
