"""IVF-PQ — coarse IVF cells + product-quantized residuals.

This is the memory-efficient index real vector DBs reach for at scale. It layers
two ideas already built from scratch elsewhere in this repo:

1. **IVF coarse quantizer** (:mod:`kmeans`): partition the corpus into ``nlist``
   Voronoi cells; a query only scans the ``nprobe`` nearest cells.
2. **Product quantization** (:mod:`pq`) of the **residual** ``x - centroid`` (not
   the raw vector). Residuals are small and centered, so a shared PQ codebook
   models them well. Each vector is stored as ``m`` bytes instead of ``d``
   float32s.

Search uses Asymmetric Distance Computation: for each probed cell, the query's
residual is turned into an ``(m, ksub)`` distance table once, and every coded
vector in the cell is scored by ``m`` table lookups. The result is approximate
(both the cell pruning and the quantization lose information) but the memory
footprint drops by an order of magnitude — the trade this index exists to make.
"""

from __future__ import annotations

import numpy as np

from .kmeans import KMeans
from .pq import ProductQuantizer
from .vectors import as_mask, exact_subset_search, get_metric


class IVFPQIndex:
    """Inverted-file index with product-quantized residuals.

    Parameters
    ----------
    nlist:
        Number of coarse Voronoi cells.
    m:
        PQ subspaces (bytes per stored vector).
    ksub:
        PQ centroids per subspace (<= 256).
    nprobe:
        Default cells scanned per query.
    random_state:
        Seed for reproducible cells / codebooks.
    """

    def __init__(
        self,
        nlist: int = 128,
        m: int = 8,
        ksub: int = 256,
        nprobe: int = 8,
        random_state: int | None = None,
    ) -> None:
        self.nlist = nlist
        self.m = m
        self.ksub = ksub
        self.nprobe = nprobe
        self.random_state = random_state
        self._coarse_dist = get_metric("l2")

        self.centroids_: np.ndarray | None = None
        self.pq_: ProductQuantizer | None = None
        self.codes_: np.ndarray | None = None          # (n, m) uint8 residual codes
        self.inverted_lists_: list[np.ndarray] = []     # point ids per cell
        self.vectors_: np.ndarray | None = None         # kept only if reranking
        self.n_: int = 0
        self.d_: int = 0
        self.build_stats_: dict = {}

    @property
    def n(self) -> int:
        return self.n_

    def build(self, vectors: np.ndarray, keep_vectors: bool = False) -> "IVFPQIndex":
        """Build the index.

        ``keep_vectors`` retains the full-precision corpus so :meth:`search` can
        exact-rerank a PQ shortlist and recover near-exact recall. It costs the
        memory PQ was saving, so it's opt-in — the production pattern is to keep
        vectors on cheaper storage and rerank only the shortlist.
        """
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n_, self.d_ = X.shape
        nlist = min(self.nlist, self.n_)
        self.vectors_ = X if keep_vectors else None

        # 1. coarse quantizer
        km = KMeans(n_clusters=nlist, n_init=1, random_state=self.random_state).fit(X)
        self.centroids_ = km.centers_
        labels = km.labels_

        # 2. residuals, then a PQ trained on them
        residuals = X - self.centroids_[labels]
        self.pq_ = ProductQuantizer(
            m=self.m, ksub=self.ksub, random_state=self.random_state
        ).fit(residuals)
        self.codes_ = self.pq_.encode(residuals)

        # 3. inverted lists (which point ids live in each cell)
        self.inverted_lists_ = [
            np.where(labels == c)[0].astype(np.int64) for c in range(nlist)
        ]
        self.build_stats_ = {
            "nlist": nlist,
            "m": self.m,
            "ksub": self.ksub,
            "avg_cell_size": self.n_ / nlist,
            "code_bytes_per_vector": self.m,
            "compression_ratio": (self.d_ * 4) / self.m,
        }
        return self

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: int | None = None,
        allowed=None,
        rerank: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k-NN via cell probing + ADC over PQ codes.

        ``rerank`` (> 0) over-fetches that many PQ candidates, then re-scores
        them with exact distances on the full-precision vectors and returns the
        true top-k — recovering most of the recall PQ gives up. Requires the
        index to have been built with ``keep_vectors=True``.
        """
        if self.centroids_ is None or self.pq_ is None:
            raise RuntimeError("IVFPQIndex must be built before search")
        query = np.asarray(query, dtype=np.float32)
        nprobe = self.nprobe if nprobe is None else nprobe
        nprobe = min(nprobe, len(self.centroids_))
        fetch = max(k, rerank) if rerank else k

        centroid_dists = self._coarse_dist(query, self.centroids_)
        probe_cells = np.argpartition(centroid_dists, nprobe - 1)[:nprobe]

        mask = as_mask(allowed, self.n_) if allowed is not None else None
        if mask is not None:
            n_allowed = int(mask.sum())
            # Pre-filtering on a selective filter: probing cells would starve.
            if n_allowed <= max(4 * k, int(0.02 * self.n_)):
                ids = np.where(mask)[0]
                if self.vectors_ is not None:
                    # Full precision available — exact answer over the subset.
                    return exact_subset_search(
                        query, self.vectors_, ids, k, self._coarse_dist
                    )
                # Compressed-only: score every matching code by ADC. Still
                # guarantees k results, just approximate distances.
                scored = []
                for c in range(len(self.centroids_)):
                    cell = self.inverted_lists_[c]
                    sel = cell[mask[cell]] if cell.size else cell
                    if sel.size:
                        table = self.pq_.distance_tables(query - self.centroids_[c])
                        scored.append((sel, self.pq_.adc_distances(self.codes_[sel], table)))
                if not scored:
                    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
                a_ids = np.concatenate([s[0] for s in scored])
                a_d = np.concatenate([s[1] for s in scored])
                kk = min(k, a_ids.size)
                part = np.argpartition(a_d, kk - 1)[:kk]
                order = np.argsort(a_d[part])
                top = part[order]
                return a_ids[top], np.sqrt(np.maximum(a_d[top], 0.0)).astype(np.float32)

        cand_ids: list[np.ndarray] = []
        cand_dists: list[np.ndarray] = []
        for c in probe_cells:
            ids = self.inverted_lists_[c]
            if mask is not None and ids.size:
                ids = ids[mask[ids]]
            if ids.size == 0:
                continue
            # Residual of the query wrt this cell's centroid, then ADC.
            table = self.pq_.distance_tables(query - self.centroids_[c])
            dists = self.pq_.adc_distances(self.codes_[ids], table)
            cand_ids.append(ids)
            cand_dists.append(dists)

        if not cand_ids:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        all_ids = np.concatenate(cand_ids)
        all_d = np.concatenate(cand_dists)
        kk = min(fetch, all_ids.size)
        part = np.argpartition(all_d, kk - 1)[:kk]
        order = np.argsort(all_d[part])
        top = part[order]
        top_ids = all_ids[top]

        if rerank and self.vectors_ is not None:
            # Exact re-score of the PQ shortlist, then the true top-k.
            exact = self._coarse_dist(query, self.vectors_[top_ids])
            ro = np.argsort(exact)[:k]
            return top_ids[ro], exact[ro].astype(np.float32)

        # ADC returns squared distances; sqrt to report on the L2 scale.
        top_ids = top_ids[:k]
        return top_ids, np.sqrt(np.maximum(all_d[top][:k], 0.0)).astype(np.float32)

    def memory_bytes(self) -> int:
        """Estimated index footprint: codes + codebooks + coarse centroids."""
        pq_mem = self.pq_.memory_bytes(self.n_) if self.pq_ else 0
        cent = self.centroids_.nbytes if self.centroids_ is not None else 0
        return pq_mem + cent
