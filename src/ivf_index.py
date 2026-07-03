"""IVF (inverted file) index — cluster once, probe a few cells at query time.

The idea is the classic space/accuracy trade of coarse quantization:

* **Build**: run k-means to carve the corpus into ``nlist`` Voronoi cells and
  store an inverted list of point ids per cell.
* **Search**: instead of scanning all ``n`` points, find the ``nprobe`` cells
  whose centroids are nearest the query and scan only the points inside them.

If a query's true neighbors all live in the probed cells, recall is perfect; if
some spill into unprobed neighboring cells, we miss them. Turning ``nprobe`` up
trades latency for recall — and quantifying exactly that curve is the whole
point of the benchmark suite. Because each probed cell is scanned *exactly*,
IVF's only error source is this cell-boundary spill.
"""

from __future__ import annotations

import numpy as np

from .kmeans import KMeans
from .vectors import get_metric


class IVFIndex:
    """Inverted-file index over a fixed corpus.

    Parameters
    ----------
    nlist:
        Number of Voronoi cells (k-means clusters). A common rule of thumb is
        ``nlist ~= sqrt(n)``.
    metric:
        Distance metric for the *fine* (within-cell) comparison.
    nprobe:
        Default number of cells to scan per query; overridable per-search.
    random_state:
        Seed passed through to k-means for reproducible cells.
    """

    def __init__(
        self,
        nlist: int = 128,
        metric: str = "l2",
        nprobe: int = 8,
        random_state: int | None = None,
    ) -> None:
        self.nlist = nlist
        self.metric = metric
        self.nprobe = nprobe
        self.random_state = random_state
        self._dist = get_metric(metric)

        self.vectors_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        # Inverted lists: for each cell, the array of point ids assigned to it.
        self.inverted_lists_: list[np.ndarray] = []
        self.build_stats_: dict = {}

    @property
    def n(self) -> int:
        return 0 if self.vectors_ is None else self.vectors_.shape[0]

    def build(self, vectors: np.ndarray) -> "IVFIndex":
        self.vectors_ = np.ascontiguousarray(vectors, dtype=np.float32)
        nlist = min(self.nlist, self.n)

        km = KMeans(n_clusters=nlist, random_state=self.random_state, n_init=1)
        labels = km.fit_predict(self.vectors_)
        self.centroids_ = km.centers_

        # Group point ids by cell into the inverted lists.
        self.inverted_lists_ = [
            np.where(labels == c)[0].astype(np.int64) for c in range(nlist)
        ]
        self.build_stats_ = {
            "nlist": nlist,
            "kmeans_iters": km.n_iter_,
            "kmeans_inertia": km.inertia_,
            "avg_cell_size": self.n / nlist,
            "empty_cells": int(sum(len(l) == 0 for l in self.inverted_lists_)),
        }
        return self

    def search(
        self, query: np.ndarray, k: int = 10, nprobe: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, distances)`` of approximate k nearest neighbors."""
        if self.vectors_ is None or self.centroids_ is None:
            raise RuntimeError("IVFIndex must be built before search")
        query = np.asarray(query, dtype=np.float32)
        nprobe = self.nprobe if nprobe is None else nprobe
        nprobe = min(nprobe, len(self.centroids_))

        # Coarse step: rank cells by centroid distance, take the nearest nprobe.
        centroid_dists = self._dist(query, self.centroids_)
        probe_cells = np.argpartition(centroid_dists, nprobe - 1)[:nprobe]

        # Gather candidate point ids from the probed cells.
        candidate_ids = np.concatenate(
            [self.inverted_lists_[c] for c in probe_cells]
        ) if nprobe else np.empty(0, dtype=np.int64)
        if candidate_ids.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        # Fine step: exact distances to just the candidates, then top-k.
        cand_vecs = self.vectors_[candidate_ids]
        dists = self._dist(query, cand_vecs)
        kk = min(k, candidate_ids.size)
        part = np.argpartition(dists, kk - 1)[:kk]
        order = np.argsort(dists[part])
        top = part[order]
        return candidate_ids[top], dists[top]

    def search_batch(
        self, queries: np.ndarray, k: int = 10, nprobe: int | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        idxs, dsts = [], []
        for q in np.asarray(queries, dtype=np.float32):
            i, d = self.search(q, k, nprobe)
            idxs.append(i)
            dsts.append(d)
        return idxs, dsts
