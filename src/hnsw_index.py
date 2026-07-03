"""HNSW — Hierarchical Navigable Small World graph index, from scratch.

HNSW is the algorithm behind most production vector databases (FAISS-HNSW,
hnswlib, Qdrant, Weaviate). It builds a multi-layer proximity graph and answers
queries by greedy graph traversal instead of clustering. This is a direct
implementation of Malkov & Yashunin (2016), "Efficient and robust approximate
nearest neighbor search using Hierarchical Navigable Small World graphs".

Mental model
------------
* Think of a skip list generalized to a graph. Each inserted point gets a random
  maximum *layer* drawn from an exponential distribution, so layers thin out
  toward the top. The top layers are a sparse "express network" of long-range
  hops; layer 0 contains every point and holds the dense, short-range links.
* **Search** enters at the single top node, greedily walks toward the query on
  each layer (ef=1) until it can't get closer, drops a layer, and repeats. Only
  at layer 0 does it widen the beam to ``ef_search`` to collect the final k.
* **Insert** does the same descent to find good entry points, then at each layer
  from the node's top level down to 0 runs a beam search of width
  ``ef_construction`` and wires the new node to ``M`` selected neighbors
  (bidirectionally), pruning over-full neighbor lists.

The key knobs — ``M`` (graph degree), ``ef_construction`` (build beam width),
``ef_search`` (query beam width) — trade build time / memory / recall, and the
benchmark suite maps that surface out explicitly.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from .vectors import get_metric


class HNSWIndex:
    """A hierarchical navigable small world graph over a fixed corpus.

    Parameters
    ----------
    M:
        Number of bidirectional links created per node per layer (layer 0 gets
        up to ``2*M``). Higher M -> better recall, more memory, slower build.
    ef_construction:
        Beam width during insertion. Higher -> better graph, slower build.
    ef_search:
        Default beam width during query. Higher -> better recall, slower query.
        Overridable per-search.
    metric:
        Distance metric (``"l2"`` | ``"cosine"`` | ...).
    random_state:
        Seed for the layer-assignment RNG (reproducible graphs).
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "l2",
        random_state: int | None = None,
    ) -> None:
        self.M = M
        self.M_max0 = 2 * M          # layer-0 nodes may hold twice as many links
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric
        self._dist = get_metric(metric)
        self._rng = np.random.default_rng(random_state)
        # Level-generation normalization factor (Malkov & Yashunin, eq. for mL).
        self._mL = 1.0 / math.log(M) if M > 1 else 1.0

        self.vectors_: np.ndarray | None = None
        # graph[layer] maps node id -> list of neighbor ids on that layer.
        self.graph_: list[dict[int, list[int]]] = []
        self.entry_point_: int | None = None
        self.max_level_: int = -1
        self.build_stats_: dict = {}

    @property
    def n(self) -> int:
        return 0 if self.vectors_ is None else self.vectors_.shape[0]

    # ------------------------------------------------------------------ #
    # distance helpers
    # ------------------------------------------------------------------ #
    def _d_point(self, q: np.ndarray, node: int) -> float:
        """Distance from a query vector to a single stored node."""
        return float(self._dist(q, self.vectors_[node : node + 1])[0])

    def _random_level(self) -> int:
        """Draw a node's maximum layer ~ floor(-ln(U) * mL)."""
        u = max(self._rng.random(), 1e-12)
        return int(math.floor(-math.log(u) * self._mL))

    # ------------------------------------------------------------------ #
    # core graph search (Algorithm 2 in the paper)
    # ------------------------------------------------------------------ #
    def _search_layer(
        self, q: np.ndarray, entry_points: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """Beam search on one layer; returns up to ``ef`` nearest as (dist, id).

        Maintains a min-heap ``candidates`` of the frontier to expand and a
        max-heap ``results`` (stored with negated distances) of the best ef
        found so far. Stops when the closest unexpanded candidate is farther
        than the current worst result.
        """
        visited: set[int] = set(entry_points)
        candidates: list[tuple[float, int]] = []   # min-heap by distance
        results: list[tuple[float, int]] = []       # max-heap: (-dist, id)

        for ep in entry_points:
            d = self._d_point(q, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            worst = -results[0][0]
            if dist_c > worst:
                break  # nothing closer left to explore
            for nb in self.graph_[layer].get(c, ()):  # expand neighbors
                if nb in visited:
                    continue
                visited.add(nb)
                d = self._d_point(q, nb)
                worst = -results[0][0]
                if d < worst or len(results) < ef:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(results, (-d, nb))
                    if len(results) > ef:
                        heapq.heappop(results)  # drop current worst
        return [(-nd, i) for nd, i in results]

    def _select_neighbors_heuristic(
        self, q: np.ndarray, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """Pick ``m`` neighbors favoring diversity (Algorithm 4, simplified).

        A candidate is only kept if it is closer to the query than it is to any
        already-selected neighbor. This avoids clustering all links in one
        direction and keeps the graph navigable — plain "m closest" tends to
        create redundant edges and hurts recall.
        """
        # Ascending by distance to the query.
        cand = sorted(candidates, key=lambda x: x[0])
        selected: list[int] = []
        for dist_q, c in cand:
            if len(selected) >= m:
                break
            keep = True
            for s in selected:
                d_cs = self._d_point(self.vectors_[c], s)
                if d_cs < dist_q:  # closer to a chosen neighbor than to query
                    keep = False
                    break
            if keep:
                selected.append(c)
        # If the heuristic was too aggressive, backfill with nearest remaining.
        if len(selected) < m:
            for _, c in cand:
                if c not in selected:
                    selected.append(c)
                    if len(selected) >= m:
                        break
        return selected

    # ------------------------------------------------------------------ #
    # insertion (Algorithm 1)
    # ------------------------------------------------------------------ #
    def _insert(self, node: int) -> None:
        q = self.vectors_[node]
        level = self._random_level()

        # Ensure a graph dict exists for every layer up to this node's level.
        while len(self.graph_) <= level:
            self.graph_.append({})
        for lyr in range(level + 1):
            self.graph_[lyr].setdefault(node, [])

        # First node ever inserted becomes the entry point.
        if self.entry_point_ is None:
            self.entry_point_ = node
            self.max_level_ = level
            return

        ep = self.entry_point_
        # Phase 1: greedy descent from the top down to level+1 (beam width 1).
        for lyr in range(self.max_level_, level, -1):
            nearest = self._search_layer(q, [ep], ef=1, layer=lyr)
            ep = min(nearest, key=lambda x: x[0])[1]

        # Phase 2: from min(level, max_level) down to 0, connect neighbors.
        for lyr in range(min(level, self.max_level_), -1, -1):
            found = self._search_layer(q, [ep], self.ef_construction, lyr)
            m = self.M_max0 if lyr == 0 else self.M
            neighbors = self._select_neighbors_heuristic(q, found, self.M)

            # Wire node <-> neighbor bidirectionally.
            self.graph_[lyr][node] = list(neighbors)
            for nb in neighbors:
                nb_links = self.graph_[lyr].setdefault(nb, [])
                nb_links.append(node)
                # Prune the neighbor's list if it now exceeds the layer cap.
                if len(nb_links) > m:
                    nb_cands = [(self._d_point(self.vectors_[nb], x), x) for x in nb_links]
                    self.graph_[lyr][nb] = self._select_neighbors_heuristic(
                        self.vectors_[nb], nb_cands, m
                    )
            ep = min(found, key=lambda x: x[0])[1]

        # Raise the global entry point if this node reached a new top layer.
        if level > self.max_level_:
            self.max_level_ = level
            self.entry_point_ = node

    def build(self, vectors: np.ndarray) -> "HNSWIndex":
        import time

        self.vectors_ = np.ascontiguousarray(vectors, dtype=np.float32)
        self.graph_ = []
        self.entry_point_ = None
        self.max_level_ = -1

        t0 = time.perf_counter()
        # Insert in a shuffled order so layer assignment isn't correlated with
        # any ordering already present in the corpus.
        order = self._rng.permutation(self.n)
        for node in order:
            self._insert(int(node))
        build_time = time.perf_counter() - t0

        degrees = [len(v) for v in self.graph_[0].values()] if self.graph_ else [0]
        self.build_stats_ = {
            "n": self.n,
            "levels": len(self.graph_),
            "entry_point": self.entry_point_,
            "avg_degree_layer0": float(np.mean(degrees)),
            "max_degree_layer0": int(np.max(degrees)),
            "build_time_s": build_time,
        }
        return self

    # ------------------------------------------------------------------ #
    # query (Algorithm 5)
    # ------------------------------------------------------------------ #
    def search(
        self, query: np.ndarray, k: int = 10, ef_search: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.vectors_ is None or self.entry_point_ is None:
            raise RuntimeError("HNSWIndex must be built before search")
        q = np.asarray(query, dtype=np.float32)
        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, k)  # beam must be at least k wide to return k results

        ep = self.entry_point_
        # Greedy descent through the express layers (beam width 1).
        for lyr in range(self.max_level_, 0, -1):
            nearest = self._search_layer(q, [ep], ef=1, layer=lyr)
            ep = min(nearest, key=lambda x: x[0])[1]

        # Wide beam search at the base layer.
        found = self._search_layer(q, [ep], ef, layer=0)
        found.sort(key=lambda x: x[0])
        top = found[:k]
        idx = np.array([i for _, i in top], dtype=np.int64)
        dist = np.array([d for d, _ in top], dtype=np.float32)
        return idx, dist

    def search_batch(
        self, queries: np.ndarray, k: int = 10, ef_search: int | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        idxs, dsts = [], []
        for qv in np.asarray(queries, dtype=np.float32):
            i, d = self.search(qv, k, ef_search)
            idxs.append(i)
            dsts.append(d)
        return idxs, dsts
