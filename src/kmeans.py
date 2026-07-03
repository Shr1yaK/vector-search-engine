"""k-means clustering from scratch (Lloyd's algorithm + k-means++ seeding).

This is a *core learning target*, not a convenience import — the IVF index
needs to partition the corpus into cells, and that partitioning is exactly a
k-means problem. Implementing it here (instead of calling ``sklearn``) is the
point: it forces the vectorized distance math, the empty-cluster handling, and
the convergence bookkeeping to be explicit and testable.

Algorithm
---------
1. **k-means++ seeding**: pick the first center at random, then pick each
   subsequent center with probability proportional to its squared distance to
   the nearest already-chosen center. This spreads seeds out and dramatically
   improves both convergence speed and final inertia versus uniform-random
   seeding.
2. **Lloyd iterations**: assign every point to its nearest center, then move
   each center to the mean of its assigned points. Repeat until assignments
   stop changing (or center movement falls below ``tol``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pairwise_sq_dists(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared L2 distances between every point and every center.

    Returns an (n_points, n_centers) matrix using the
    ||p||^2 + ||c||^2 - 2 p.c expansion so the heavy lifting is one GEMM.
    """
    p_sq = np.einsum("ij,ij->i", points, points)[:, None]      # (n, 1)
    c_sq = np.einsum("ij,ij->i", centers, centers)[None, :]    # (1, k)
    cross = points @ centers.T                                  # (n, k)
    d2 = p_sq + c_sq - 2.0 * cross
    # Floating-point error can produce tiny negatives; clamp to 0.
    return np.maximum(d2, 0.0)


@dataclass
class KMeansResult:
    centers: np.ndarray       # (k, d)
    labels: np.ndarray        # (n,) assignment of each point to a center
    inertia: float            # sum of squared distances to assigned centers
    n_iter: int               # Lloyd iterations actually run


class KMeans:
    """Lloyd's k-means with k-means++ initialization.

    Parameters
    ----------
    n_clusters:
        Number of clusters ``k``.
    max_iter:
        Cap on Lloyd iterations.
    tol:
        Convergence threshold on the total center movement (squared).
    n_init:
        Number of independent restarts; the run with the lowest inertia wins
        (k-means is non-convex, so restarts matter).
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 3,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    # --- seeding --------------------------------------------------------- #
    def _kmeanspp_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = X.shape[0]
        k = self.n_clusters
        centers = np.empty((k, X.shape[1]), dtype=X.dtype)

        # First center: uniform random point.
        centers[0] = X[rng.integers(n)]
        # Track squared distance from each point to the nearest chosen center.
        closest_sq = _pairwise_sq_dists(X, centers[:1]).ravel()

        for c in range(1, k):
            total = closest_sq.sum()
            if total == 0:  # all points already coincide with a center
                centers[c] = X[rng.integers(n)]
            else:
                probs = closest_sq / total
                centers[c] = X[rng.choice(n, p=probs)]
            # Update nearest-center distances with the new center.
            new_sq = _pairwise_sq_dists(X, centers[c : c + 1]).ravel()
            closest_sq = np.minimum(closest_sq, new_sq)
        return centers

    # --- one Lloyd run --------------------------------------------------- #
    def _single_run(self, X: np.ndarray, rng: np.random.Generator) -> KMeansResult:
        centers = self._kmeanspp_init(X, rng)
        labels = np.zeros(X.shape[0], dtype=np.int64)

        for it in range(1, self.max_iter + 1):
            # Assignment step.
            d2 = _pairwise_sq_dists(X, centers)
            new_labels = d2.argmin(axis=1)

            # Update step: each center becomes the mean of its members.
            new_centers = centers.copy()
            for c in range(self.n_clusters):
                members = X[new_labels == c]
                if len(members) == 0:
                    # Empty cluster: re-seed it on the point that is currently
                    # worst-served (farthest from its assigned center). This
                    # keeps all k cells alive, which the IVF index relies on.
                    worst = d2[np.arange(len(X)), new_labels].argmax()
                    new_centers[c] = X[worst]
                else:
                    new_centers[c] = members.mean(axis=0)

            shift = float(np.sum((new_centers - centers) ** 2))
            centers = new_centers
            labels = new_labels
            if shift <= self.tol:
                break

        d2 = _pairwise_sq_dists(X, centers)
        labels = d2.argmin(axis=1)
        inertia = float(d2[np.arange(len(X)), labels].sum())
        return KMeansResult(centers=centers, labels=labels, inertia=inertia, n_iter=it)

    # --- public API ------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self.n_clusters > len(X):
            raise ValueError(
                f"n_clusters={self.n_clusters} exceeds n_samples={len(X)}"
            )
        base_seed = np.random.SeedSequence(self.random_state)
        best: KMeansResult | None = None
        for run, child in enumerate(base_seed.spawn(self.n_init)):
            rng = np.random.default_rng(child)
            result = self._single_run(X, rng)
            if best is None or result.inertia < best.inertia:
                best = result
        assert best is not None
        self.centers_ = best.centers
        self.labels_ = best.labels
        self.inertia_ = best.inertia
        self.n_iter_ = best.n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of ``X`` to its nearest center."""
        if self.centers_ is None:
            raise RuntimeError("KMeans must be fit before predict")
        X = np.ascontiguousarray(X, dtype=np.float32)
        return _pairwise_sq_dists(X, self.centers_).argmin(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
