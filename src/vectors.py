"""Data loading, feature normalization, and distance functions.

This is the foundation layer. Everything else in the project operates on the
``float32`` matrix produced here, so the choices made in this file — which audio
features become dimensions, how they are scaled, which metric defines
"nearest" — propagate through every index and every benchmark.

Design notes
------------
* Spotify's raw audio features live on wildly different scales: ``tempo`` runs
  0–250 BPM, ``loudness`` runs roughly -60–0 dB, while ``valence`` /
  ``energy`` / ``danceability`` are already in [0, 1]. Feeding those raw into a
  Euclidean metric would let tempo dominate the distance entirely. We therefore
  fit a normalizer on the corpus and reuse *the exact same* transform on every
  query vector (see :class:`Normalizer`).
* We keep the fitted statistics around so the semantic app can turn a
  human-scale target ("high energy, low acousticness") into the model's
  internal normalized space and back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:  # pandas is only needed for CSV loading, not for the index math.
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a hard dependency in practice
    pd = None

# Optional C++ acceleration (stretch goal). Built via ``python cpp/build.py``.
# When present, the hot-path distance kernels run in compiled code; when
# absent, everything transparently falls back to the numpy implementations
# below, so the library is fully functional either way.
try:
    import vecsearch_native as _native
    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - extension is optional
    _native = None
    NATIVE_AVAILABLE = False


def native_available() -> bool:
    """True if the compiled C++ distance kernels are importable."""
    return NATIVE_AVAILABLE


# The audio features we treat as searchable dimensions. Ordering is fixed and
# load-bearing: a query vector must present its values in this same order.
FEATURE_COLUMNS: tuple[str, ...] = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
)

# Human-readable descriptions, used by the app and the mood translator prompt.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "danceability": "how suitable a track is for dancing (0 = least, 1 = most)",
    "energy": "perceptual intensity and activity (0 = calm, 1 = energetic)",
    "loudness": "overall loudness in decibels (roughly -60 to 0)",
    "speechiness": "presence of spoken words (0 = music, 1 = talk)",
    "acousticness": "confidence the track is acoustic (0 = electric, 1 = acoustic)",
    "instrumentalness": "likelihood the track has no vocals (0 = vocals, 1 = instrumental)",
    "liveness": "presence of a live audience (0 = studio, 1 = live)",
    "valence": "musical positivity (0 = sad/angry, 1 = happy/euphoric)",
    "tempo": "estimated beats per minute",
}


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
@dataclass
class Normalizer:
    """A fitted feature scaler that can be applied to corpus *and* queries.

    Two schemes are supported:

    * ``"zscore"`` (default): subtract mean, divide by std. Best pairing with a
      Euclidean metric because it equalizes each feature's spread.
    * ``"minmax"``: scale each feature to [0, 1]. Handy when you want query
      values expressed on the raw feature scale to be interpretable.
    """

    method: str
    center: np.ndarray  # mean (zscore) or min (minmax), shape (d,)
    scale: np.ndarray   # std  (zscore) or (max-min) (minmax), shape (d,)

    @classmethod
    def fit(cls, matrix: np.ndarray, method: str = "zscore") -> "Normalizer":
        matrix = np.asarray(matrix, dtype=np.float64)
        if method == "zscore":
            center = matrix.mean(axis=0)
            scale = matrix.std(axis=0)
        elif method == "minmax":
            center = matrix.min(axis=0)
            scale = matrix.max(axis=0) - center
        else:
            raise ValueError(f"unknown normalization method: {method!r}")
        # Guard against constant columns (std or range == 0) to avoid div-by-0.
        scale = np.where(scale == 0, 1.0, scale)
        return cls(method=method, center=center, scale=scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        return ((matrix - self.center) / self.scale).astype(np.float32)

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        return (matrix * self.scale + self.center).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """Holds the corpus in both raw and normalized form plus track metadata."""

    vectors: np.ndarray            # normalized, shape (n, d), float32
    raw: np.ndarray                # un-normalized features, shape (n, d)
    feature_names: tuple[str, ...]
    normalizer: Normalizer
    metadata: "pd.DataFrame | None" = None  # track_name, artists, genre, ...

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @property
    def d(self) -> int:
        return self.vectors.shape[1]

    def describe(self, indices: Iterable[int]) -> "pd.DataFrame":
        """Return metadata rows for the given row indices (for pretty results)."""
        if self.metadata is None:
            raise ValueError("no metadata attached to this dataset")
        return self.metadata.iloc[list(indices)]


def load_spotify(
    csv_path: str,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    method: str = "zscore",
    limit: int | None = None,
    dropna: bool = True,
    sample_seed: int | None = None,
) -> Dataset:
    """Load the Spotify Tracks CSV into a normalized :class:`Dataset`.

    Parameters
    ----------
    csv_path:
        Path to ``spotify_tracks.csv`` (see ``README`` for how to fetch it).
    feature_columns:
        Which columns become vector dimensions.
    method:
        Normalization scheme passed to :class:`Normalizer`.
    limit:
        Optionally cap the number of rows (useful for quick experiments).
    dropna:
        Drop rows with missing feature values (a handful exist in the raw data).
    sample_seed:
        If set (and ``limit`` is set), take a *random* seeded sample of ``limit``
        rows instead of the first ``limit``. This matters: the raw CSV is sorted
        by ``track_genre``, so taking the head yields only the alphabetically
        earliest genres (acoustic, afrobeat, …). Sampling gives a corpus that
        spans all 114 genres — what the app wants. Benchmarks leave this ``None``
        so their corpus stays byte-identical across runs.
    """
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to load the Spotify CSV")

    df = pd.read_csv(csv_path)
    if dropna:
        df = df.dropna(subset=list(feature_columns))
    if limit is not None:
        if sample_seed is not None and limit < len(df):
            df = df.sample(n=limit, random_state=sample_seed)
        else:
            df = df.iloc[:limit]
    df = df.reset_index(drop=True)

    raw = df[list(feature_columns)].to_numpy(dtype=np.float64)
    normalizer = Normalizer.fit(raw, method=method)
    vectors = normalizer.transform(raw)

    meta_cols = [
        c
        for c in ("track_id", "track_name", "artists", "album_name", "track_genre", "popularity")
        if c in df.columns
    ]
    metadata = df[meta_cols].copy() if meta_cols else None

    return Dataset(
        vectors=vectors,
        raw=raw.astype(np.float32),
        feature_names=tuple(feature_columns),
        normalizer=normalizer,
        metadata=metadata,
    )


def from_matrix(matrix: np.ndarray, method: str = "zscore") -> Dataset:
    """Build a :class:`Dataset` from a bare matrix (used heavily in tests)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    normalizer = Normalizer.fit(matrix, method=method)
    return Dataset(
        vectors=normalizer.transform(matrix),
        raw=matrix.astype(np.float32),
        feature_names=tuple(f"f{i}" for i in range(matrix.shape[1])),
        normalizer=normalizer,
    )


# --------------------------------------------------------------------------- #
# Distance functions
# --------------------------------------------------------------------------- #
# All index implementations route their distance math through these helpers so
# that swapping metrics (or later swapping in the C++ kernel) touches one place.

def l2_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Euclidean distance from ``query`` (d,) to every row of ``matrix`` (n, d).

    Uses the expansion ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, which lets us lean
    on a single optimized matrix-vector product instead of materializing the
    (n, d) difference tensor.
    """
    query = np.asarray(query, dtype=np.float32)
    diff = matrix - query
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def l2_distance_sq(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance (skips the sqrt; order-preserving for k-NN)."""
    query = np.asarray(query, dtype=np.float32)
    diff = matrix - query
    return np.einsum("ij,ij->i", diff, diff)


def cosine_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance = 1 - cosine similarity, in [0, 2]."""
    query = np.asarray(query, dtype=np.float32)
    qn = np.linalg.norm(query)
    mn = np.linalg.norm(matrix, axis=1)
    denom = np.where((qn * mn) == 0, 1.0, qn * mn)
    sim = (matrix @ query) / denom
    return 1.0 - sim


_METRICS = {
    "l2": l2_distance,
    "euclidean": l2_distance,
    "sqeuclidean": l2_distance_sq,
    "cosine": cosine_distance,
}


# --------------------------------------------------------------------------- #
# Native-accelerated kernels (used by the indexes on their hot paths)
# --------------------------------------------------------------------------- #
def l2_sq_point(query: np.ndarray, vec: np.ndarray, use_native: bool = True) -> float:
    """Squared L2 between two (d,) vectors — the HNSW graph-traversal hot path.

    Dispatches to the C++ kernel when it is built and ``use_native`` is set,
    otherwise uses numpy. Kept as a single chokepoint so the whole graph search
    speeds up by flipping one flag.
    """
    if use_native and NATIVE_AVAILABLE:
        return _native.l2_sq_point(
            np.ascontiguousarray(query, dtype=np.float32),
            np.ascontiguousarray(vec, dtype=np.float32),
        )
    diff = np.asarray(query, dtype=np.float32) - np.asarray(vec, dtype=np.float32)
    return float(diff @ diff)


def l2_batch_native(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Euclidean distances (native if available) for the brute-force/IVF scan."""
    if NATIVE_AVAILABLE:
        return _native.l2_batch(
            np.ascontiguousarray(query, dtype=np.float32),
            np.ascontiguousarray(matrix, dtype=np.float32),
        )
    return l2_distance(query, matrix)


def get_metric(name: str):
    """Look up a distance function by name (``l2`` | ``cosine`` | ...)."""
    try:
        return _METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric {name!r}; choose from {sorted(_METRICS)}")


def pairwise_to_point(query: np.ndarray, matrix: np.ndarray, metric: str = "l2") -> np.ndarray:
    """Distances from one query to all rows, dispatched by metric name."""
    return get_metric(metric)(query, matrix)


def exact_subset_search(
    query: np.ndarray,
    vectors: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
    dist_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k over an explicit subset of rows.

    Used by every approximate index as its **pre-filtering** strategy: when a
    metadata filter is highly selective, scanning the handful of matching rows
    exactly is both cheaper *and* more accurate than post-filtering an
    approximate traversal (which can starve and return fewer than k results).
    """
    if candidate_ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    dists = dist_fn(query, vectors[candidate_ids])
    kk = min(k, candidate_ids.size)
    part = np.argpartition(dists, kk - 1)[:kk]
    order = np.argsort(dists[part])
    top = part[order]
    return candidate_ids[top], dists[top].astype(np.float32)


def as_mask(allowed, n: int) -> np.ndarray:
    """Normalize a metadata filter into a boolean mask of length ``n``.

    ``allowed`` may be a boolean mask (returned as-is) or any iterable of row
    ids (turned into a mask). This lets every index accept the same filter
    argument — the mechanism behind "search only within these genres".
    """
    arr = np.asarray(list(allowed) if not isinstance(allowed, np.ndarray) else allowed)
    if arr.dtype == bool:
        if arr.shape[0] != n:
            raise ValueError(f"boolean mask length {arr.shape[0]} != n {n}")
        return arr
    mask = np.zeros(n, dtype=bool)
    if arr.size:
        mask[arr.astype(np.int64)] = True
    return mask
