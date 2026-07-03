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
    """
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to load the Spotify CSV")

    df = pd.read_csv(csv_path)
    if dropna:
        df = df.dropna(subset=list(feature_columns))
    if limit is not None:
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


def get_metric(name: str):
    """Look up a distance function by name (``l2`` | ``cosine`` | ...)."""
    try:
        return _METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric {name!r}; choose from {sorted(_METRICS)}")


def pairwise_to_point(query: np.ndarray, matrix: np.ndarray, metric: str = "l2") -> np.ndarray:
    """Distances from one query to all rows, dispatched by metric name."""
    return get_metric(metric)(query, matrix)
