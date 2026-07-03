import numpy as np
import pytest

from src.vectors import (
    Normalizer,
    cosine_distance,
    from_matrix,
    get_metric,
    l2_distance,
    l2_distance_sq,
)


def test_normalizer_zscore_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6)) * 10 + 3
    norm = Normalizer.fit(X, method="zscore")
    Z = norm.transform(X)
    # z-scored columns have ~0 mean and ~1 std.
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-4)
    # inverse_transform recovers the original values.
    assert np.allclose(norm.inverse_transform(Z), X, atol=1e-3)


def test_normalizer_minmax_range():
    rng = np.random.default_rng(1)
    X = rng.uniform(-5, 20, size=(300, 4))
    Z = Normalizer.fit(X, method="minmax").transform(X)
    assert Z.min() >= -1e-6 and Z.max() <= 1 + 1e-6


def test_normalizer_constant_column_no_nan():
    X = np.ones((10, 3))  # zero variance -> must not divide by zero
    Z = Normalizer.fit(X).transform(X)
    assert not np.isnan(Z).any()


def test_l2_distance_known_values():
    q = np.array([0, 0, 0], dtype=np.float32)
    M = np.array([[3, 4, 0], [0, 0, 0], [1, 2, 2]], dtype=np.float32)
    assert np.allclose(l2_distance(q, M), [5.0, 0.0, 3.0])
    assert np.allclose(l2_distance_sq(q, M), [25.0, 0.0, 9.0])


def test_cosine_distance_bounds_and_identity():
    q = np.array([1.0, 0.0])
    M = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    d = cosine_distance(q, M)
    assert np.isclose(d[0], 0.0)      # identical direction
    assert np.isclose(d[1], 1.0)      # orthogonal
    assert np.isclose(d[2], 2.0)      # opposite


def test_metric_dispatch_and_error():
    assert get_metric("l2") is l2_distance
    assert get_metric("euclidean") is l2_distance
    with pytest.raises(ValueError):
        get_metric("manhattan")


def test_from_matrix_builds_dataset():
    X = np.arange(20, dtype=float).reshape(5, 4)
    ds = from_matrix(X)
    assert ds.n == 5 and ds.d == 4
    assert ds.vectors.dtype == np.float32
