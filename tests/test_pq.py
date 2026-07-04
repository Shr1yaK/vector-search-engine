import numpy as np
import pytest

from src.pq import ProductQuantizer, _subspace_bounds


def test_subspace_bounds_cover_all_dims_evenly():
    b = _subspace_bounds(9, 2)          # 9 doesn't divide by 2
    assert b == [(0, 5), (5, 9)]        # sizes 5 and 4, contiguous, cover 0..9
    b = _subspace_bounds(8, 4)
    assert b == [(0, 2), (2, 4), (4, 6), (6, 8)]
    # bounds always tile [0, d)
    assert _subspace_bounds(128, 16)[-1][1] == 128


def test_encode_shape_and_dtype(blobs):
    X, _ = blobs
    pq = ProductQuantizer(m=4, ksub=16, random_state=0).fit(X)
    codes = pq.encode(X)
    assert codes.shape == (len(X), 4)
    assert codes.dtype == np.uint8
    assert codes.max() < 16


def test_decode_beats_random_reconstruction(blobs):
    X, _ = blobs
    pq = ProductQuantizer(m=4, ksub=32, random_state=0).fit(X)
    recon = pq.decode(pq.encode(X))
    pq_err = np.mean(np.sum((X - recon) ** 2, axis=1))
    # baseline: reconstruct every vector with the global mean
    base_err = np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1))
    assert pq_err < 0.25 * base_err     # quantization captures most structure


def test_adc_approximates_true_squared_distance(blobs):
    X, _ = blobs
    pq = ProductQuantizer(m=4, ksub=32, random_state=0).fit(X)
    codes = pq.encode(X)
    rng = np.random.default_rng(0)
    q = X[rng.integers(len(X))]
    table = pq.distance_tables(q)
    approx = pq.adc_distances(codes, table)
    true = np.sum((X - q) ** 2, axis=1)
    # ADC estimates should correlate strongly with the true distances
    corr = np.corrcoef(approx, true)[0, 1]
    assert corr > 0.95


def test_memory_is_compressive():
    pq = ProductQuantizer(m=8, ksub=256, random_state=0)
    pq.fit(np.random.default_rng(0).random((2000, 64)).astype(np.float32))
    # codes are 8 bytes/vector vs 64*4 = 256 bytes raw
    assert pq.memory_bytes(2000) < 2000 * 64 * 4


def test_ksub_over_256_rejected():
    with pytest.raises(ValueError):
        ProductQuantizer(m=4, ksub=512)
