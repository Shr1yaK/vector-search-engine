import numpy as np

from src.brute_force import BruteForceIndex


def test_returns_sorted_and_self_is_nearest(uniform_data):
    bf = BruteForceIndex(uniform_data)
    ids, dists = bf.search(uniform_data[10], k=5)
    assert ids[0] == 10                 # a point is its own nearest neighbor
    assert np.isclose(dists[0], 0.0)
    assert np.all(np.diff(dists) >= 0)  # distances ascending


def test_k_capped_to_n():
    X = np.random.default_rng(0).random((4, 3)).astype(np.float32)
    bf = BruteForceIndex(X)
    ids, _ = bf.search(X[0], k=99)
    assert len(ids) == 4                # cannot return more than n


def test_known_answer():
    X = np.array([[0, 0], [1, 0], [5, 5], [0, 2]], dtype=np.float32)
    bf = BruteForceIndex(X)
    ids, _ = bf.search(np.array([0, 0], dtype=np.float32), k=3)
    assert list(ids) == [0, 1, 3]       # by increasing distance from origin


def test_batch_matches_single(uniform_data, query_set):
    bf = BruteForceIndex(uniform_data)
    bidx, _ = bf.search_batch(query_set, k=7)
    for i, q in enumerate(query_set):
        sidx, _ = bf.search(q, k=7)
        assert list(sidx) == list(bidx[i])
