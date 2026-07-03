import numpy as np
import pytest

from src.brute_force import BruteForceIndex
from src.hnsw_index import HNSWIndex


def _recall(approx, truth):
    return len(set(approx.tolist()) & set(truth.tolist())) / len(truth)


@pytest.fixture(scope="module")
def built(uniform_data):
    return HNSWIndex(M=16, ef_construction=200, random_state=0).build(uniform_data)


def test_high_recall_vs_brute_force(built, uniform_data, query_set):
    bf = BruteForceIndex(uniform_data)
    recalls = []
    for q in query_set:
        t_ids, _ = bf.search(q, k=10)
        a_ids, _ = built.search(q, k=10, ef_search=100)
        recalls.append(_recall(a_ids, t_ids))
    assert np.mean(recalls) > 0.90


def test_returns_exactly_k(built):
    ids, dists = built.search(built.vectors_[0], k=8, ef_search=64)
    assert len(ids) == 8 and len(dists) == 8
    assert np.all(np.diff(dists) >= 0)  # sorted ascending


def test_self_is_nearest(built):
    ids, dists = built.search(built.vectors_[123], k=5, ef_search=64)
    assert ids[0] == 123
    assert np.isclose(dists[0], 0.0, atol=1e-4)


def test_graph_has_expected_structure(built):
    stats = built.build_stats_
    assert stats["levels"] >= 1
    assert built.entry_point_ is not None
    # layer-0 average degree should be within the configured cap (2*M).
    assert stats["avg_degree_layer0"] <= built.M_max0 + 1e-6


def test_ef_at_least_k_enforced(built):
    # Even if a caller passes ef_search < k, we still return k results.
    ids, _ = built.search(built.vectors_[0], k=10, ef_search=2)
    assert len(ids) == 10


@pytest.mark.skipif(
    not __import__("src.vectors", fromlist=["native_available"]).native_available(),
    reason="C++ extension not built",
)
def test_native_matches_numpy_recall(uniform_data, query_set):
    """The C++ hot-path must not change which neighbors are returned."""
    bf = BruteForceIndex(uniform_data)
    h_np = HNSWIndex(M=16, ef_construction=100, random_state=0, use_native=False).build(uniform_data)
    h_c = HNSWIndex(M=16, ef_construction=100, random_state=0, use_native=True).build(uniform_data)
    for q in query_set[:20]:
        t = bf.search(q, 10)[0]
        r_np = _recall(h_np.search(q, 10, ef_search=64)[0], t)
        r_c = _recall(h_c.search(q, 10, ef_search=64)[0], t)
        assert abs(r_np - r_c) < 1e-9
