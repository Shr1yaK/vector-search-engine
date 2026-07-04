"""Metadata-filtered search: results must stay inside the allowed set, and the
exact (brute-force) filter must match brute force run on the subset alone."""

import numpy as np
import pytest

from src.brute_force import BruteForceIndex
from src.hnsw_index import HNSWIndex
from src.ivf_index import IVFIndex
from src.vectors import as_mask


@pytest.fixture(scope="module")
def allowed_mask(uniform_data):
    # An arbitrary but reproducible ~30% subset of rows.
    rng = np.random.default_rng(5)
    return rng.random(len(uniform_data)) < 0.3


def test_as_mask_from_ids_and_bool():
    m = as_mask([0, 2, 4], 5)
    assert list(m) == [True, False, True, False, True]
    assert as_mask(m, 5) is not None and as_mask(m, 5).sum() == 3
    with pytest.raises(ValueError):
        as_mask(np.array([True, False]), 5)   # wrong-length bool mask


def test_brute_force_filter_matches_subset(uniform_data, query_set, allowed_mask):
    bf = BruteForceIndex(uniform_data)
    ids_subset = np.where(allowed_mask)[0]
    sub = BruteForceIndex(uniform_data[ids_subset])  # index built on subset only
    for q in query_set:
        f_ids, _ = bf.search(q, k=10, allowed=allowed_mask)
        # exact filter == searching an index of just the allowed rows
        s_ids, _ = sub.search(q, k=10)
        assert list(f_ids) == list(ids_subset[s_ids])


def test_all_indexes_respect_the_filter(uniform_data, query_set, allowed_mask):
    bf = BruteForceIndex(uniform_data)
    ivf = IVFIndex(nlist=40, random_state=0).build(uniform_data)
    hnsw = HNSWIndex(M=16, ef_construction=100, random_state=0).build(uniform_data)
    for q in query_set[:20]:
        for ids in (
            bf.search(q, 10, allowed=allowed_mask)[0],
            ivf.search(q, 10, nprobe=16, allowed=allowed_mask)[0],
            hnsw.search(q, 10, ef_search=64, allowed=allowed_mask)[0],
        ):
            assert allowed_mask[ids].all()   # never returns a disallowed row


def test_approx_filter_recall_reasonable(uniform_data, query_set, allowed_mask):
    """Filtered HNSW/IVF should still recover most of the exact filtered top-k."""
    bf = BruteForceIndex(uniform_data)
    hnsw = HNSWIndex(M=16, ef_construction=200, random_state=0).build(uniform_data)
    hits = tot = 0
    for q in query_set:
        truth = set(bf.search(q, 10, allowed=allowed_mask)[0].tolist())
        got = set(hnsw.search(q, 10, ef_search=100, allowed=allowed_mask)[0].tolist())
        hits += len(truth & got)
        tot += len(truth)
    assert hits / tot > 0.85


def test_empty_filter_returns_nothing(uniform_data):
    bf = BruteForceIndex(uniform_data)
    ids, dists = bf.search(uniform_data[0], k=10, allowed=np.zeros(len(uniform_data), bool))
    assert len(ids) == 0 and len(dists) == 0
