import numpy as np
import pytest

from src.brute_force import BruteForceIndex
from src.ivfpq_index import IVFPQIndex


def _recall(approx, truth):
    return len(set(approx.tolist()) & set(truth.tolist())) / len(truth)


def test_compression_ratio_and_memory(blobs):
    X, _ = blobs                        # d = 8
    ipq = IVFPQIndex(nlist=16, m=4, ksub=16, random_state=0).build(X)
    assert ipq.build_stats_["compression_ratio"] == (8 * 4) / 4
    assert ipq.memory_bytes() < X.nbytes    # the whole point: smaller than raw


def test_returns_valid_ids_and_k(blobs):
    X, _ = blobs
    ipq = IVFPQIndex(nlist=16, m=4, ksub=16, nprobe=8, random_state=0).build(X)
    ids, dists = ipq.search(X[0], k=10)
    assert len(ids) == 10
    assert ids.min() >= 0 and ids.max() < len(X)
    assert np.all(np.diff(dists) >= 0)


def test_rerank_recovers_recall(blobs):
    X, _ = blobs
    bf = BruteForceIndex(X)
    ipq = IVFPQIndex(nlist=16, m=4, ksub=32, nprobe=16, random_state=0).build(
        X, keep_vectors=True
    )
    rng = np.random.default_rng(1)
    qs = X[rng.integers(len(X), size=60)]
    plain = np.mean([_recall(ipq.search(q, 10)[0], bf.search(q, 10)[0]) for q in qs])
    reranked = np.mean(
        [_recall(ipq.search(q, 10, rerank=50)[0], bf.search(q, 10)[0]) for q in qs]
    )
    assert reranked >= plain            # exact rerank never hurts
    assert reranked > 0.85              # and recovers most of the lost recall


def test_respects_metadata_filter(blobs):
    X, labels = blobs
    ipq = IVFPQIndex(nlist=16, m=4, ksub=16, nprobe=16, random_state=0).build(X)
    allowed = labels == 3               # only points from one true cluster
    ids, _ = ipq.search(X[0], k=10, nprobe=16, allowed=allowed)
    assert allowed[ids].all()


def test_rerank_without_vectors_falls_back(blobs):
    X, _ = blobs
    ipq = IVFPQIndex(nlist=16, m=4, ksub=16, random_state=0).build(X)  # no keep_vectors
    # asking to rerank without stored vectors must not crash; returns PQ result
    ids, _ = ipq.search(X[0], k=5, rerank=50)
    assert len(ids) == 5


def test_unbuilt_search_raises():
    with pytest.raises(RuntimeError):
        IVFPQIndex().search(np.zeros(8, dtype=np.float32))
