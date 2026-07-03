"""A saved-then-loaded index must answer queries identically to the original."""

import numpy as np
import pytest

from src.hnsw_index import HNSWIndex
from src.ivf_index import IVFIndex


def _same_results(a, b, queries, k, **kw):
    for q in queries:
        ia, da = a.search(q, k, **kw)
        ib, db = b.search(q, k, **kw)
        assert list(ia) == list(ib)
        assert np.allclose(da, db)


def test_ivf_roundtrip(tmp_path, uniform_data, query_set):
    ivf = IVFIndex(nlist=30, nprobe=8, random_state=0).build(uniform_data)
    ivf.save(tmp_path / "ivf")
    loaded = IVFIndex.load(tmp_path / "ivf")

    assert loaded.nlist == ivf.nlist and loaded.metric == ivf.metric
    assert loaded.n == ivf.n
    # inverted lists partition all points, same as before
    assert sum(len(l) for l in loaded.inverted_lists_) == uniform_data.shape[0]
    _same_results(ivf, loaded, query_set, k=10, nprobe=8)


def test_hnsw_roundtrip(tmp_path, uniform_data, query_set):
    h = HNSWIndex(M=12, ef_construction=100, random_state=0).build(uniform_data)
    h.save(tmp_path / "hnsw")
    loaded = HNSWIndex.load(tmp_path / "hnsw")

    assert loaded.entry_point_ == h.entry_point_
    assert loaded.max_level_ == h.max_level_
    assert len(loaded.graph_) == len(h.graph_)
    _same_results(h, loaded, query_set, k=10, ef_search=64)


def test_load_rejects_wrong_type(tmp_path, uniform_data):
    IVFIndex(nlist=10, random_state=0).build(uniform_data).save(tmp_path / "x")
    with pytest.raises(ValueError):
        HNSWIndex.load(tmp_path / "x")   # it's an IVF index on disk


def test_save_unbuilt_raises(tmp_path):
    with pytest.raises(RuntimeError):
        IVFIndex().save(tmp_path / "nope")
    with pytest.raises(RuntimeError):
        HNSWIndex().save(tmp_path / "nope")
