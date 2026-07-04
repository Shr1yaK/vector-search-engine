"""Smoke tests for the CLI's index-building dispatch (no dataset needed)."""

import numpy as np
import pytest

from src.search import _build


@pytest.mark.parametrize("index", ["brute", "ivf", "ivfpq", "hnsw"])
def test_build_returns_working_search(index, uniform_data):
    search = _build(index, uniform_data)
    ids, dists = search(uniform_data[0], 5, None)
    assert len(ids) == 5
    assert ids[0] == 0 or index in {"ivf", "ivfpq"}  # exact/hnsw find self first


def test_build_search_accepts_filter(uniform_data):
    search = _build("hnsw", uniform_data)
    allowed = np.zeros(len(uniform_data), dtype=bool)
    allowed[:100] = True
    ids, _ = search(uniform_data[0], 5, allowed)
    assert allowed[ids].all()
