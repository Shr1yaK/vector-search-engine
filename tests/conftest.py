"""Shared fixtures. Tests run on synthetic data so they're fast and deterministic
and don't depend on the (git-ignored) Spotify CSV being present."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def blobs():
    """A reproducible set of well-separated Gaussian clusters in R^8.

    Well-separated blobs give unambiguous nearest neighbors, so approximate
    indexes should recover them at high recall — a meaningful correctness bar.
    """
    rng = np.random.default_rng(42)
    d = 8
    centers = rng.uniform(-10, 10, size=(12, d))
    pts, labels = [], []
    for c, ctr in enumerate(centers):
        pts.append(ctr + rng.normal(scale=0.4, size=(250, d)))
        labels += [c] * 250
    X = np.vstack(pts).astype(np.float32)
    perm = rng.permutation(len(X))
    return X[perm], np.array(labels)[perm]


@pytest.fixture(scope="session")
def uniform_data():
    """Unstructured uniform points — a harder, structure-free recall test."""
    rng = np.random.default_rng(7)
    return rng.random((2000, 16)).astype(np.float32)


@pytest.fixture(scope="session")
def query_set(uniform_data):
    rng = np.random.default_rng(99)
    return uniform_data[rng.integers(len(uniform_data), size=50)]
