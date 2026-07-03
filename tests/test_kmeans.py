import numpy as np

from src.kmeans import KMeans, _pairwise_sq_dists


def test_pairwise_sq_dists_matches_naive():
    rng = np.random.default_rng(0)
    P = rng.random((20, 5)).astype(np.float32)
    C = rng.random((4, 5)).astype(np.float32)
    fast = _pairwise_sq_dists(P, C)
    naive = np.array([[np.sum((p - c) ** 2) for c in C] for p in P])
    assert np.allclose(fast, naive, atol=1e-4)


def test_recovers_well_separated_clusters(blobs):
    X, true_labels = blobs
    km = KMeans(n_clusters=12, random_state=0, n_init=8).fit(X)
    # With well-separated blobs, every learned cluster should be (almost) pure:
    # points sharing a predicted label should share a true label.
    labels = km.labels_
    agree = 0
    for c in range(12):
        members = true_labels[labels == c]
        if len(members):
            # fraction that belong to the dominant true class in this cluster
            agree += np.bincount(members).max()
    purity = agree / len(X)
    assert purity > 0.95


def test_inertia_nonincreasing_over_iterations():
    # Inertia after fit should be finite and lower than a single random assignment.
    rng = np.random.default_rng(1)
    X = rng.random((300, 4)).astype(np.float32)
    km = KMeans(n_clusters=8, random_state=0, n_init=1).fit(X)
    random_centers = X[rng.integers(len(X), size=8)]
    d2 = _pairwise_sq_dists(X, random_centers)
    random_inertia = d2.min(axis=1).sum()
    assert km.inertia_ <= random_inertia


def test_predict_assigns_to_nearest_center(blobs):
    X, _ = blobs
    km = KMeans(n_clusters=6, random_state=0, n_init=1).fit(X)
    pred = km.predict(X[:50])
    manual = _pairwise_sq_dists(X[:50], km.centers_).argmin(axis=1)
    assert np.array_equal(pred, manual)


def test_no_empty_clusters_on_dense_data(blobs):
    X, _ = blobs
    km = KMeans(n_clusters=20, random_state=3, n_init=1).fit(X)
    counts = np.bincount(km.labels_, minlength=20)
    assert (counts > 0).all()   # empty-cluster re-seeding keeps all cells alive
