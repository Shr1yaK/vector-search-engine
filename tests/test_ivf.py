import numpy as np

from src.brute_force import BruteForceIndex
from src.ivf_index import IVFIndex


def _recall(approx, truth):
    return len(set(approx.tolist()) & set(truth.tolist())) / len(truth)


def test_full_probe_equals_brute_force(uniform_data, query_set):
    """Probing every cell must reproduce the exact brute-force result set."""
    bf = BruteForceIndex(uniform_data)
    ivf = IVFIndex(nlist=32, random_state=0).build(uniform_data)
    for q in query_set:
        t_ids, _ = bf.search(q, k=10)
        a_ids, _ = ivf.search(q, k=10, nprobe=ivf.nlist)  # scan all cells
        assert _recall(a_ids, t_ids) == 1.0


def test_recall_increases_with_nprobe(uniform_data, query_set):
    bf = BruteForceIndex(uniform_data)
    ivf = IVFIndex(nlist=45, random_state=0).build(uniform_data)
    truth = [bf.search(q, 10)[0] for q in query_set]
    recalls = []
    for nprobe in (1, 4, 16):
        r = np.mean([
            _recall(ivf.search(q, 10, nprobe=nprobe)[0], t)
            for q, t in zip(query_set, truth)
        ])
        recalls.append(r)
    # Monotonic non-decreasing recall as we probe more cells.
    assert recalls[0] <= recalls[1] <= recalls[2]
    assert recalls[-1] > 0.9


def test_build_stats_populated(uniform_data):
    ivf = IVFIndex(nlist=20, random_state=0).build(uniform_data)
    assert ivf.build_stats_["nlist"] == 20
    assert ivf.build_stats_["empty_cells"] == 0
    assert sum(len(l) for l in ivf.inverted_lists_) == len(uniform_data)
