import numpy as np

from src.benchmark import BenchResult, recall_at_k, results_to_markdown, run_benchmark


def test_recall_at_k_exact_and_partial():
    truth = np.array([1, 2, 3, 4])
    assert recall_at_k(np.array([1, 2, 3, 4]), truth) == 1.0
    assert recall_at_k(np.array([1, 2, 9, 9]), truth) == 0.5
    assert recall_at_k(np.array([9, 9]), truth) == 0.0


def test_recall_at_k_empty_truth_is_one():
    assert recall_at_k(np.array([1]), np.array([])) == 1.0


def test_run_benchmark_smoke(uniform_data):
    queries = uniform_data[:20]
    results = run_benchmark(
        uniform_data,
        queries,
        k=10,
        ivf_configs=[{"nlist": 30, "nprobe": 8}],
        hnsw_configs=[{"M": 8, "ef_construction": 50, "ef_search": 32}],
        verbose=False,
    )
    kinds = {r.index for r in results}
    assert kinds == {"brute_force", "ivf", "hnsw"}
    bf = next(r for r in results if r.index == "brute_force")
    assert bf.recall_at_k == 1.0                 # oracle is exact by definition
    assert all(0.0 <= r.recall_at_k <= 1.0 for r in results)
    assert all(r.qps > 0 for r in results)


def test_markdown_table_renders():
    r = BenchResult(
        index="ivf", params={"nprobe": 8}, n=100, d=8, k=10,
        build_time_s=0.1, mean_latency_ms=0.2, p95_latency_ms=0.3,
        qps=5000, recall_at_k=0.95, memory_mb=1.2,
    )
    md = results_to_markdown([r])
    assert "recall@k" in md and "nprobe=8" in md and "0.950" in md
