"""High-dimensional benchmark — the regime where HNSW earns its keep.

On the 9-D Spotify audio features, IVF wins: at low dimensionality a linear scan
of a few cells is cheap and HNSW's graph-traversal overhead doesn't pay off.
But production vector search runs on *embeddings* — hundreds of dimensions —
where brute force and cell-scans get expensive and HNSW's logarithmic-ish hop
count dominates. This script reproduces that crossover on synthetic clustered
data so the benchmark tells the whole story, not just the easy half.

    python benchmarks/bench_highdim.py --n 20000 --d 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import run_benchmark, save_results, results_to_markdown  # noqa: E402


def make_clustered(n: int, d: int, n_clusters: int, seed: int) -> np.ndarray:
    """Synthetic clustered vectors — realistic-ish structure for ANN to exploit."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(n_clusters, d))
    assign = rng.integers(n_clusters, size=n)
    return (centers[assign] + rng.normal(scale=1.0, size=(n, d))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--clusters", type=int, default=100)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()

    X = make_clustered(args.n, args.d, args.clusters, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    queries = X[rng.integers(args.n, size=args.queries)] + rng.normal(
        scale=0.5, size=(args.queries, args.d)
    ).astype(np.float32)

    results = run_benchmark(X, queries, k=args.k, seed=args.seed)
    tag = f"highdim_d{args.d}"
    save_results(results, args.out, tag=tag)
    print("\n" + results_to_markdown(results))
    print(f"\nsaved -> {args.out}/benchmark_{tag}.{{json,md}}")


if __name__ == "__main__":
    main()
