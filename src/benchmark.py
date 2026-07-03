"""Benchmark harness: the methodology, not just "it works".

Approximate nearest-neighbor search is a multi-objective trade-off. An index is
never simply "better" — it is faster *at some recall*, or smaller *at some
latency*. This module measures the four axes that matter and reports them
side-by-side so the trade-offs are legible:

* **recall@k** — fraction of the true k nearest neighbors (from the brute-force
  oracle) that the approximate index actually returned. The accuracy axis.
* **latency** — mean and p95 per-query wall time, plus queries/sec (QPS).
* **build time** — one-off cost to construct the index.
* **memory** — estimated resident bytes of the index data structures.

Run ``python -m src.benchmark`` to regenerate ``benchmarks/results/*.{json,md}``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .brute_force import BruteForceIndex
from .hnsw_index import HNSWIndex
from .ivf_index import IVFIndex


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def recall_at_k(approx_ids: np.ndarray, truth_ids: np.ndarray) -> float:
    """Overlap fraction between an approximate result set and the true k-NN.

    Both inputs are id arrays for a single query. We compare as sets so ordering
    differences within the returned neighbors don't count against recall.
    """
    if len(truth_ids) == 0:
        return 1.0
    return len(set(approx_ids.tolist()) & set(truth_ids.tolist())) / len(truth_ids)


def _percentile_ms(times_s: list[float], p: float) -> float:
    return float(np.percentile(times_s, p) * 1000.0)


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class BenchResult:
    index: str                 # "brute_force" | "ivf" | "hnsw"
    params: dict               # config knobs for this run
    n: int
    d: int
    k: int
    build_time_s: float
    mean_latency_ms: float
    p95_latency_ms: float
    qps: float
    recall_at_k: float
    memory_mb: float
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# memory estimation
# --------------------------------------------------------------------------- #
def _ivf_memory_bytes(index: IVFIndex) -> int:
    v = index.vectors_.nbytes
    c = index.centroids_.nbytes
    lists = sum(arr.nbytes for arr in index.inverted_lists_)
    return v + c + lists


def _hnsw_memory_bytes(index: HNSWIndex) -> int:
    v = index.vectors_.nbytes
    # 8 bytes per directed edge (int64 id) + rough dict/list overhead per node.
    edges = sum(len(nbrs) for layer in index.graph_ for nbrs in layer.values())
    nodes = sum(len(layer) for layer in index.graph_)
    return v + edges * 8 + nodes * 56


# --------------------------------------------------------------------------- #
# per-index measurement
# --------------------------------------------------------------------------- #
def _time_queries(search_fn, queries: np.ndarray, truth: list[np.ndarray], k: int):
    """Run every query, timing each and scoring recall against the oracle."""
    latencies: list[float] = []
    recalls: list[float] = []
    for q, t_ids in zip(queries, truth):
        t0 = time.perf_counter()
        ids, _ = search_fn(q)
        latencies.append(time.perf_counter() - t0)
        recalls.append(recall_at_k(ids, t_ids))
    mean_lat = float(np.mean(latencies))
    return {
        "mean_latency_ms": mean_lat * 1000.0,
        "p95_latency_ms": _percentile_ms(latencies, 95),
        "qps": 1.0 / mean_lat if mean_lat > 0 else float("inf"),
        "recall_at_k": float(np.mean(recalls)),
    }


def run_benchmark(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    ivf_configs: list[dict] | None = None,
    hnsw_configs: list[dict] | None = None,
    metric: str = "l2",
    seed: int = 0,
    verbose: bool = True,
) -> list[BenchResult]:
    """Benchmark brute-force, IVF (nprobe sweep), and HNSW (ef sweep).

    ``ivf_configs`` / ``hnsw_configs`` are lists of parameter dicts. Sensible
    defaults sweep the recall/latency knob of each index.
    """
    n, d = vectors.shape
    results: list[BenchResult] = []

    def log(*a):
        if verbose:
            print(*a, flush=True)

    # --- ground truth (brute force) ------------------------------------- #
    log(f"[bench] n={n} d={d} queries={len(queries)} k={k} metric={metric}")
    t0 = time.perf_counter()
    bf = BruteForceIndex(vectors, metric=metric)
    bf_build = time.perf_counter() - t0
    truth = [bf.search(q, k)[0] for q in queries]

    bf_stats = _time_queries(lambda q: bf.search(q, k), queries, truth, k)
    results.append(
        BenchResult(
            index="brute_force", params={}, n=n, d=d, k=k,
            build_time_s=bf_build, memory_mb=vectors.nbytes / 1e6,
            **bf_stats,
        )
    )
    log(f"[bench] brute_force: recall={bf_stats['recall_at_k']:.3f} "
        f"lat={bf_stats['mean_latency_ms']:.3f}ms qps={bf_stats['qps']:.0f}")

    # --- IVF sweep ------------------------------------------------------ #
    ivf_configs = ivf_configs or [
        {"nlist": int(np.sqrt(n)) or 1, "nprobe": p} for p in (1, 4, 8, 16, 32)
    ]
    ivf_cache: dict[int, IVFIndex] = {}
    for cfg in ivf_configs:
        nlist = cfg.get("nlist", int(np.sqrt(n)) or 1)
        nprobe = cfg["nprobe"]
        if nlist not in ivf_cache:  # build once per nlist, reuse across nprobe
            t0 = time.perf_counter()
            ivf_cache[nlist] = IVFIndex(
                nlist=nlist, metric=metric, random_state=seed
            ).build(vectors)
            ivf_cache[nlist]._build_time = time.perf_counter() - t0  # type: ignore[attr-defined]
        ivf = ivf_cache[nlist]
        stats = _time_queries(lambda q: ivf.search(q, k, nprobe=nprobe), queries, truth, k)
        results.append(
            BenchResult(
                index="ivf", params={"nlist": nlist, "nprobe": nprobe},
                n=n, d=d, k=k, build_time_s=ivf._build_time,  # type: ignore[attr-defined]
                memory_mb=_ivf_memory_bytes(ivf) / 1e6, extra=ivf.build_stats_, **stats,
            )
        )
        log(f"[bench] ivf(nlist={nlist},nprobe={nprobe}): "
            f"recall={stats['recall_at_k']:.3f} lat={stats['mean_latency_ms']:.3f}ms "
            f"qps={stats['qps']:.0f} speedup={bf_stats['mean_latency_ms']/stats['mean_latency_ms']:.1f}x")

    # --- HNSW sweep ----------------------------------------------------- #
    hnsw_configs = hnsw_configs or [
        {"M": 16, "ef_construction": 200, "ef_search": ef} for ef in (10, 20, 50, 100)
    ]
    hnsw_cache: dict[tuple, HNSWIndex] = {}
    for cfg in hnsw_configs:
        M = cfg.get("M", 16)
        efc = cfg.get("ef_construction", 200)
        efs = cfg["ef_search"]
        key = (M, efc)
        if key not in hnsw_cache:  # build once per (M, ef_construction)
            hnsw_cache[key] = HNSWIndex(
                M=M, ef_construction=efc, metric=metric, random_state=seed
            ).build(vectors)
        h = hnsw_cache[key]
        stats = _time_queries(lambda q: h.search(q, k, ef_search=efs), queries, truth, k)
        results.append(
            BenchResult(
                index="hnsw", params={"M": M, "ef_construction": efc, "ef_search": efs},
                n=n, d=d, k=k, build_time_s=h.build_stats_["build_time_s"],
                memory_mb=_hnsw_memory_bytes(h) / 1e6, extra=h.build_stats_, **stats,
            )
        )
        log(f"[bench] hnsw(M={M},efc={efc},efs={efs}): "
            f"recall={stats['recall_at_k']:.3f} lat={stats['mean_latency_ms']:.3f}ms "
            f"qps={stats['qps']:.0f} speedup={bf_stats['mean_latency_ms']/stats['mean_latency_ms']:.1f}x")

    return results


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def results_to_markdown(results: list[BenchResult]) -> str:
    lines = [
        "| index | params | recall@k | mean ms | p95 ms | QPS | build s | mem MB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        params = ", ".join(f"{k}={v}" for k, v in r.params.items()) or "—"
        lines.append(
            f"| {r.index} | {params} | {r.recall_at_k:.3f} | {r.mean_latency_ms:.3f} "
            f"| {r.p95_latency_ms:.3f} | {r.qps:.0f} | {r.build_time_s:.2f} | {r.memory_mb:.2f} |"
        )
    return "\n".join(lines)


def save_results(results: list[BenchResult], out_dir: str, tag: str = "latest") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    (out / f"benchmark_{tag}.json").write_text(json.dumps(payload, indent=2))
    (out / f"benchmark_{tag}.md").write_text(results_to_markdown(results) + "\n")


def _main() -> None:
    import argparse

    from .vectors import load_spotify

    ap = argparse.ArgumentParser(description="Benchmark the vecsearch indexes.")
    ap.add_argument("--data", default="data/spotify_tracks.csv")
    ap.add_argument("--n", type=int, default=20000, help="corpus size cap")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", default="l2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--tag", default="latest")
    args = ap.parse_args()

    ds = load_spotify(args.data, limit=args.n, method="zscore")
    rng = np.random.default_rng(args.seed)
    q_idx = rng.choice(ds.n, size=min(args.queries, ds.n), replace=False)
    queries = ds.vectors[q_idx]

    results = run_benchmark(
        ds.vectors, queries, k=args.k, metric=args.metric, seed=args.seed
    )
    save_results(results, args.out, tag=args.tag)
    print("\n" + results_to_markdown(results))
    print(f"\nsaved -> {args.out}/benchmark_{args.tag}.{{json,md}}")


if __name__ == "__main__":
    _main()
