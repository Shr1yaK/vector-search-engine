"""Reproducible A/B benchmark: pure-Python vs C++ hot-path HNSW.

Builds the same HNSW graph twice — once routing single-pair distances through
numpy, once through the compiled ``vecsearch_native`` kernel — and reports the
build/query speedup while confirming recall is unchanged (the whole point: the
optimization must not move the answers).

    python benchmarks/bench_native.py            # uses data/spotify_tracks.csv
    python benchmarks/bench_native.py --n 8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brute_force import BruteForceIndex  # noqa: E402
from src.hnsw_index import HNSWIndex  # noqa: E402
from src.vectors import load_spotify, native_available  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/spotify_tracks.csv")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=50)
    ap.add_argument("--out", default="benchmarks/results/native_speedup.json")
    args = ap.parse_args()

    if not native_available():
        raise SystemExit(
            "vecsearch_native is not built. Run `python cpp/build.py` first."
        )

    ds = load_spotify(args.data, limit=args.n)
    bf = BruteForceIndex(ds.vectors)
    rng = np.random.default_rng(3)
    qs = ds.vectors[rng.integers(ds.n, size=args.queries)]
    truth = [set(bf.search(q, args.k)[0].tolist()) for q in qs]

    rows = {}
    for native in (False, True):
        t0 = time.perf_counter()
        h = HNSWIndex(M=16, ef_construction=200, random_state=0, use_native=native).build(ds.vectors)
        build_s = time.perf_counter() - t0

        hits = 0
        t0 = time.perf_counter()
        for q, tr in zip(qs, truth):
            hits += len(tr & set(h.search(q, args.k, ef_search=args.ef)[0].tolist()))
        query_ms = (time.perf_counter() - t0) / len(qs) * 1000.0
        recall = hits / (args.k * len(qs))
        key = "native" if native else "numpy"
        rows[key] = {"build_s": build_s, "query_ms": query_ms, "recall": recall}
        print(f"use_native={native!s:5s}  build={build_s:6.2f}s  "
              f"query={query_ms:.3f}ms  recall@{args.k}={recall:.3f}")

    speedup = {
        "build_x": rows["numpy"]["build_s"] / rows["native"]["build_s"],
        "query_x": rows["numpy"]["query_ms"] / rows["native"]["query_ms"],
        "recall_delta": rows["native"]["recall"] - rows["numpy"]["recall"],
    }
    print(f"\nspeedup: build {speedup['build_x']:.2f}x, query {speedup['query_x']:.2f}x, "
          f"recall delta {speedup['recall_delta']:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": vars(args), "results": rows, "speedup": speedup}, indent=2
    ))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
