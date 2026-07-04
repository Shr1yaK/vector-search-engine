# Benchmarks & analysis

Approximate nearest-neighbor search is a multi-objective trade-off — an index is
never simply "faster", it's *faster at some recall*. This project measures four
axes (recall@k, latency, build time, memory) and reports them side by side. All
numbers below are reproducible:

```bash
python -m src.benchmark --n 20000 --queries 300 --tag n20k   # d=9 audio data
python benchmarks/bench_highdim.py --n 20000 --d 128         # d=128 embeddings
python benchmarks/bench_native.py                            # C++ vs numpy
```

Raw outputs are committed under `benchmarks/results/`.

## Metric definitions

- **recall@k** — |approx ∩ true| / k, where `true` is the brute-force result.
  Compared as sets, so intra-result ordering isn't penalized.
- **latency** — per-query wall time; mean and p95 reported.
- **QPS** — 1 / mean latency.
- **build time** — one-off index construction cost.
- **memory** — estimated bytes of the index structures (vectors + centroids +
  inverted lists for IVF; vectors + graph edges for HNSW).

## Finding 1 — dimensionality decides the winner

**d = 9 (Spotify audio features), n = 20k:**

| index | params | recall@10 | mean ms | QPS | speedup |
|---|---|---|---|---|---|
| brute force | — | 1.000 | 0.324 | 3,084 | 1.0× |
| IVF | nprobe=1 | 0.707 | 0.015 | 65,428 | 21.2× |
| IVF | nprobe=4 | 0.960 | 0.030 | 32,802 | 10.6× |
| IVF | nprobe=8 | 0.979 | 0.081 | 12,283 | 4.0× |
| HNSW | ef=10 | 0.980 | 0.414 | 2,413 | 0.8× |
| HNSW | ef=50 | 0.981 | 0.944 | 1,059 | 0.3× |

At d=9, IVF gives 0.96 recall at a 10× speedup, while HNSW is *slower than brute
force* — the graph traversal's Python-level bookkeeping (heaps, visited sets,
dict lookups) costs more than the tiny 9-dim distance it saves. Low dimensions
favor the simple cell scan.

**d = 128 (synthetic clustered embeddings), n = 20k:**

| index | params | recall@10 | mean ms | QPS | speedup |
|---|---|---|---|---|---|
| brute force | — | 1.000 | 0.732 | 1,366 | 1.0× |
| HNSW | ef=10 | 0.986 | 0.241 | 4,157 | 3.0× |
| HNSW | ef=50 | 1.000 | 0.346 | 2,890 | 2.1× |
| IVF | nprobe=4 | 1.000 | 0.055 | 18,191 | 13.3× |

At d=128 the per-distance cost rises, brute force slows, and HNSW now beats it
3×. IVF remains strong on this cleanly-clustered synthetic data, but HNSW's
advantage over exact search is the point: **its edge grows with dimensionality**,
which is exactly the regime real embeddings (RAG, recommendations) live in.

## Finding 2 — the knobs are the curve

Neither approximate index has a single operating point. `nprobe` (IVF) and
`ef_search` (HNSW) slide each index along its own recall/latency curve. On the
d=9 data, IVF moves from (0.71 recall, 0.015 ms) at nprobe=1 to (0.98, 0.081 ms)
at nprobe=8 — you buy recall with latency, continuously. Picking an index means
picking a point on the curve for your recall budget.

## Finding 3 — the bottleneck was call overhead, not math

Profiling the pure-Python HNSW build showed time dominated by the millions of
single-pair distance calls, not the arithmetic. Porting just that kernel to C++
(`cpp/distance.cpp`, pybind11):

| HNSW on 8k tracks | numpy | C++ kernel | speedup |
|---|---|---|---|
| build time | 33.9 s | 13.6 s | 2.50× |
| query latency | 0.775 ms | 0.322 ms | 2.41× |
| recall@10 | 0.987 | 0.987 | +0.0000 |

A 2.4× speedup on both build and query with **zero** change in which neighbors
are returned — the recall delta is exactly 0.0000, verified in the test suite
(`test_hnsw.py::test_native_matches_numpy_recall`). That's the discipline: an
optimization that moves the answers is a bug, not a speedup.

## Finding 4 — PQ buys memory, rerank buys the recall back

IVF-PQ stores product-quantized codes instead of full vectors. On the d=128
synthetic corpus:

| index | recall@10 | mean ms | memory | vs raw |
|---|---|---|---|---|
| brute force | 1.000 | 0.737 | 10.24 MB | 1× |
| hnsw (ef=50) | 1.000 | 0.301 | 15.61 MB | 1.5× larger |
| ivfpq (no rerank) | 0.363 | 0.642 | **0.36 MB** | **28× smaller** |
| ivfpq (rerank 100) | 0.943 | 0.662 | 0.36 MB | 28× smaller |

Raw PQ recall is low (0.36) because quantization is lossy — but it's an
excellent *candidate generator*: the true top-10 land inside its top-100
shortlist ~99% of the time. Re-ranking that shortlist with exact distances lifts
recall to 0.94 while the index itself stays 28× smaller than the raw vectors.
This is the memory/accuracy knob production systems live on (billions of vectors
in RAM), and the reason HNSW — which stores full vectors *plus* a graph — is the
largest index here, not the smallest.

## Caveats & honesty

- Benchmarks are single-machine, single-threaded Python (except the C++ kernel);
  absolute numbers are illustrative, the *relative* trade-offs are the point.
- The d=128 corpus is synthetic clustered data, not real embeddings; it's there
  to demonstrate the dimensionality crossover, clearly labeled as such.
- Pure-Python graph traversal caps HNSW's absolute throughput far below a C++
  library like hnswlib — expected, and precisely why the hot-path port exists.
