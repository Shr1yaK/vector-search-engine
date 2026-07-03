# C++ hot-path (pybind11)

The HNSW build/query loop is dominated not by arithmetic but by the per-call
overhead of running a distance function on one vector pair at a time. At d≈9,
numpy's call machinery costs more than the handful of multiply-adds it guards.
`distance.cpp` reimplements those kernels as tight, allocation-free C++ loops
exposed to Python via pybind11.

## Build

```bash
pip install pybind11
python cpp/build.py
```

This drops `vecsearch_native.*.so` in the repo root. `src/vectors.py` imports it
automatically when present and falls back to numpy when it isn't, so the library
works with or without the extension.

## Measured impact

`python benchmarks/bench_native.py` — on 8k tracks, routing HNSW's single-pair
distances through the C++ kernel gave a **~2.4× build and query speedup with
identical recall@10** (see `benchmarks/results/native_speedup.json`).
