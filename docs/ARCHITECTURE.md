# Architecture

This document explains how each layer works and why it's built the way it is.
Everything sits on numpy; the only compiled code is an optional C++ distance
kernel.

## Data flow

```
spotify_tracks.csv
      │  load + select audio features (vectors.load_spotify)
      ▼
raw feature matrix  ──fit──▶  Normalizer (z-score)
      │                           │
      │  transform                │  transform (same stats)
      ▼                           ▼
normalized corpus (n×d)      normalized query (1×d)
      │                           │
      │ build                     │ search(k)
      ▼                           ▼
 [ BruteForce | IVF | HNSW ] ───▶ top-k ids + distances ───▶ track metadata
```

The same fitted `Normalizer` transforms both the corpus and every query, so
they live in one space. Mood queries enter this pipeline at the "normalized
query" node via `mood_translator.mood_to_query_vector`.

## Layer 1 — vectors & distances (`vectors.py`)

- `FEATURE_COLUMNS`: the 9 audio features used as dimensions (danceability,
  energy, loudness, speechiness, acousticness, instrumentalness, liveness,
  valence, tempo).
- `Normalizer`: fits `center`/`scale` (mean/std or min/max), guards constant
  columns against divide-by-zero, and can invert the transform (used by the app
  to display targets on the human scale).
- Distance kernels: `l2_distance`, `l2_distance_sq` (skips the sqrt;
  order-preserving for k-NN), `cosine_distance`. L2 uses
  `einsum("ij,ij->i", diff, diff)` to avoid materializing intermediate tensors.

## Layer 2 — brute force (`brute_force.py`)

Exact k-NN by computing all n distances and taking the smallest k via
`argpartition` (O(n) selection) then sorting only those k. Exact by
construction, so it is the ground-truth oracle every other index is scored
against. O(n·d) per query — it does not scale, which is the whole motivation for
the approximate indexes.

## Layer 3 — k-means (`kmeans.py`)

Lloyd's algorithm, from scratch:

1. **k-means++ seeding** — first center random; each subsequent center sampled
   with probability ∝ squared distance to the nearest chosen center. Spreads
   seeds out; big win over uniform-random init.
2. **Assignment** — vectorized nearest-center via the
   `‖p‖² + ‖c‖² − 2·p·c` expansion (one GEMM instead of an n×k×d tensor).
3. **Update** — each center → mean of its members. **Empty clusters** are
   re-seeded on the currently worst-served point, keeping all k cells alive
   (the IVF index depends on this).
4. **Restarts** — `n_init` independent runs; lowest inertia wins (k-means is
   non-convex).

## Layer 4 — IVF index (`ivf_index.py`)

Coarse quantization:

- **Build**: k-means carves the corpus into `nlist` Voronoi cells; store an
  inverted list of point ids per cell.
- **Search**: rank cells by centroid distance, scan only the `nprobe` nearest
  cells *exactly*, take top-k from those candidates.

Its only error source is boundary spill — a true neighbor sitting in an
unprobed neighboring cell. Raising `nprobe` trades latency for recall; at
`nprobe == nlist` it degenerates to exact search (a property the tests assert).

## Layer 5 — HNSW index (`hnsw_index.py`)

A hierarchical navigable small-world graph (Malkov & Yashunin 2016), the
algorithm behind most production vector DBs:

- **Layers**: each node gets a max level ~ `floor(−ln(U)·mL)`, `mL = 1/ln(M)`,
  so layers thin toward the top — a skip-list generalized to a graph. Top layers
  are a sparse long-range "express network"; layer 0 holds every node.
- **Search** (Algorithm 5): enter at the top node, greedily descend each layer
  with beam width 1, then widen to `ef_search` at layer 0 and return top-k.
- **Insert** (Algorithm 1): greedy-descend to the node's level, then from there
  down to 0 run an `ef_construction`-wide beam search and connect to `M`
  neighbors, bidirectionally, pruning over-full lists.
- **Neighbor selection** (Algorithm 4, heuristic): keep a candidate only if it's
  closer to the query than to any already-chosen neighbor — favors diverse edges
  and keeps the graph navigable, versus plain "M nearest" which clusters edges.

Knobs: `M` (degree; memory/recall), `ef_construction` (build beam; build
time/quality), `ef_search` (query beam; latency/recall).

## Layer 6 — C++ hot-path (`cpp/distance.cpp`)

HNSW's traversal is millions of single-pair distances. At d≈9 the numpy
per-call overhead dwarfs the ~9 multiply-adds. `distance.cpp` reimplements the
scalar and batched L2 kernels as tight, allocation-free loops exposed via
pybind11; the batch path releases the GIL. `vectors.py` imports the extension
when built and falls back to numpy otherwise, so behavior is identical either
way — only speed changes (~2.4×; see `docs/BENCHMARKS.md`).

## Layer 7 — semantic app (`mood_translator.py`, `app.py`)

- `mood_translator`: a curated lexicon maps mood terms → weighted per-feature
  targets; overlapping terms combine as weighted averages; the result is
  projected into the corpus's normalized space. Fully offline and deterministic;
  the "why these tracks" explanation is generated from the terms that fired.
- `app.py`: Streamlit UI — a Discover tab (mood → ranked tracks + feature-profile
  chart) and a Benchmarks tab (recall/latency/memory from saved runs).
