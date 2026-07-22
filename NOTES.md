# NOTES — the complete guide to this project

This is the one document that explains **everything**: what the project is, how
every piece works, why it's built this way, how to run and verify it, and how to
talk about it in an interview. If you read only one file, read this one.

`README.md` is the polished public-facing pitch. `docs/ARCHITECTURE.md` and
`docs/BENCHMARKS.md` go deep on specific areas. This file ties it all together
in plain language.

---

## 1. What this project is (in one paragraph)

It's a **vector search engine built from scratch** — the thing that lives
*inside* a vector database like FAISS or Pinecone — plus a **semantic music
search app** on top of it. Given 114,000 Spotify tracks described by audio
features (energy, valence, tempo, acousticness…), it finds the tracks most
similar to a query. It implements four different search indexes by hand (no
library shortcuts), measures their speed/accuracy/memory trade-offs rigorously,
speeds up the hot path with C++, and wraps it in an app where you type a mood
("chill rainy day coding music") and get matching songs.

**The point it proves:** most ML candidates *import* a vector database. This
builds the index itself — the clustering, the graph search, the compression —
which is the harder, more impressive systems story, and it's directly relevant
because vector search underlies every RAG / recommendation system being shipped.

---

## 2. The big idea (the mental model)

Everything a track "is" gets turned into a **point in 9-dimensional space** (9
audio features). Two songs that *feel* similar sit close together; songs that
feel different sit far apart. "Search" = "find the nearest points to this one."

The naive way to find nearest points is to measure the distance to **every**
point and take the smallest — that's **brute force**. It's exact but slow: it
doesn't scale to millions of vectors. So the whole field (and this project) is
about **approximate** methods that are *almost* as accurate but dramatically
faster: don't look at every point, be clever about which ones you check.

That's the entire story. Everything below is variations on "be clever about
which points you check, and prove you didn't lose much accuracy doing it."

---

## 3. The four indexes (what they are, in plain words)

| Index | One-line idea | Wins when |
|---|---|---|
| **Brute force** | Check every point. Exact, slow. The "correct answer" everything else is graded against. | You need ground truth, or the dataset is tiny. |
| **IVF** | Cluster the points into groups once; at search time only look inside the few nearest groups. | Low-dimensional data (like our 9 audio features). |
| **HNSW** | Build a graph where each point links to nearby points; "walk" the graph toward the query. | High-dimensional data (real embeddings, 100s of dims). |
| **IVF-PQ** | Like IVF, but *compress* each vector to ~16 bytes so billions fit in RAM. | Memory is the constraint (huge datasets). |

Analogies that help:
- **IVF** = finding a restaurant by first picking the right neighborhood, then
  only walking that neighborhood — instead of walking the whole city.
- **HNSW** = a subway map. Express lines (top layers) get you to the right
  region in a few hops; local lines (bottom layer) get you the last mile.
- **IVF-PQ** = storing a *thumbnail* of each vector instead of the full image.
  You lose detail but fit way more; if you need precision, you re-check the few
  finalists at full resolution ("re-ranking").

---

## 4. How the pieces fit — file by file

Read the code in this order; each builds on the last.

```
src/
├── vectors.py        FOUNDATION. Loads the CSV, normalizes features so no single
│                     feature dominates, defines distance functions (L2, cosine).
│                     Everything operates on the matrix this produces.
├── brute_force.py    Exact nearest-neighbor search. The ground-truth "oracle".
├── kmeans.py         Clustering from scratch (Lloyd's algorithm + k-means++).
│                     Used by IVF to make its groups.
├── ivf_index.py      IVF index: cluster once, probe the nearest few cells.
├── hnsw_index.py     HNSW index: the multi-layer navigable graph. The meatiest file.
├── pq.py             Product Quantization: compress vectors into tiny codes.
├── ivfpq_index.py    IVF + PQ combined, with optional exact re-ranking.
├── benchmark.py      Measures recall / latency / build time / memory for all indexes.
├── mood_translator.py  Turns "sad rainy night" into a query vector (offline, rule-based).
├── search.py         Command-line search (python -m src.search "...").
└── app.py            The Streamlit web app.

cpp/
├── distance.cpp      C++ version of the distance math (the speed-critical inner loop).
└── build.py          One command to compile it.

benchmarks/           Scripts that produce the numbers and the plot in the README.
tests/                65 tests. The proof that every index actually works.
scripts/fetch_data.py Downloads the dataset.
docs/                 Architecture + benchmark write-ups + the plot image.
```

**The key relationships:**
- Everything depends on `vectors.py`.
- `ivf_index.py` uses `kmeans.py`.
- `ivfpq_index.py` uses `kmeans.py` **and** `pq.py`.
- `hnsw_index.py` optionally uses the compiled `cpp/distance.cpp`.
- `app.py` and `search.py` use `mood_translator.py` + any index.
- Every index is checked against `brute_force.py` in the tests.

---

## 5. How each algorithm actually works (the interview-level detail)

### Normalization (`vectors.py`)
Audio features are on wildly different scales: tempo is 0–250 BPM, loudness is
−60–0 dB, valence is 0–1. If you measured distance raw, tempo would swamp
everything. So we **z-score** each feature (subtract mean, divide by standard
deviation) so they're all comparable. Crucially, the *same* transform is applied
to the corpus and to every query — otherwise the query would live in a different
space than the data (a classic "train/serve skew" bug).

### k-means (`kmeans.py`)
Splits points into `k` groups. Algorithm: (1) **k-means++** picks smart starting
centers spread far apart; (2) assign every point to its nearest center; (3) move
each center to the average of its members; (4) repeat until stable. We handle the
edge case where a cluster goes empty (re-seed it) and run it a few times keeping
the best, because k-means can get stuck in bad local optima.

### IVF (`ivf_index.py`)
Build: run k-means to carve the data into `nlist` cells; remember which points
are in each cell ("inverted lists"). Search: find the `nprobe` cells whose
centers are nearest the query, and only scan the points in those cells. The knob
`nprobe` trades speed for accuracy — probe more cells, catch more true neighbors,
but slower. Its only error is a true neighbor sitting just across a cell border
in an unprobed cell.

### HNSW (`hnsw_index.py`) — the hard one
A graph where each point connects to ~16 nearby points, arranged in **layers**.
Higher layers are sparse (few points, long-range links); the bottom layer has
everything (dense, short-range links) — like a skip list turned into a graph.
- **Search:** start at the top, greedily hop toward the query until you can't get
  closer, drop a layer, repeat. Only at the bottom do you widen the search
  (`ef_search`) to collect the final k. This finds neighbors in roughly
  *logarithmic* hops instead of scanning everything.
- **Insert:** each new point gets a random top layer (exponential distribution),
  finds its nearest existing points via the same greedy search, and links to the
  best `M` of them (bidirectionally), with a diversity heuristic so links don't
  all point the same direction.
- Knobs: `M` (links per node → memory/recall), `ef_construction` (build
  quality), `ef_search` (query accuracy).

### Product Quantization (`pq.py`) + IVF-PQ (`ivfpq_index.py`)
Split each vector into `m` chunks; run k-means in each chunk's subspace to get a
codebook of 256 representative sub-vectors; store each vector as `m` bytes (which
codebook entry each chunk is closest to). A 128-dim float vector (512 bytes)
becomes 16 bytes — **32× smaller**. Search never decompresses: it precomputes a
small table of distances from the query to every codebook entry, then a stored
vector's distance is just `m` table lookups added up ("ADC"). IVF-PQ does this on
the *residual* (vector minus its cell center), which quantizes more accurately.
Because compression loses precision, recall is low (~0.5) — but the true
neighbors are almost always in the top-100 shortlist, so an optional exact
**re-rank** of that shortlist recovers recall to ~0.94 while keeping the 20–30×
memory saving.

### C++ hot-path (`cpp/distance.cpp`)
HNSW does *millions* of tiny distance calculations during a build. In Python,
the per-call overhead (not the math) dominates. Rewriting just that inner loop in
C++ (exposed to Python via pybind11) made the build and query **2.4× faster with
identical results**. This is the "profiled it, found the real bottleneck, fixed
it, verified nothing changed" story.

### Mood translator (`mood_translator.py`)
Fully offline, no API/LLM. A hand-built dictionary maps mood words to target
audio-feature values ("chill" → low energy, acoustic; "workout" → high energy,
fast tempo). Overlapping words get averaged. The result is projected into the
same normalized space as the corpus and handed to an index. It's deterministic,
free, testable, and self-explaining (the app's "why these tracks" text comes
straight from which words fired).

---

## 6. What the benchmarks taught (the findings that matter)

These are the real, defensible results — the "methodology, not just it works"
part. Full detail in `docs/BENCHMARKS.md`.

1. **Dimensionality decides the winner.** On the 9-D audio data, **IVF wins**
   (0.96 recall at 10× speedup); HNSW is actually *slower than brute force*
   because its graph-walking overhead isn't worth it at low dimensions. On
   synthetic 128-D data, it **flips** — HNSW beats brute force 3×. Real
   embeddings are high-dimensional, which is why HNSW dominates production.
2. **The knobs are a curve, not a point.** `nprobe` (IVF) and `ef_search`
   (HNSW) slide each index along its own accuracy/speed trade-off. Choosing an
   index means choosing a point on that curve for your accuracy budget.
3. **The bottleneck was call overhead, not math.** Hence the C++ port: 2.4×
   faster, recall delta exactly 0.0000.
4. **PQ buys memory; re-ranking buys the accuracy back.** IVF-PQ is 28× smaller
   than raw vectors at 128-D; raw recall 0.36 → 0.94 after re-ranking a
   shortlist. This is why HNSW (which stores full vectors *plus* a graph) is the
   *largest* index, not the smallest.

**The single best interview line:** *"I showed IVF wins at low dimensionality
while HNSW wins on high-dimensional embeddings, and cut HNSW build/query time
2.4× with a C++ kernel at zero recall cost."*

---

## 7. How to RUN everything

One-time setup (use `python3`, not `python`, on your Mac):
```bash
cd ~/Downloads/vector-search-engine
python3 -m pip install -r requirements.txt   # install dependencies
python3 scripts/fetch_data.py                # download the dataset (~20 MB)
python3 cpp/build.py                          # optional: compile the C++ speedup
```

Run the tests (proof it all works):
```bash
python3 -m pytest                             # expect: 65 passed
```

Search from the command line:
```bash
python3 -m src.search "chill rainy day coding music" --k 5
python3 -m src.search "high energy gym workout" --index ivf --k 8
python3 -m src.search "acoustic study" --index ivfpq --genre acoustic --genre ambient
python3 -m src.search "sad breakup song" --index brute --k 5     # exact, for comparison
```
Flags: `--index hnsw|ivf|ivfpq|brute`, `--k N`, `--n N` (how many tracks to load),
`--genre X` (repeatable filter).

Launch the web app:
```bash
python3 -m streamlit run src/app.py           # opens in your browser
```
In the app: type a mood, switch indexes in the sidebar, open the **Benchmarks**
tab, try the genre/popularity **Filters** expander.

Regenerate the benchmarks and plot:
```bash
python3 -m src.benchmark --n 20000 --queries 300     # d=9 audio data
python3 benchmarks/bench_highdim.py --n 20000 --d 128 # d=128 embeddings
python3 benchmarks/bench_native.py                    # C++ vs numpy
python3 benchmarks/plot_results.py                    # regenerate the README plot
```

There's also a `Makefile`: `make test`, `make app`, `make bench`, `make native`,
etc. Run `make help` to list targets.

---

## 8. How to CHECK / verify the claims yourself

**Claim: "approximate search matches exact search."** Run both, compare:
```bash
python3 -c "
from src.vectors import load_spotify
from src.brute_force import BruteForceIndex
from src.hnsw_index import HNSWIndex
ds = load_spotify('data/spotify_tracks.csv', limit=20000)
bf, h = BruteForceIndex(ds.vectors), HNSWIndex(random_state=0).build(ds.vectors)
q = ds.vectors[42]
print('exact:', sorted(bf.search(q,10)[0].tolist()))
print('hnsw :', sorted(h.search(q,10)[0].tolist()))
"
```
The two lists should overlap almost completely. That's the whole game.

**Claim: "the tests prove correctness."** `python3 -m pytest -v` shows all 65,
named by what they check (e.g. `test_full_probe_equals_brute_force`,
`test_native_matches_numpy_recall`).

**Claim: "C++ is 2.4× faster with no accuracy loss."**
```bash
python3 benchmarks/bench_native.py
```
Prints build/query times for numpy vs C++ and confirms recall delta ≈ 0.

**Claim: "IVF-PQ is ~20× smaller."** The benchmark output has a `memory` column;
compare the `ivfpq` row to `brute_force`.

---

## 9. The screenshot task (yes, this is the one thing left for YOU)

The README has a placeholder for an app screenshot at `docs/img/app.png` that
doesn't exist yet (I couldn't capture the browser UI headlessly). To add it:

1. `python3 -m streamlit run src/app.py`
2. In the browser, type a mood, let results show (optionally open the Benchmarks
   tab or the Filters).
3. Take a screenshot: **Cmd+Shift+4**, drag over the app area (or Cmd+Shift+4
   then Spacebar to grab the whole window).
4. Save/move it to `docs/img/app.png` in the project.
5. Commit it:
   ```bash
   cd ~/Downloads/vector-search-engine
   git add docs/img/app.png
   git commit -m "Docs: add app screenshot"
   git push
   ```

A short **screen recording** (Cmd+Shift+5 → Record) of you typing a mood and
getting results is even better for a portfolio/LinkedIn post, but it doesn't go
in the repo — keep it as a file to share. The screenshot is the one thing the
repo itself is missing.

---

## 10. Honest limitations (know these before an interview)

- **It's single-machine, single-threaded Python** (except the C++ kernel). The
  *relative* trade-offs are the result; the absolute speeds aren't competitive
  with a production C++ library like hnswlib — and that's stated openly. The
  point is understanding, not beating FAISS.
- **The d=128 benchmark uses synthetic data**, clearly labeled, to show the
  dimensionality crossover (the real audio data is only 9-D).
- **Genre labels in results can look mismatched** — the engine matches on how
  tracks *sound* (audio features), not on genre tags. That's intended.
- **Loading only the first N rows** (`--n`) pulls alphabetically-early genres
  first (the CSV is sorted by genre), so small `--n` shows limited genres.

---

## 11. Hiccups found in portfolio review — and how they were fixed

A pass over live screenshots of the running app surfaced real issues. Recording
them here because *"what broke and how I found it"* is often a better interview
answer than the feature list.

**1. The corpus was genre-biased (the worst one).**
`load_spotify(limit=n)` took the **first n rows**, but the CSV is sorted by
`track_genre` — so a 15k corpus contained only 15 alphabetically-early genres
(acoustic, afrobeat, anime, blues…). Every result looked like obscure afrobeat,
and the genre filter had no "pop"/"rock"/"edm" to pick. *Fix:* added
`sample_seed` to `load_spotify`; the app now loads a **seeded random sample**, so
15k tracks span all **114 genres**. Benchmarks deliberately keep the old
head-based behaviour so their corpus stays identical across runs.

**2. Selective filters starved the approximate indexes (a real bug).**
Asking for 10 results with a narrow filter (e.g. 194 of 15,000 tracks) returned
only **2**. Cause: post-filtering an approximate traversal — the HNSW beam and
the IVF probed cells filled with non-matching points, so fewer than k survived.
*Fix:* every index now switches strategy on **filter selectivity**. Below a
threshold it **pre-filters** — scans the matching subset exactly
(`vectors.exact_subset_search`) — which is both cheaper *and* exact; above it,
post-filters with a beam widened in proportion to what the filter discards. This
is the pre- vs post-filtering trade real filtered-ANN systems make. Regression
test: `test_filtering.py::test_selective_filter_does_not_starve`.

**3. The app's latency number contradicted the README.**
The Discover tab timed **one** query, so IVF showed 0.67 ms vs brute force's
0.47 ms — the opposite of the benchmark's 10.6× speedup. At sub-millisecond
scale a single reading is mostly interpreter/framework noise. *Fix:* the app now
reports the **mean of 20 repeated queries**, with a tooltip pointing at the
controlled 300-query benchmark. Lesson: never quote a single timing of a
sub-millisecond operation.

**4. IVF-PQ was benchmarked but not selectable.** The Benchmarks tab charted
four indexes while the Discover picker offered three. *Fix:* IVF-PQ added to the
picker (with `rerank=100`, since raw PQ recall is ~0.5).

**5. The C++ speedup was only a number in a JSON file.** *Fix:* the "How it
works" tab now has a **live micro-benchmark button** that runs the kernel both
ways on the spot and prints the speedup *and* the max difference between the two
results (≈1e-7 — identical answers, just faster).

**6. Smaller things.** Dead `chart_df` variable removed; build time now shown as
a metric (HNSW ~25 s vs IVF <1 s makes the trade visible); a caption explains
`similarity = 1/(1+distance)` and why genre labels can look unrelated; a warning
appears before building HNSW over ≥50k tracks (minutes in pure Python); the
Benchmarks tab embeds the committed **log-axis** figure, since Streamlit's linear
scatter bunches the fast indexes together.

**7. Not a bug:** the "Record screen" / "Next steps" modal is **Arc's built-in
browser recorder**, not a project feature. Useful for making a demo video — but
never describe it as something the project does.

**Worth pointing at in an interview:** at 15k tracks, HNSW, IVF, IVF-PQ and brute
force return the *same* top-10 with identical similarity scores for the same
query. That's live proof of "approximate ≈ exact" — better than citing a JSON
file.

## 12. Glossary (quick reference)

- **Vector / embedding** — a list of numbers representing an item (here, a song's
  audio features).
- **k-NN** — k nearest neighbors; the k closest points to a query.
- **Recall@k** — of the true k nearest, what fraction did the index return.
  1.0 = perfect.
- **Latency / QPS** — time per query / queries per second.
- **ANN** — approximate nearest neighbor (trades a little accuracy for big speed).
- **IVF** — inverted file (clustering-based index).
- **HNSW** — hierarchical navigable small world (graph-based index).
- **PQ** — product quantization (vector compression).
- **Centroid** — a cluster's center point.
- **Residual** — a vector minus its cluster center.
- **ADC** — asymmetric distance computation (PQ's table-lookup distance trick).
- **Re-ranking** — re-scoring a shortlist with exact distances to recover accuracy.

---

## 13. If you want to keep improving it later

Ideas discussed but intentionally *not* built (the project is complete without
them): inner-product / MIPS metric (relevant to recommendations & RAG), a k-NN
genre *classifier* on top of the index (adds a downstream ML task but overlaps
with coursework and has a low accuracy ceiling), scalar quantization, a disk-based
index. None are necessary — reach for them only if a specific job asks for it.

The best *next* thing is usually not another algorithm — it's being able to
explain the ones here fluently. This file is meant to get you there.
