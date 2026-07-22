"""Streamlit app: semantic mood-based music discovery over a from-scratch index.

Two surfaces:

* **Discover** — type a mood ("chill rainy day coding music"), we translate it
  to an audio-feature target, query the hand-built HNSW/IVF index, and show the
  matching tracks with a transparent "why these tracks" breakdown and a
  feature-profile chart.
* **Benchmarks** — the recall/latency/build/memory trade-offs across
  brute-force, IVF, and HNSW, loaded from the saved benchmark runs, so the
  systems story sits right next to the product.

Run:  streamlit run src/app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Allow `streamlit run src/app.py` from the repo root to import the package.
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vectors import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, load_spotify, native_available  # noqa: E402
from src.brute_force import BruteForceIndex  # noqa: E402
from src.ivf_index import IVFIndex  # noqa: E402
from src.ivfpq_index import IVFPQIndex  # noqa: E402
from src.hnsw_index import HNSWIndex  # noqa: E402
from src.mood_translator import available_moods, mood_to_query_vector  # noqa: E402

DATA_PATH = ROOT / "data" / "spotify_tracks.csv"
RESULTS_DIR = ROOT / "benchmarks" / "results"
PLOT_PATH = ROOT / "docs" / "img" / "benchmark_tradeoff.png"

# Seed for the corpus sample. The raw CSV is sorted by genre, so loading the
# first N rows would only ever surface alphabetically-early genres; sampling
# gives a corpus spanning all 114.
SAMPLE_SEED = 0

# Timing a single query in the app measures Python + framework overhead as much
# as the index. We repeat the query and report the mean so the number shown is
# comparable to the controlled benchmark.
LATENCY_REPEATS = 20

st.set_page_config(page_title="Semantic Music Search", page_icon="🎧", layout="wide")


# --------------------------------------------------------------------------- #
# cached heavy objects
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading Spotify tracks…")
def _load(n: int):
    # sample_seed => a genre-representative corpus (see SAMPLE_SEED note above).
    return load_spotify(str(DATA_PATH), limit=n, sample_seed=SAMPLE_SEED)


@st.cache_resource(show_spinner="Building HNSW graph index… (slowest to build)")
def _build_hnsw(n: int):
    ds = _load(n)
    t0 = time.perf_counter()
    idx = HNSWIndex(M=16, ef_construction=200, random_state=0).build(ds.vectors)
    return idx, time.perf_counter() - t0


@st.cache_resource(show_spinner="Building IVF index…")
def _build_ivf(n: int):
    ds = _load(n)
    nlist = int(np.sqrt(n)) or 1
    t0 = time.perf_counter()
    idx = IVFIndex(nlist=nlist, nprobe=8, random_state=0).build(ds.vectors)
    return idx, time.perf_counter() - t0


@st.cache_resource(show_spinner="Building IVF-PQ index (compressed)…")
def _build_ivfpq(n: int):
    ds = _load(n)
    nlist = int(np.sqrt(n)) or 1
    t0 = time.perf_counter()
    # m=3 subspaces over the 9 audio features; keep_vectors enables exact rerank,
    # which is what makes PQ's results competitive rather than merely small.
    idx = IVFPQIndex(nlist=nlist, m=3, ksub=256, nprobe=16, random_state=0).build(
        ds.vectors, keep_vectors=True
    )
    return idx, time.perf_counter() - t0


@st.cache_resource(show_spinner="Preparing brute-force oracle…")
def _build_bf(n: int):
    ds = _load(n)
    t0 = time.perf_counter()
    idx = BruteForceIndex(ds.vectors)
    return idx, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🎧 Semantic Music Search")
st.sidebar.caption("Vector search engine built from scratch — k-means · IVF · HNSW")

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found at `{DATA_PATH}`.\n\n"
        "Download it (see README) and place `spotify_tracks.csv` in `data/`."
    )
    st.stop()

n = st.sidebar.select_slider(
    "Corpus size (tracks indexed)",
    options=[5000, 10000, 15000, 25000, 50000, 114000],
    value=15000,
    help="Larger = richer results but slower first-time index build. "
         "The corpus is a seeded random sample, so every genre is represented.",
)
index_name = st.sidebar.radio(
    "Index",
    ["HNSW (graph)", "IVF (clustering)", "IVF-PQ (compressed)", "Brute force (exact)"],
    index=0,
    help="All four are implemented from scratch. They should return near-identical "
         "results — the difference is speed, memory, and build cost.",
)

# HNSW's pure-Python build is the slow one; warn before someone waits minutes.
if index_name.startswith("HNSW") and n >= 50000:
    st.sidebar.warning(
        f"Building an HNSW graph over {n:,} tracks in pure Python takes several "
        "minutes (one time, then cached). IVF or IVF-PQ build in seconds."
    )

st.sidebar.markdown(
    f"**C++ hot-path:** {'✅ active' if native_available() else '➖ numpy fallback'}"
)
st.sidebar.caption(
    "The distance kernel is compiled C++ (pybind11) when built — "
    "~2.4× faster build and query at identical recall."
    if native_available() else
    "Run `python cpp/build.py` to enable the compiled distance kernel (~2.4× faster)."
)

ds = _load(n)


def get_index():
    """Return (index, build_seconds, search_callable)."""
    if index_name.startswith("HNSW"):
        idx, bt = _build_hnsw(n)
        return idx, bt, lambda q, k, allowed=None: idx.search(q, k, ef_search=80, allowed=allowed)
    if index_name.startswith("IVF-PQ"):
        idx, bt = _build_ivfpq(n)
        # rerank: PQ alone is lossy; re-scoring a shortlist exactly restores quality.
        return idx, bt, lambda q, k, allowed=None: idx.search(
            q, k, nprobe=16, allowed=allowed, rerank=100
        )
    if index_name.startswith("IVF"):
        idx, bt = _build_ivf(n)
        return idx, bt, lambda q, k, allowed=None: idx.search(q, k, nprobe=16, allowed=allowed)
    idx, bt = _build_bf(n)
    return idx, bt, lambda q, k, allowed=None: idx.search(q, k, allowed=allowed)


tab_discover, tab_bench, tab_about = st.tabs(["🔎 Discover", "📊 Benchmarks", "ℹ️ How it works"])

# --------------------------------------------------------------------------- #
# Discover tab
# --------------------------------------------------------------------------- #
with tab_discover:
    st.subheader("Describe a vibe, get tracks")
    col_in, col_k = st.columns([4, 1])
    with col_in:
        mood = st.text_input(
            "Mood / activity / feeling",
            value="chill rainy day coding music",
            placeholder="e.g. high energy gym workout, sad breakup song, happy summer party",
        )
    with col_k:
        k = st.number_input("Results", 1, 50, 10)

    st.caption("Recognized mood words include: " + ", ".join(available_moods()[:24]) + " …")

    # Metadata filters — the "search only within X" feature. Builds a boolean
    # mask over the corpus that every index accepts via `allowed=`.
    allowed_mask = None
    if ds.metadata is not None:
        with st.expander("Filters (optional) — genre, popularity"):
            fc1, fc2 = st.columns([3, 2])
            genres = sorted(ds.metadata["track_genre"].dropna().unique().tolist()) \
                if "track_genre" in ds.metadata.columns else []
            picked = fc1.multiselect("Restrict to genres", genres, default=[])
            min_pop = fc2.slider("Minimum popularity", 0, 100, 0) \
                if "popularity" in ds.metadata.columns else 0
            mask = np.ones(ds.n, dtype=bool)
            if picked:
                mask &= ds.metadata["track_genre"].isin(picked).to_numpy()
            if min_pop > 0:
                mask &= (ds.metadata["popularity"].to_numpy() >= min_pop)
            if picked or min_pop > 0:
                allowed_mask = mask
                st.caption(f"Filter active — {int(mask.sum()):,} of {ds.n:,} tracks match.")

    if mood.strip():
        index_obj, build_t, search = get_index()
        qv, mq = mood_to_query_vector(mood, ds)

        # Repeat the query and take the mean: a single timing at sub-millisecond
        # scale is dominated by interpreter noise, not the index.
        ids, dists = search(qv, int(k), allowed=allowed_mask)
        t0 = time.perf_counter()
        for _ in range(LATENCY_REPEATS):
            search(qv, int(k), allowed=allowed_mask)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / LATENCY_REPEATS

        st.info(f"**Why these tracks:** {mq.explanation(FEATURE_DESCRIPTIONS)}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Query latency", f"{elapsed_ms:.3f} ms",
                  help=f"Mean of {LATENCY_REPEATS} repeated queries. Single-shot "
                       "timings at this scale are mostly interpreter noise — see "
                       "the Benchmarks tab for the controlled 300-query runs.")
        m2.metric("Index build", f"{build_t:.2f} s",
                  help="One-time cost to construct this index (then cached). "
                       "HNSW is the expensive one; that's the trade for fast queries.")
        m3.metric("Index", index_name.split(" (")[0])
        m4.metric("Corpus", f"{ds.n:,} tracks")

        if len(ids):
            rows = ds.metadata.iloc[ids].copy()
            rows.insert(0, "similarity", np.round(1.0 / (1.0 + dists), 3))
            show_cols = [c for c in ("track_name", "artists", "track_genre", "popularity", "similarity") if c in rows.columns]
            st.dataframe(rows[show_cols], use_container_width=True, hide_index=True)
            st.caption(
                "`similarity` = 1 / (1 + distance) — 1.0 is an exact match, and it "
                "falls off as tracks get further away in audio-feature space. "
                "Genre labels can look unrelated because the engine matches how a "
                "track *sounds*, not how it's tagged."
            )

            # Feature profile: target vs. mean of the returned tracks (raw scale).
            st.markdown("##### Audio-feature profile — your target vs. the results")
            target = mq.raw_vector
            got_mean = ds.raw[ids].mean(axis=0)
            prof = pd.DataFrame(
                {"target": target, "results (avg)": got_mean},
                index=list(FEATURE_COLUMNS),
            )
            # Normalize tempo/loudness onto a comparable 0-1 scale for the chart.
            prof_disp = prof.copy()
            for feat in ("tempo", "loudness"):
                lo, hi = ds.raw[:, list(FEATURE_COLUMNS).index(feat)].min(), ds.raw[:, list(FEATURE_COLUMNS).index(feat)].max()
                prof_disp.loc[feat] = (prof.loc[feat] - lo) / (hi - lo + 1e-9)
            st.bar_chart(prof_disp)
        else:
            st.warning("No results — try a different mood phrase.")


# --------------------------------------------------------------------------- #
# Benchmarks tab
# --------------------------------------------------------------------------- #
with tab_bench:
    st.subheader("Recall vs. latency vs. memory — measured, not asserted")
    result_files = sorted(RESULTS_DIR.glob("benchmark_*.json")) if RESULTS_DIR.exists() else []
    if not result_files:
        st.warning(
            "No saved benchmark runs found. Generate one with:\n\n"
            "```\npython -m src.benchmark --n 20000 --queries 300 --tag n20k\n```"
        )
    else:
        pick = st.selectbox("Benchmark run", [f.name for f in result_files],
                            index=len(result_files) - 1)
        data = json.loads((RESULTS_DIR / pick).read_text())
        df = pd.DataFrame(data)
        df["label"] = df.apply(
            lambda r: r["index"] + (" " + ", ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else ""),
            axis=1,
        )
        st.markdown("**Recall@k vs. mean query latency** (up-and-to-the-left is better)")
        st.scatter_chart(df, x="mean_latency_ms", y="recall_at_k", color="index",
                         size="memory_mb")
        st.caption(
            "Bubble size = index memory. Note the linear x-axis bunches the fast "
            "indexes together — the committed figure below uses a log axis."
        )
        st.dataframe(
            df[["index", "label", "recall_at_k", "mean_latency_ms", "p95_latency_ms", "qps", "build_time_s", "memory_mb"]],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Brute force is exact (recall 1.0) but scans everything. IVF and HNSW "
            "trade a little recall for large speedups; the knobs (nprobe / ef_search) "
            "slide each index along its own curve. IVF-PQ gives up recall for a "
            "~10–28× smaller index, and recovers it by exact-reranking a shortlist."
        )

        if PLOT_PATH.exists():
            st.markdown("---")
            st.markdown("**The committed figure** (log latency axis, both dimensionalities)")
            st.image(str(PLOT_PATH), use_container_width=True)
            st.caption(
                "Left: at d=9 IVF wins. Right: at d=128 the picture flips and HNSW "
                "beats exact search — its advantage grows with dimensionality. "
                "Regenerate with `python benchmarks/plot_results.py`."
            )


# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
### What's under the hood

Every layer here is implemented from scratch on top of numpy — no FAISS, no
Pinecone, no sklearn:

1. **Vectors & distances** — Spotify audio features (energy, valence, tempo,
   acousticness, …) normalized into a shared space.
2. **Brute force** — exact k-NN, the ground-truth oracle.
3. **k-means (Lloyd + k-means++)** — carves the corpus into cells.
4. **IVF index** — probe only the nearest cells instead of scanning all points.
5. **HNSW index** — a multi-layer navigable small-world graph; greedy traversal
   finds neighbors in logarithmic-ish hops.
6. **Product quantization / IVF-PQ** — compresses each vector to a few bytes
   (~10–28× smaller) and scores via table lookups, with an exact rerank of the
   shortlist to recover the recall compression costs.
7. **C++ hot-path (pybind11)** — the per-hop distance kernel, ported to compiled
   code for a ~2.4× build/query speedup at identical recall.
8. **Mood translator** — a deterministic lexicon mapping natural-language mood
   to an audio-feature target (fully offline, self-explaining).

The **Benchmarks** tab shows the measured recall/latency/memory trade-offs that
justify each index choice.
"""
    )

    st.markdown("---")
    st.markdown("##### Prove the C++ hot-path, live")
    st.caption(
        "Runs the same distance kernel through the compiled extension and through "
        "numpy, right now, on this machine."
    )
    if st.button("Run the C++ vs numpy micro-benchmark"):
        if not native_available():
            st.warning("Extension not built. Run `python cpp/build.py`, then restart.")
        else:
            import vecsearch_native as _vn
            from src.vectors import l2_distance

            rng = np.random.default_rng(0)
            mat = np.ascontiguousarray(rng.random((50000, 9), dtype=np.float32))
            q = np.ascontiguousarray(rng.random(9).astype(np.float32))

            t0 = time.perf_counter()
            for _ in range(50):
                np_out = l2_distance(q, mat)
            np_ms = (time.perf_counter() - t0) * 1000 / 50

            t0 = time.perf_counter()
            for _ in range(50):
                c_out = _vn.l2_batch(q, mat)
            c_ms = (time.perf_counter() - t0) * 1000 / 50

            c1, c2, c3 = st.columns(3)
            c1.metric("numpy", f"{np_ms:.3f} ms")
            c2.metric("C++ kernel", f"{c_ms:.3f} ms")
            c3.metric("Speedup", f"{np_ms / c_ms:.2f}×")
            st.success(
                f"Max absolute difference between the two results: "
                f"{float(np.max(np.abs(np_out - c_out))):.2e} — identical answers, "
                "just faster."
            )

    st.caption("Built by Shriya Kansal.")
