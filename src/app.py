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
from src.hnsw_index import HNSWIndex  # noqa: E402
from src.mood_translator import available_moods, mood_to_query_vector  # noqa: E402

DATA_PATH = ROOT / "data" / "spotify_tracks.csv"
RESULTS_DIR = ROOT / "benchmarks" / "results"

st.set_page_config(page_title="Semantic Music Search", page_icon="🎧", layout="wide")


# --------------------------------------------------------------------------- #
# cached heavy objects
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading Spotify tracks…")
def _load(n: int):
    ds = load_spotify(str(DATA_PATH), limit=n)
    return ds


@st.cache_resource(show_spinner="Building HNSW graph index…")
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


@st.cache_resource(show_spinner="Preparing brute-force oracle…")
def _build_bf(n: int):
    ds = _load(n)
    return BruteForceIndex(ds.vectors)


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
    help="Larger = richer results but slower first-time index build.",
)
index_name = st.sidebar.radio(
    "Index", ["HNSW (graph)", "IVF (clustering)", "Brute force (exact)"], index=0
)
st.sidebar.markdown(
    f"**C++ hot-path:** {'✅ active' if native_available() else '➖ numpy fallback'}"
)

ds = _load(n)


def get_index():
    if index_name.startswith("HNSW"):
        idx, bt = _build_hnsw(n)
        return idx, bt, {"search": lambda q, k, allowed=None: idx.search(q, k, ef_search=80, allowed=allowed)}
    if index_name.startswith("IVF"):
        idx, bt = _build_ivf(n)
        return idx, bt, {"search": lambda q, k, allowed=None: idx.search(q, k, nprobe=16, allowed=allowed)}
    idx = _build_bf(n)
    return idx, 0.0, {"search": lambda q, k, allowed=None: idx.search(q, k, allowed=allowed)}


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
        idx, build_t, ops = get_index()
        qv, mq = mood_to_query_vector(mood, ds)

        t0 = time.perf_counter()
        ids, dists = ops["search"](qv, int(k), allowed=allowed_mask)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        st.info(f"**Why these tracks:** {mq.explanation(FEATURE_DESCRIPTIONS)}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Query latency", f"{elapsed_ms:.2f} ms")
        m2.metric("Index", index_name.split(" ")[0])
        m3.metric("Corpus", f"{ds.n:,} tracks")

        if len(ids):
            rows = ds.metadata.iloc[ids].copy()
            rows.insert(0, "similarity", np.round(1.0 / (1.0 + dists), 3))
            show_cols = [c for c in ("track_name", "artists", "track_genre", "popularity", "similarity") if c in rows.columns]
            st.dataframe(rows[show_cols], use_container_width=True, hide_index=True)

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
        chart_df = df[["mean_latency_ms", "recall_at_k", "label"]].set_index("mean_latency_ms")
        st.scatter_chart(df, x="mean_latency_ms", y="recall_at_k", color="index", size="memory_mb")
        st.dataframe(
            df[["index", "label", "recall_at_k", "mean_latency_ms", "p95_latency_ms", "qps", "build_time_s", "memory_mb"]],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Brute force is exact (recall 1.0) but scans everything. IVF and HNSW "
            "trade a little recall for large speedups; the knobs (nprobe / ef_search) "
            "slide each index along its own curve."
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
6. **C++ hot-path (pybind11)** — the per-hop distance kernel, ported to compiled
   code for a ~2.4× build/query speedup at identical recall.
7. **Mood translator** — a deterministic lexicon mapping natural-language mood
   to an audio-feature target (fully offline, self-explaining).

The **Benchmarks** tab shows the measured recall/latency/memory trade-offs that
justify each index choice.
"""
    )
    st.caption("Built by Shriya Kansal.")
