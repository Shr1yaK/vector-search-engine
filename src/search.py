"""Command-line semantic search — the app's core without the browser.

    python -m src.search "chill rainy day coding music"
    python -m src.search "high energy workout" --index ivf --k 5 --n 30000
    python -m src.search "acoustic study" --genre acoustic --genre ambient

Loads the corpus, builds the chosen index, translates the mood string to a query
vector, and prints the matching tracks. Fully offline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .brute_force import BruteForceIndex
from .hnsw_index import HNSWIndex
from .ivf_index import IVFIndex
from .ivfpq_index import IVFPQIndex
from .mood_translator import mood_to_query_vector
from .vectors import load_spotify

DATA = Path(__file__).resolve().parent.parent / "data" / "spotify_tracks.csv"


def _build(index: str, vectors: np.ndarray):
    """Return a search callable ``(query, k, allowed) -> (ids, dists)``."""
    nlist = int(np.sqrt(len(vectors))) or 1
    if index == "brute":
        idx = BruteForceIndex(vectors)
        return lambda q, k, a: idx.search(q, k, allowed=a)
    if index == "ivf":
        idx = IVFIndex(nlist=nlist, nprobe=16, random_state=0).build(vectors)
        return lambda q, k, a: idx.search(q, k, nprobe=16, allowed=a)
    if index == "ivfpq":
        idx = IVFPQIndex(nlist=nlist, m=3, ksub=64, nprobe=16, random_state=0).build(
            vectors, keep_vectors=True
        )
        return lambda q, k, a: idx.search(q, k, nprobe=16, allowed=a, rerank=50)
    idx = HNSWIndex(M=16, ef_construction=200, random_state=0).build(vectors)
    return lambda q, k, a: idx.search(q, k, ef_search=80, allowed=a)


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic music search from the CLI.")
    ap.add_argument("mood", help="natural-language mood, e.g. 'sad rainy night'")
    ap.add_argument("--index", choices=["hnsw", "ivf", "ivfpq", "brute"], default="hnsw")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=15000, help="corpus size cap")
    ap.add_argument("--genre", action="append", default=[], help="restrict to genre (repeatable)")
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"dataset not found at {args.data}; run scripts/fetch_data.py")

    ds = load_spotify(args.data, limit=args.n)
    t0 = time.perf_counter()
    search = _build(args.index, ds.vectors)
    build_ms = (time.perf_counter() - t0) * 1000

    allowed = None
    if args.genre and ds.metadata is not None and "track_genre" in ds.metadata.columns:
        allowed = ds.metadata["track_genre"].isin(args.genre).to_numpy()
        if not allowed.any():
            print(f"(no tracks in genres {args.genre} within the first {ds.n} rows)")

    qv, mq = mood_to_query_vector(args.mood, ds)
    t0 = time.perf_counter()
    ids, dists = search(qv, args.k, allowed)
    query_ms = (time.perf_counter() - t0) * 1000

    print(f'\n"{args.mood}"  [{args.index}, {ds.n:,} tracks, '
          f"build {build_ms:.0f} ms, query {query_ms:.2f} ms]")
    print(f"why: {mq.explanation()}\n")
    if len(ids) == 0:
        print("no results.")
        return
    for rank, (i, d) in enumerate(zip(ids, dists), 1):
        r = ds.metadata.iloc[i]
        sim = 1.0 / (1.0 + float(d))
        print(f"{rank:2d}. {r['track_name'][:40]:40s}  {r['artists'][:24]:24s}"
              f"  [{r['track_genre']}]  sim={sim:.3f}")


if __name__ == "__main__":
    main()
