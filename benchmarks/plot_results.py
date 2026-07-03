"""Render the recall/latency trade-off figure from saved benchmark runs.

Produces `docs/img/benchmark_tradeoff.png`: two panels (the d=9 Spotify audio
data and the d=128 synthetic embeddings) plotting recall@10 against mean query
latency. Each index is one series; the approximate indexes trace a curve as
their knob (nprobe / ef_search) sweeps, brute force is a single exact point.
Up-and-to-the-left is better.

    python benchmarks/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "img" / "benchmark_tradeoff.png"

# Validated categorical palette (blue / aqua / red) + chart chrome from the
# design system. aqua is sub-3:1 on the surface, so every series is also
# direct-labeled and given a distinct marker (the relief rule).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
STYLE = {
    "brute_force": {"color": "#2a78d6", "marker": "o", "label": "brute force (exact)"},
    "ivf": {"color": "#1baf7a", "marker": "s", "label": "IVF"},
    "hnsw": {"color": "#e34948", "marker": "^", "label": "HNSW"},
}


def _load(tag: str) -> list[dict]:
    path = RESULTS / f"benchmark_{tag}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run the benchmark for tag '{tag}' first")
    return json.loads(path.read_text())


def _panel(ax, rows: list[dict], title: str) -> None:
    ax.set_facecolor(SURFACE)
    handles = []
    for kind, st in STYLE.items():
        pts = sorted(
            [(r["mean_latency_ms"], r["recall_at_k"]) for r in rows if r["index"] == kind]
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        if kind == "brute_force":
            h = ax.scatter(xs, ys, color=st["color"], marker=st["marker"], s=90,
                           zorder=5, edgecolor=SURFACE, linewidth=1.5, label=st["label"])
        else:
            (h,) = ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=8,
                           linewidth=2, zorder=4, markeredgecolor=SURFACE,
                           markeredgewidth=1.2, label=st["label"])
        handles.append(h)

    # Legend in the empty lower-right region (recall rises fast, leaving space).
    # Distinct markers + color both carry identity (relief for the sub-3:1 aqua).
    leg = ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9,
                    labelcolor=INK, edgecolor=GRID, facecolor=SURFACE, framealpha=0.95)
    leg.get_frame().set_linewidth(0.6)

    ax.set_xscale("log")
    ax.set_xlabel("mean query latency (ms, log scale)", color=MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.grid(True, which="both", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d9 = _load("n20k")
    hd = _load("highdim_d128")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    _panel(ax1, d9, "Spotify audio features  (d = 9)  —  IVF wins")
    _panel(ax2, hd, "Synthetic embeddings  (d = 128)  —  HNSW wins")
    ax1.set_ylabel("recall@10  (vs exact)", color=MUTED, fontsize=9)
    fig.suptitle("Recall vs. latency — up and to the left is better",
                 color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
