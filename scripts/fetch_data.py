"""Download the Spotify Tracks dataset into ``data/spotify_tracks.csv``.

The CSV (~20 MB, 114k tracks) is intentionally kept out of git. This script
fetches it from a public mirror so the project is reproducible with one command:

    python scripts/fetch_data.py

Source: the "Spotify Tracks Dataset" (114k tracks, 125 genres, Spotify audio
features), mirrored on the Hugging Face Hub.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
DEST = Path(__file__).resolve().parent.parent / "data" / "spotify_tracks.csv"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already present: {DEST} ({DEST.stat().st_size/1e6:.1f} MB)")
        return 0
    print(f"Downloading Spotify tracks dataset ->\n  {DEST}")
    try:
        def _progress(block, block_size, total):
            done = block * block_size
            if total > 0:
                pct = min(100, done * 100 // total)
                sys.stdout.write(f"\r  {pct:3d}%  ({done/1e6:5.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(URL, DEST, _progress)
        print(f"\nDone: {DEST.stat().st_size/1e6:.1f} MB")
    except Exception as exc:  # pragma: no cover
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print(
            "You can also grab it manually from Kaggle "
            "('Spotify Tracks Dataset', 114k tracks) and save it as "
            f"{DEST}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
