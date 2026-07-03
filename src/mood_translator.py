"""Translate natural-language mood text into an audio-feature query vector.

This is the semantic layer that sits on top of the index: a human types
"chill rainy day coding music" and we need a point in the same feature space as
the corpus so the HNSW/IVF index can find neighbors.

Design choice: **fully offline, deterministic, no API.** Instead of calling an
LLM, we use a curated *mood lexicon* — each descriptive term contributes target
values (and a confidence weight) for a subset of the audio features. Overlapping
terms are combined by a weighted average per feature. This keeps the app fast,
free, reproducible, and unit-testable, and it makes the mapping fully
inspectable — the "why these tracks" explanation is generated from the exact
terms that fired, not a black box.

The output is expressed on the *raw* feature scale (danceability in [0,1],
tempo in BPM, ...). Callers normalize it with the corpus
:class:`~src.vectors.Normalizer` before querying the index, so the query lives
in exactly the same space as the indexed vectors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .vectors import FEATURE_COLUMNS, Dataset


# Each lexicon entry maps a mood term to (feature -> (target_value, weight)).
# Targets are on the raw feature scale. Weights encode how strongly the term
# constrains that feature (a mood word may pin some features and leave others
# free). Feature keys must be a subset of vectors.FEATURE_COLUMNS.
#
# This lexicon is deliberately hand-tuned from the semantics of Spotify's audio
# features (see FEATURE_DESCRIPTIONS) — it is the domain knowledge of the app.
MOOD_LEXICON: dict[str, dict[str, tuple[float, float]]] = {
    # --- energy / intensity --------------------------------------------- #
    "chill": {"energy": (0.25, 1.0), "valence": (0.5, 0.4), "tempo": (95, 0.6), "acousticness": (0.6, 0.7)},
    "relaxing": {"energy": (0.2, 1.0), "acousticness": (0.7, 0.8), "tempo": (85, 0.6)},
    "calm": {"energy": (0.18, 1.0), "acousticness": (0.7, 0.7), "tempo": (80, 0.6)},
    "mellow": {"energy": (0.3, 0.9), "valence": (0.45, 0.4), "acousticness": (0.6, 0.6)},
    "energetic": {"energy": (0.9, 1.0), "tempo": (135, 0.7), "danceability": (0.7, 0.6)},
    "intense": {"energy": (0.92, 1.0), "loudness": (-4, 0.6), "tempo": (140, 0.6)},
    "hype": {"energy": (0.9, 1.0), "danceability": (0.8, 0.8), "valence": (0.7, 0.5)},
    "upbeat": {"energy": (0.75, 0.9), "valence": (0.8, 0.9), "tempo": (125, 0.6), "danceability": (0.7, 0.6)},
    # --- valence / emotion ---------------------------------------------- #
    "happy": {"valence": (0.9, 1.0), "energy": (0.7, 0.5), "danceability": (0.65, 0.4)},
    "joyful": {"valence": (0.92, 1.0), "energy": (0.75, 0.5)},
    "sad": {"valence": (0.12, 1.0), "energy": (0.3, 0.6), "acousticness": (0.6, 0.5)},
    "melancholy": {"valence": (0.2, 1.0), "energy": (0.35, 0.6), "acousticness": (0.6, 0.6)},
    "moody": {"valence": (0.3, 0.8), "energy": (0.4, 0.5)},
    "romantic": {"valence": (0.6, 0.7), "energy": (0.4, 0.6), "acousticness": (0.55, 0.6)},
    "angry": {"valence": (0.2, 0.8), "energy": (0.9, 0.9), "loudness": (-4, 0.6)},
    "dark": {"valence": (0.2, 0.9), "energy": (0.55, 0.4)},
    "euphoric": {"valence": (0.9, 1.0), "energy": (0.85, 0.8), "danceability": (0.7, 0.5)},
    "nostalgic": {"valence": (0.5, 0.6), "energy": (0.4, 0.5), "acousticness": (0.5, 0.5)},
    # --- activity / context --------------------------------------------- #
    "workout": {"energy": (0.9, 1.0), "tempo": (140, 0.8), "danceability": (0.7, 0.6), "valence": (0.65, 0.4)},
    "gym": {"energy": (0.9, 0.9), "tempo": (138, 0.7), "danceability": (0.72, 0.6)},
    "running": {"energy": (0.85, 0.9), "tempo": (160, 0.9)},
    "party": {"danceability": (0.85, 1.0), "energy": (0.85, 0.9), "valence": (0.8, 0.7), "tempo": (122, 0.5)},
    "dance": {"danceability": (0.9, 1.0), "energy": (0.75, 0.6), "tempo": (124, 0.6)},
    "study": {"energy": (0.25, 0.9), "instrumentalness": (0.8, 0.9), "speechiness": (0.04, 0.7), "acousticness": (0.6, 0.5)},
    "coding": {"energy": (0.35, 0.8), "instrumentalness": (0.75, 0.9), "speechiness": (0.04, 0.7)},
    "focus": {"energy": (0.3, 0.9), "instrumentalness": (0.8, 0.9), "speechiness": (0.04, 0.7)},
    "sleep": {"energy": (0.1, 1.0), "acousticness": (0.85, 0.9), "instrumentalness": (0.7, 0.7), "tempo": (70, 0.7)},
    "meditation": {"energy": (0.1, 1.0), "acousticness": (0.85, 0.9), "instrumentalness": (0.85, 0.9), "tempo": (65, 0.6)},
    "driving": {"energy": (0.7, 0.7), "valence": (0.65, 0.5), "tempo": (120, 0.4)},
    "rainy": {"energy": (0.3, 0.7), "valence": (0.4, 0.6), "acousticness": (0.6, 0.6)},
    "morning": {"energy": (0.5, 0.6), "valence": (0.7, 0.6), "acousticness": (0.5, 0.4)},
    "night": {"energy": (0.5, 0.5), "valence": (0.45, 0.4)},
    "summer": {"valence": (0.8, 0.8), "energy": (0.75, 0.6), "danceability": (0.7, 0.5)},
    # --- sonic texture -------------------------------------------------- #
    "acoustic": {"acousticness": (0.9, 1.0), "energy": (0.35, 0.5)},
    "instrumental": {"instrumentalness": (0.9, 1.0), "speechiness": (0.04, 0.6)},
    "vocal": {"instrumentalness": (0.05, 0.9), "speechiness": (0.1, 0.3)},
    "live": {"liveness": (0.8, 0.9)},
    "loud": {"loudness": (-3, 0.9), "energy": (0.8, 0.5)},
    "quiet": {"loudness": (-20, 0.8), "energy": (0.25, 0.5)},
    "fast": {"tempo": (160, 0.9), "energy": (0.75, 0.4)},
    "slow": {"tempo": (70, 0.9), "energy": (0.3, 0.5)},
}

# A few multi-word phrases resolve to a canonical single term above.
PHRASE_ALIASES: dict[str, str] = {
    "rainy day": "rainy",
    "work out": "workout",
    "chilled out": "chill",
    "laid back": "chill",
    "feel good": "happy",
    "pick me up": "upbeat",
    "background music": "instrumental",
    "deep focus": "focus",
}

# Neutral fallback midpoints (raw scale) used for features no term constrained.
_NEUTRAL: dict[str, float] = {
    "danceability": 0.55,
    "energy": 0.5,
    "loudness": -8.0,
    "speechiness": 0.08,
    "acousticness": 0.4,
    "instrumentalness": 0.2,
    "liveness": 0.2,
    "valence": 0.5,
    "tempo": 118.0,
}


@dataclass
class MoodQuery:
    """The parsed result of a mood string."""

    text: str
    raw_vector: np.ndarray                      # (d,) on the raw feature scale
    matched_terms: list[str]                    # lexicon terms that fired
    feature_targets: dict[str, float]           # feature -> resolved raw target
    constrained: dict[str, bool] = field(default_factory=dict)  # term-driven?

    def explanation(self, feature_descriptions: dict[str, str] | None = None) -> str:
        """Human-readable account of how the text became a query."""
        if not self.matched_terms:
            return ("No mood keywords recognized — falling back to a neutral "
                    "profile. Try words like 'chill', 'energetic', 'sad', "
                    "'workout', 'study', 'party'.")
        terms = ", ".join(f"'{t}'" for t in self.matched_terms)
        # Report the most strongly constrained features.
        pinned = [f for f, on in self.constrained.items() if on]
        bits = []
        for f in ("energy", "valence", "danceability", "acousticness",
                  "instrumentalness", "tempo"):
            if f in pinned:
                v = self.feature_targets[f]
                if f == "tempo":
                    bits.append(f"tempo≈{v:.0f} BPM")
                else:
                    bits.append(f"{f}≈{v:.2f}")
        return f"Matched {terms} → target " + ", ".join(bits) + "."


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def translate_mood(text: str) -> MoodQuery:
    """Turn a mood string into a :class:`MoodQuery` on the raw feature scale.

    Terms are matched (including a few multi-word phrases), each contributing
    weighted targets per feature; features are then resolved as the
    weight-weighted average of their contributions, or the neutral midpoint if
    nothing fired.
    """
    lowered = text.lower()
    matched: list[str] = []

    # Resolve multi-word phrase aliases first, then single tokens.
    fired_terms: list[str] = []
    for phrase, canon in PHRASE_ALIASES.items():
        if phrase in lowered:
            fired_terms.append(canon)
            matched.append(phrase)
    for tok in _tokenize(text):
        if tok in MOOD_LEXICON:
            fired_terms.append(tok)
            if tok not in matched:
                matched.append(tok)

    # Accumulate weighted sums per feature.
    num: dict[str, float] = {f: 0.0 for f in FEATURE_COLUMNS}
    den: dict[str, float] = {f: 0.0 for f in FEATURE_COLUMNS}
    for term in fired_terms:
        for feat, (val, weight) in MOOD_LEXICON[term].items():
            num[feat] += val * weight
            den[feat] += weight

    feature_targets: dict[str, float] = {}
    constrained: dict[str, bool] = {}
    for f in FEATURE_COLUMNS:
        if den[f] > 0:
            feature_targets[f] = num[f] / den[f]
            constrained[f] = True
        else:
            feature_targets[f] = _NEUTRAL[f]
            constrained[f] = False

    raw_vector = np.array([feature_targets[f] for f in FEATURE_COLUMNS], dtype=np.float32)
    return MoodQuery(
        text=text,
        raw_vector=raw_vector,
        matched_terms=matched,
        feature_targets=feature_targets,
        constrained=constrained,
    )


def mood_to_query_vector(text: str, dataset: Dataset) -> tuple[np.ndarray, MoodQuery]:
    """Translate mood text and project it into the dataset's normalized space.

    Returns ``(normalized_query_vector, MoodQuery)`` ready to hand to any index.
    """
    mq = translate_mood(text)
    normalized = dataset.normalizer.transform(mq.raw_vector.reshape(1, -1))[0]
    return normalized, mq


def available_moods() -> list[str]:
    """All recognized single-word mood terms (for UI hints/autocomplete)."""
    return sorted(MOOD_LEXICON.keys())
