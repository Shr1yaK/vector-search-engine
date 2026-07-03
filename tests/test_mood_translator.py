import numpy as np

from src.vectors import FEATURE_COLUMNS, from_matrix
from src.mood_translator import (
    MOOD_LEXICON,
    available_moods,
    mood_to_query_vector,
    translate_mood,
)

_FI = {f: i for i, f in enumerate(FEATURE_COLUMNS)}


def _feat(mq, name):
    return mq.raw_vector[_FI[name]]


def test_chill_is_low_energy_and_energetic_is_high():
    chill = translate_mood("chill")
    energetic = translate_mood("energetic")
    assert _feat(chill, "energy") < 0.4
    assert _feat(energetic, "energy") > 0.7
    assert _feat(chill, "energy") < _feat(energetic, "energy")


def test_sad_is_low_valence_happy_is_high():
    assert _feat(translate_mood("sad"), "valence") < 0.3
    assert _feat(translate_mood("happy"), "valence") > 0.7


def test_workout_is_fast_tempo():
    assert _feat(translate_mood("workout"), "tempo") > 130


def test_study_is_instrumental():
    assert _feat(translate_mood("study focus"), "instrumentalness") > 0.6


def test_matched_terms_reported():
    mq = translate_mood("high energy gym workout")
    assert "gym" in mq.matched_terms and "workout" in mq.matched_terms


def test_multiword_phrase_alias():
    mq = translate_mood("music for a rainy day")
    assert "rainy day" in mq.matched_terms


def test_unknown_text_falls_back_to_neutral():
    mq = translate_mood("zzxqq nonsense tokens")
    assert mq.matched_terms == []
    # every feature is unconstrained -> neutral midpoints
    assert all(not v for v in mq.constrained.values())
    assert "neutral" in mq.explanation().lower() or "no mood" in mq.explanation().lower()


def test_combined_terms_are_weighted_average():
    # 'happy' (valence high) + 'sad' (valence low) should land in between.
    mq = translate_mood("happy sad")
    v = _feat(mq, "valence")
    assert 0.2 < v < 0.9


def test_projects_into_dataset_space():
    X = np.random.default_rng(0).random((100, len(FEATURE_COLUMNS))).astype(np.float32)
    ds = from_matrix(X)
    qv, mq = mood_to_query_vector("energetic party", ds)
    assert qv.shape == (len(FEATURE_COLUMNS),)
    assert np.isfinite(qv).all()


def test_lexicon_features_are_valid():
    valid = set(FEATURE_COLUMNS)
    for term, feats in MOOD_LEXICON.items():
        for f in feats:
            assert f in valid, f"{term} references unknown feature {f}"
    assert len(available_moods()) == len(MOOD_LEXICON)
