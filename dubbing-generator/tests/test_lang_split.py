"""Unit tests for the ES/EN language splitter and BJJ term list."""

from dubbing_generator.tts.bjj_en_terms import DEFAULT_BJJ_EN_TERMS


def test_default_bjj_terms_is_frozen_set():
    assert isinstance(DEFAULT_BJJ_EN_TERMS, frozenset)
    assert len(DEFAULT_BJJ_EN_TERMS) >= 25


def test_default_bjj_terms_are_lowercase():
    for term in DEFAULT_BJJ_EN_TERMS:
        assert term == term.lower(), f"non-lowercase term: {term!r}"


def test_default_bjj_terms_core_coverage():
    core = {"underhook", "overhook", "guard", "two on one", "lapel", "mount"}
    assert core <= DEFAULT_BJJ_EN_TERMS
