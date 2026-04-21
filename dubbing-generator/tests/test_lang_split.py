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


from dubbing_generator.tts.lang_split import split_by_language


def test_split_all_spanish():
    result = split_by_language("esto es todo en español", frozenset({"guard"}))
    assert result == [("es", "esto es todo en español")]


def test_split_empty_terms_returns_single_es_span():
    result = split_by_language("aplicamos un two on one", frozenset())
    assert result == [("es", "aplicamos un two on one")]


def test_split_embedded_english_term():
    terms = frozenset({"two on one", "guard"})
    result = split_by_language(
        "aplicamos un two on one desde la guard",
        terms,
    )
    assert result == [
        ("es", "aplicamos un "),
        ("en", "two on one"),
        ("es", " desde la "),
        ("en", "guard"),
    ]


def test_split_english_at_start_and_end():
    terms = frozenset({"underhook"})
    result = split_by_language("underhook y luego underhook", terms)
    assert result == [
        ("en", "underhook"),
        ("es", " y luego "),
        ("en", "underhook"),
    ]


def test_split_word_boundary_no_substring_match():
    """'underhooks' must match as itself, not produce 'underhook' + 's'."""
    terms = frozenset({"underhook", "underhooks"})
    result = split_by_language("los underhooks son clave", terms)
    assert result == [
        ("es", "los "),
        ("en", "underhooks"),
        ("es", " son clave"),
    ]


def test_split_case_insensitive_match_preserves_original_casing():
    terms = frozenset({"guard"})
    result = split_by_language("la Guard cerrada", terms)
    assert result == [
        ("es", "la "),
        ("en", "Guard"),
        ("es", " cerrada"),
    ]


def test_split_prefers_longer_match():
    """'two on one' must win over 'one' when both are in the set."""
    terms = frozenset({"one", "two on one"})
    result = split_by_language("aplicamos two on one", terms)
    assert result == [
        ("es", "aplicamos "),
        ("en", "two on one"),
    ]


def test_split_empty_string():
    assert split_by_language("", frozenset({"guard"})) == []
