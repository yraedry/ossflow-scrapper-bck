"""Tests for ffprobe-based has_dubbing detection introduced for the
promote feature."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from api import library_refresh as lr


def test_has_dubbing_via_doblajes_sibling(tmp_path):
    """Pre-promotion: doblajes/<name>.mkv exists → has_dubbing without ffprobe."""
    season = tmp_path / "Season 01"
    season.mkdir()
    (season / "doblajes").mkdir()
    video = season / "S01E01.mp4"
    video.write_bytes(b"\0" * 32)
    (season / "doblajes" / "S01E01.mkv").write_bytes(b"\0" * 32)

    # ffprobe must NOT be called when the sidecar shortcut hits.
    with patch.object(lr, "_probe_audio_languages", return_value=[]) as probe:
        flags = lr._video_flags(video, cached=None)
    # _video_flags still runs the audio probe to populate the cache, but the
    # has_dubbing decision comes from the sidecar regardless.
    assert flags["has_dubbing"] is True
    assert flags["is_promoted"] is False
    # Probe runs once on first scan to seed the cache
    assert probe.call_count == 1


def test_has_dubbing_via_audio_probe_post_promotion(tmp_path):
    """Post-promotion: only <name>.mkv remains, but it has Spanish audio."""
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mkv"
    video.write_bytes(b"\0" * 32)

    with patch.object(lr, "_probe_audio_languages", return_value=["spa", "eng"]):
        flags = lr._video_flags(video, cached=None)
    assert flags["has_dubbing"] is True
    assert flags["is_promoted"] is True  # ES audio + no doblajes/ sidecar
    assert flags["audio_languages"] == ["spa", "eng"]


def test_no_dubbing_for_plain_original(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mp4"
    video.write_bytes(b"\0" * 32)

    with patch.object(lr, "_probe_audio_languages", return_value=["eng"]):
        flags = lr._video_flags(video, cached=None)
    assert flags["has_dubbing"] is False
    assert flags["is_promoted"] is False


def test_audio_probe_cached_when_fingerprint_unchanged(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mkv"
    video.write_bytes(b"\0" * 32)

    fingerprint = lr._file_fingerprint(video)
    cached = {
        "audio_fingerprint": list(fingerprint),
        "audio_languages": ["spa", "eng"],
    }

    with patch.object(lr, "_probe_audio_languages") as probe:
        flags = lr._video_flags(video, cached=cached)
    probe.assert_not_called()
    assert flags["audio_languages"] == ["spa", "eng"]
    assert flags["has_dubbing"] is True


def test_audio_probe_reruns_when_file_modified(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mkv"
    video.write_bytes(b"\0" * 32)

    cached = {
        "audio_fingerprint": [0, 1],  # bogus fingerprint → must invalidate
        "audio_languages": ["eng"],
    }

    with patch.object(lr, "_probe_audio_languages", return_value=["spa", "eng"]) as probe:
        flags = lr._video_flags(video, cached=cached)
    probe.assert_called_once()
    assert flags["audio_languages"] == ["spa", "eng"]
    assert flags["has_dubbing"] is True


def test_spanish_audio_detection_accepts_aliases():
    assert lr._has_spanish_audio(["spa"]) is True
    assert lr._has_spanish_audio(["es"]) is True
    assert lr._has_spanish_audio(["esp"]) is True
    assert lr._has_spanish_audio(["spanish"]) is True
    assert lr._has_spanish_audio(["eng", "fre"]) is False
    assert lr._has_spanish_audio([]) is False
    assert lr._has_spanish_audio(None) is False
    # Untagged streams (empty string or "und") do not false-positive
    assert lr._has_spanish_audio([""]) is False
