"""Lightweight per-instructional refresh of filesystem-derived flags.

The scan cache stores per-video booleans (has_subtitles_en, has_dubbing, ...)
captured during the last ``scan_library`` pass. Over time, these drift: user
deletes a ``.srt`` outside the UI, dubbing worker writes a new ``*_DOBLADO.mkv``,
poster gets added manually, etc.

Rather than re-walking the whole library (expensive on NAS/CIFS), this module
re-stats only the sidecar files for videos that are already known, mutating
the cached entries in place. Cost: a handful of ``stat()`` calls per video,
no ``iterdir``.

Two entry points:

- :func:`refresh_instructional_flags` — refresh one cached instructional.
- :func:`rediscover_instructional` — re-``iterdir`` a single folder to pick up
  newly added / removed videos in an existing instructional.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

import subprocess

from api.scan_cache import find_poster

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _probe_duration(path: Path) -> float | None:
    """Fast ffprobe to get only duration in seconds."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    return None
_DUB_SUFFIXES = ("_DOBLADO.mkv", "_DOBLADO.mp4")
_CHAPTER_RE = re.compile(r"S\d{2}E\d{2}")


def _video_flags(video_path: Path) -> dict[str, Any]:
    """Re-stat sidecars for one video. Returns fields to merge onto the cached entry."""
    folder = video_path.parent
    base = video_path.stem
    has_srt = (folder / f"{base}.en.srt").exists() or (folder / f"{base}.srt").exists()
    # Literal ES subtitles. The dubbing-adapted file is {base}.dub.es.srt
    # (different stem), so this check unambiguously matches only the literal.
    has_es_srt = (
        (folder / f"{base}.es.srt").exists()
        or (folder / f"{base}.ES.srt").exists()
        or (folder / f"{base}_ES.srt").exists()
        or (folder / f"{base}_ESP_DUB.srt").exists()
    )
    has_dubbed = any((folder / f"{base}{sfx}").exists() for sfx in _DUB_SUFFIXES)
    try:
        size_mb = round(video_path.stat().st_size / (1024 * 1024), 1)
    except OSError:
        size_mb = None
    return {
        "has_subtitles_en": has_srt,
        "has_subtitles_es": has_es_srt,
        "has_dubbing": has_dubbed,
        "is_chapter": bool(_CHAPTER_RE.search(video_path.name)),
        **({"size_mb": size_mb} if size_mb is not None else {}),
    }


def ensure_duration(video: dict[str, Any]) -> None:
    """Populate ``duration`` via ffprobe if missing from a cached video entry."""
    if video.get("duration") is not None:
        return
    vp = video.get("path")
    if vp:
        video["duration"] = _probe_duration(Path(vp))


def refresh_instructional_flags(item: dict[str, Any]) -> dict[str, Any]:
    """Mutate cached instructional in place: re-stat poster + per-video flags.

    Drops videos whose file no longer exists on disk. Returns the same dict.
    """
    folder_str = item.get("path")
    if not folder_str:
        return item
    folder = Path(folder_str)

    poster = find_poster(folder)
    item["has_poster"] = poster is not None
    item["poster_filename"] = poster.name if poster else None

    videos = item.get("videos") or []
    fresh_videos: list[dict[str, Any]] = []
    subtitled = dubbed = chapters = 0
    for v in videos:
        if not isinstance(v, dict):
            continue
        vp_str = v.get("path")
        if not vp_str:
            continue
        vp = Path(vp_str)
        if not vp.exists():
            continue
        v.update(_video_flags(vp))
        fresh_videos.append(v)
        if v.get("has_subtitles_en"):
            subtitled += 1
        if v.get("has_dubbing"):
            dubbed += 1
        if v.get("is_chapter"):
            chapters += 1

    item["videos"] = fresh_videos
    item["total_videos"] = len(fresh_videos)
    item["subtitled"] = subtitled
    item["dubbed"] = dubbed
    item["chapters_detected"] = chapters
    return item


def rediscover_instructional(item: dict[str, Any]) -> dict[str, Any]:
    """Re-walk the instructional folder to pick up added/removed video files.

    Preserves per-video fields already in cache for files that still exist
    (duration, etc.), adds entries for new videos, drops entries for deleted ones.
    Then delegates to :func:`refresh_instructional_flags` for sidecar re-stat.
    """
    folder_str = item.get("path")
    if not folder_str:
        return item
    folder = Path(folder_str)
    if not folder.exists() or not folder.is_dir():
        return item

    existing_by_path = {v.get("path"): v for v in (item.get("videos") or []) if isinstance(v, dict)}
    discovered: list[dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(folder):
        dp = Path(dirpath)
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            vp = dp / fn
            key = str(vp)
            cached = existing_by_path.get(key)
            if cached is not None:
                discovered.append(cached)
            else:
                discovered.append({
                    "filename": fn,
                    "path": key,
                    "duration": None,  # lazy — filled by ensure_duration later
                })

    item["videos"] = discovered
    return refresh_instructional_flags(item)


def refresh_many(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply :func:`refresh_instructional_flags` to a sequence of cached items."""
    out: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(refresh_instructional_flags(it))
    return out
