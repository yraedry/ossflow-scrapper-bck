"""Persistent settings for the BJJ Instructional Processor.

Settings are stored in the unified SQLite DB (``bjj_service_kit.db``).
The DB path defaults to ``/data/db/bjj.db`` but can be overridden via
``BJJ_DB_PATH``. A legacy JSON file at ``CONFIG_DIR/settings.json`` is
auto-imported on first run if present, then renamed to ``settings.json.bak``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from bjj_service_kit.db import init_db, session_scope
from bjj_service_kit.db.models import Setting

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", "/data/config"))
LEGACY_SETTINGS_FILE = CONFIG_DIR / "settings.json"

_DEFAULTS: dict[str, Any] = {
    "library_path": "",
    "voice_profile_default": None,
    "processing_defaults": {
        "chapters": {"dry_run": False, "verbose": True},
        "subtitles": {"verbose": True},
        "translate": {},
        "dubbing": {"use_model_voice": False},
    },
    "custom_prompts": {},
    "telegram_api_id": None,
    "telegram_api_hash": None,
    "deepl_api_key": None,
    "openai_api_key": None,
    "translation_provider": "openai",
    "translation_model": "gpt-4o-mini",
    "translation_fallback_provider": "deepl",
    # Industry-standard iso-synchronous translation: the translator compacts
    # each ES line to fit the SRT slot so TTS comes out on-time without audio
    # stretch. Only works with OpenAI provider (budget-aware prompt).
    "translation_dubbing_mode": True,
    "translation_dubbing_cps": 16.0,   # chars/sec: 16 fills slots with small overshoot that stretch absorbs (14 left silence)
    "author_aliases": {},
}

_TELEGRAM_HASH_RE = re.compile(r"^[0-9a-fA-F]{32}$")

_initialized = False


def _ensure_initialized() -> None:
    """Ensure DB schema exists and import legacy JSON once."""
    global _initialized
    if _initialized:
        return
    try:
        init_db()
    except Exception as exc:
        log.warning("init_db failed: %s", exc)
    _maybe_import_legacy_json()
    _initialized = True


def _maybe_import_legacy_json() -> None:
    if not LEGACY_SETTINGS_FILE.exists():
        return
    try:
        data = json.loads(LEGACY_SETTINGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        with session_scope() as s:
            # Only import keys absent from DB (idempotent)
            existing = {row.key for row in s.query(Setting).all()}
            for k, v in data.items():
                if k in existing:
                    continue
                s.add(Setting(key=k, value=json.dumps(v, ensure_ascii=False)))
        backup = LEGACY_SETTINGS_FILE.with_suffix(".json.bak")
        LEGACY_SETTINGS_FILE.rename(backup)
        log.info("Imported legacy settings.json → DB (backup at %s)", backup)
    except Exception as exc:
        log.warning("Legacy settings import failed: %s", exc)


def load_settings() -> dict[str, Any]:
    """Return settings merged with defaults."""
    _ensure_initialized()
    merged = dict(_DEFAULTS)
    try:
        with session_scope() as s:
            for row in s.query(Setting).all():
                try:
                    merged[row.key] = json.loads(row.value)
                except json.JSONDecodeError:
                    merged[row.key] = row.value
    except Exception as exc:
        log.warning("load_settings failed, returning defaults: %s", exc)
    return merged


def save_settings(data: dict[str, Any]) -> None:
    """Persist settings to DB (upsert per key)."""
    _ensure_initialized()
    with session_scope() as s:
        for k, v in data.items():
            payload = json.dumps(v, ensure_ascii=False)
            row = s.get(Setting, k)
            if row is None:
                s.add(Setting(key=k, value=payload))
            else:
                row.value = payload
    log.info("Settings saved to DB")


def get_library_path() -> Optional[str]:
    settings = load_settings()
    lp = settings.get("library_path", "")
    return lp if lp else None


def get_setting(key: str) -> Any:
    """Read a single settings value (merged with defaults)."""
    return load_settings().get(key)


# ---------------------------------------------------------------------------
# Routes (API contract unchanged)
# ---------------------------------------------------------------------------

@router.get("")
async def get_settings():
    return load_settings()


@router.put("")
async def put_settings(request: Request):
    body = await request.json()

    if not isinstance(body, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=422)

    current = load_settings()

    if "library_path" in body:
        lp = body["library_path"]
        if not isinstance(lp, str):
            return JSONResponse({"error": "library_path must be a string"}, status_code=422)
        current["library_path"] = lp

    if "voice_profile_default" in body:
        current["voice_profile_default"] = body["voice_profile_default"]

    if "processing_defaults" in body:
        pd = body["processing_defaults"]
        if not isinstance(pd, dict):
            return JSONResponse({"error": "processing_defaults must be a JSON object"}, status_code=422)
        current["processing_defaults"] = pd

    if "telegram_api_id" in body:
        tid = body["telegram_api_id"]
        if tid is not None and not (isinstance(tid, int) and not isinstance(tid, bool)):
            return JSONResponse({"error": "telegram_api_id must be an integer or null"}, status_code=422)
        current["telegram_api_id"] = tid

    if "telegram_api_hash" in body:
        th = body["telegram_api_hash"]
        if th is not None:
            if not isinstance(th, str) or not _TELEGRAM_HASH_RE.match(th):
                return JSONResponse({"error": "telegram_api_hash must be a 32-char hex string or null"}, status_code=422)
        current["telegram_api_hash"] = th

    if "deepl_api_key" in body:
        dk = body["deepl_api_key"]
        if dk is not None and not isinstance(dk, str):
            return JSONResponse({"error": "deepl_api_key must be a string or null"}, status_code=422)
        current["deepl_api_key"] = dk.strip() if isinstance(dk, str) else dk

    if "openai_api_key" in body:
        ok = body["openai_api_key"]
        if ok is not None and not isinstance(ok, str):
            return JSONResponse({"error": "openai_api_key must be a string or null"}, status_code=422)
        current["openai_api_key"] = ok.strip() if isinstance(ok, str) else ok

    for k in ("translation_provider", "translation_model", "translation_fallback_provider"):
        if k in body:
            v = body[k]
            if v is not None and not isinstance(v, str):
                return JSONResponse({"error": f"{k} must be a string or null"}, status_code=422)
            current[k] = v.strip() if isinstance(v, str) else v

    if "author_aliases" in body:
        aa = body["author_aliases"]
        if not isinstance(aa, dict):
            return JSONResponse({"error": "author_aliases must be a JSON object"}, status_code=422)
        cleaned: dict[str, str] = {}
        for k, v in aa.items():
            if not isinstance(k, str) or not isinstance(v, str):
                return JSONResponse({"error": "author_aliases keys and values must be strings"}, status_code=422)
            k2, v2 = k.strip(), v.strip()
            if k2 and v2:
                cleaned[k2] = v2
        current["author_aliases"] = cleaned

    if "custom_prompts" in body:
        cp = body["custom_prompts"]
        if not isinstance(cp, dict):
            return JSONResponse({"error": "custom_prompts must be a JSON object"}, status_code=422)
        current["custom_prompts"] = cp

    save_settings(current)
    return current
