"""Proxy endpoints to dubbing-generator for debug/analyze."""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.paths import to_container_path
from api.pipeline import _load_voice_profile_for_path
from api.settings import get_library_path

log = logging.getLogger(__name__)

DUBBING_URL = os.environ.get("DUBBING_URL", "http://localhost:8003")

router = APIRouter(prefix="/api/dubbing", tags=["dubbing"])


class AnalyzeBody(BaseModel):
    video_path: str
    srt_path: Optional[str] = None
    synthesize: bool = False
    max_phrases: Optional[int] = None
    voice_profile: Optional[str] = None


def _translate(host_path: str) -> str:
    lib = get_library_path()
    if not lib:
        return host_path
    try:
        return to_container_path(host_path, lib)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


async def _post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(f"{DUBBING_URL}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"dubbing-generator: {exc}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


@router.get("/voices")
async def list_voices() -> dict:
    """List ES voice reference WAVs available inside dubbing-generator /voices."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{DUBBING_URL}/voices")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"dubbing-generator: {exc}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


@router.post("/analyze")
async def analyze_dubbing(body: AnalyzeBody) -> dict:
    payload: dict = {
        "video_path": _translate(body.video_path),
        "synthesize": body.synthesize,
    }
    if body.srt_path:
        payload["srt_path"] = _translate(body.srt_path)
    if body.max_phrases is not None:
        payload["max_phrases"] = body.max_phrases
    vp = body.voice_profile or _load_voice_profile_for_path(body.video_path)
    if vp:
        payload["voice_profile"] = vp
    # Synthesis runs TTS on N phrases → very slow; give generous timeout
    timeout = 1200.0 if body.synthesize else 60.0
    return await _post("/analyze", payload, timeout=timeout)
