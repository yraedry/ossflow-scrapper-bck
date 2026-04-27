"""Fish Audio S2-Pro local TTS via in-container HTTP server.

Connection model: one ``httpx.Client`` per synthesizer instance. We POST
one multipart form per phrase; the reference WAV + transcript stay
constant across the episode.

Resilience:
- On HTTP error: log + return 200 ms silence so the pipeline survives.
- After ``_BREAKER_THRESHOLD`` consecutive errors: short-circuit for
  ``_BREAKER_COOLDOWN_S`` to avoid 150× hammering a dead server.
- Resets on first success.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx
from pydub import AudioSegment

from ..config import DubbingConfig

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_S = 60.0


class SynthesizerS2Pro:
    """Generate speech via the local s2.cpp HTTP server."""

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config
        self._client = httpx.Client(
            base_url=f"http://{config.s2_server_host}:{config.s2_server_port}",
            timeout=config.s2_request_timeout,
        )
        self._consecutive_failures = 0
        self._breaker_open_until: float = 0.0

    @property
    def sample_rate(self) -> int:
        return 44100  # s2.cpp emits 44.1 kHz

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        # Best-effort cleanup; pipeline should call close() explicitly.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def generate(
        self,
        text: str,
        reference_wav: Path,
        speed: Optional[float] = None,
    ) -> AudioSegment:
        """Synthesize *text*. ``reference_wav`` and ``speed`` are accepted
        for protocol parity with the other synthesizers but are unused —
        s2.cpp does not expose a speed parameter (the pipeline absorbs
        cadence post-stretch instead)."""
        if speed is not None and speed != 1.0:
            logger.debug(
                "S2-Pro ignores speed=%.3f (use post-stretch instead)", speed,
            )
        _ = reference_wav
        if not text.strip():
            return AudioSegment.silent(duration=100)

        ref_path = Path(self.cfg.s2_ref_audio_path)
        if not ref_path.exists():
            logger.error("S2-Pro ref audio missing: %s", ref_path)
            return AudioSegment.silent(duration=200)

        # Circuit breaker check.
        now = time.monotonic()
        if now < self._breaker_open_until:
            logger.warning(
                "S2-Pro circuit breaker open (cooldown %.0f s remaining)",
                self._breaker_open_until - now,
            )
            return AudioSegment.silent(duration=200)

        params = {
            "max_new_tokens": self.cfg.s2_max_tokens,
            "temperature": self.cfg.s2_temperature,
            "top_p": self.cfg.s2_top_p,
            "top_k": self.cfg.s2_top_k,
        }

        try:
            with ref_path.open("rb") as fh:
                resp = self._client.post(
                    "/generate",
                    files={"reference": (ref_path.name, fh, "audio/wav")},
                    data={
                        "text": text,
                        "reference_text": self.cfg.s2_ref_text,
                        "params": json.dumps(params),
                    },
                )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._breaker_open_until = now + _BREAKER_COOLDOWN_S
                logger.error(
                    "S2-Pro circuit breaker tripped (%d failures); "
                    "cooling down %.0f s", self._consecutive_failures,
                    _BREAKER_COOLDOWN_S,
                )
                # Reset counter on trip so post-cooldown probe gets a clean
                # baseline; a single failure after cooldown re-trips immediately.
                self._consecutive_failures = 0
            logger.warning("S2-Pro synthesize failed (text=%r): %s",
                           text[:80], exc)
            return AudioSegment.silent(duration=200)

        self._consecutive_failures = 0
        try:
            return AudioSegment.from_file(io.BytesIO(resp.content), format="wav")
        except Exception as exc:  # noqa: BLE001
            logger.warning("S2-Pro returned undecodable audio (%r): %s",
                           text[:80], exc)
            return AudioSegment.silent(duration=200)
