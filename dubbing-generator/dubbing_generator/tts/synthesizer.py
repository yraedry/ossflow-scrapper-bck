"""Chatterbox Multilingual TTS wrapper — voice cloning with 23 languages."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from pydub import AudioSegment

from ..config import DubbingConfig

logger = logging.getLogger(__name__)


class Synthesizer:
    """Generate speech from text using Chatterbox Multilingual TTS.

    Clones the instructor's voice from a reference WAV and outputs Spanish
    (or any of 23 supported languages). Handles long text by splitting at
    sentence boundaries and cross-fading the resulting chunks.
    """

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config
        self._model = None
        self._sr: int | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Lazy-load the Chatterbox multilingual model onto the best device."""
        if self._model is not None:
            return

        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Chatterbox Multilingual TTS on %s", device)
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self._sr = int(self._model.sr)
        logger.info("Chatterbox loaded (sr=%d Hz)", self._sr)

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def sample_rate(self) -> int:
        if self._sr is None:
            self.load_model()
        return self._sr  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        reference_wav: Path,
        speed: float | None = None,
    ) -> AudioSegment:
        """Synthesize *text* with voice cloned from *reference_wav*.

        `speed` is mapped to Chatterbox's `cfg_weight` (lower = slower/calmer
        pacing, higher = tighter delivery). We keep the XTTS-style API so the
        rest of the pipeline (drift corrector, stretcher) remains untouched.
        """
        if speed is None:
            speed = self.cfg.tts_speed

        parts = self._split_long_text(text, self.cfg.tts_char_limit)

        segments: list[AudioSegment] = []
        for part in parts:
            seg = self._synthesize_chunk(part, reference_wav, speed)
            segments.append(seg)

        if not segments:
            return AudioSegment.silent(duration=100)

        result = segments[0]
        for seg in segments[1:]:
            xfade = min(self.cfg.tts_crossfade_ms, len(result), len(seg))
            if xfade > 0:
                result = result.append(seg, crossfade=xfade)
            else:
                result += seg

        # Normalize to -18 dBFS for consistent loudness across phrases
        result = self._normalize(result, target_dbfs=-18.0)
        return result

    @staticmethod
    def _normalize(audio: AudioSegment, target_dbfs: float = -18.0) -> AudioSegment:
        if audio.dBFS == float("-inf"):
            return audio
        delta = target_dbfs - audio.dBFS
        delta = max(-12.0, min(12.0, delta))
        return audio.apply_gain(delta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _speed_to_cfg_weight(self, speed: float) -> float:
        """Map speed to Chatterbox cfg_weight.

        Docs recomiendan 0.4-0.6 para voz natural. <0.35 desata tanto la
        prosodia que se oye "pensando" mid-frase (robótico). >0.6 copia
        cadencia de ref EN. Centrado en 0.45 para ref EN → texto ES.
        """
        s = max(0.85, min(1.25, speed))
        # Linear map: 0.85→0.35, 1.0→0.45, 1.25→0.60
        cfg = 0.45 + (s - 1.0) * 0.6
        return max(0.35, min(0.60, cfg))

    def _synthesize_chunk(
        self,
        text: str,
        reference_wav: Path,
        speed: float,
    ) -> AudioSegment:
        """Synthesize one short chunk using Chatterbox.

        Chatterbox ES tiene un bug conocido: tokens comunes (puntuación,
        espacios) disparan falsos positivos de repetition detection →
        forcing EOS → TTS truncado. Detectamos el corte comparando la
        duración generada contra el mínimo esperado por longitud de texto
        (~avg_ms_per_char) y reintentamos con temperature más alta.
        """
        import torchaudio as ta

        cfg_weight = self._speed_to_cfg_weight(speed)

        expected_ms = len(text) * self.cfg.avg_ms_per_char
        min_acceptable_ms = expected_ms * 0.60

        attempts = [
            (self.cfg.tts_temperature, self.cfg.tts_repetition_penalty),
            (min(self.cfg.tts_temperature + 0.15, 1.0), self.cfg.tts_repetition_penalty + 0.10),
            (min(self.cfg.tts_temperature + 0.30, 1.1), self.cfg.tts_repetition_penalty + 0.20),
        ]

        best_segment: AudioSegment | None = None
        best_ms = 0

        for attempt_idx, (temp, rep_pen) in enumerate(attempts):
            wav = self.model.generate(
                text=text,
                language_id=self.cfg.target_language,
                audio_prompt_path=str(reference_wav),
                exaggeration=self.cfg.tts_exaggeration,
                cfg_weight=cfg_weight,
                temperature=temp,
                repetition_penalty=rep_pen,
                min_p=self.cfg.tts_min_p,
                top_p=self.cfg.tts_top_p,
            )

            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", prefix="tts_", delete=False,
            )
            tmp.close()
            try:
                ta.save(tmp.name, wav, self.sample_rate)
                segment = AudioSegment.from_wav(tmp.name)
            finally:
                import os as _os
                try:
                    _os.remove(tmp.name)
                except OSError:
                    pass

            seg_ms = len(segment)
            if seg_ms > best_ms:
                best_segment = segment
                best_ms = seg_ms

            if seg_ms >= min_acceptable_ms:
                if attempt_idx > 0:
                    logger.info(
                        "TTS retry #%d succeeded for %d-char text (%d ms, expected >=%d ms)",
                        attempt_idx, len(text), seg_ms, int(min_acceptable_ms),
                    )
                return segment

            logger.warning(
                "TTS truncated on attempt #%d (%d ms < %d ms expected for %d chars), retrying…",
                attempt_idx + 1, seg_ms, int(min_acceptable_ms), len(text),
            )

        logger.error(
            "TTS truncated on all %d attempts; using best result (%d ms) for text: %r",
            len(attempts), best_ms, text[:80],
        )
        return best_segment if best_segment is not None else AudioSegment.silent(duration=100)

    def _split_long_text(self, text: str, limit: int) -> list[str]:
        """Split text into chunks up to *limit* chars at natural boundaries."""
        if len(text) <= limit:
            return [text]

        parts: list[str] = []
        remaining = text
        while len(remaining) > limit:
            best = -1
            for char in [". ", ", ", "; ", " "]:
                idx = remaining.rfind(char, 0, limit)
                if idx != -1:
                    best = idx + len(char)
                    break
            if best == -1:
                best = limit
            parts.append(remaining[:best].strip())
            remaining = remaining[best:].strip()

        if remaining:
            parts.append(remaining)
        return parts
