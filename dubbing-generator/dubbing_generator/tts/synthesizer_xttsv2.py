"""Coqui XTTS-v2 TTS wrapper — voice cloning with per-span code-switching."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from pydub import AudioSegment

from ..config import DubbingConfig
from .bjj_en_terms import DEFAULT_BJJ_EN_TERMS
from .lang_split import split_by_language

logger = logging.getLogger(__name__)

_XTTS_SAMPLE_RATE = 24000


class SynthesizerXTTSv2:
    """Generate speech from text using Coqui XTTS-v2.

    Uses the low-level ``Xtts`` API so speaker conditioning latents are
    computed once per reference WAV (cached by path + mtime) and shared
    across all chunks of the chapter, keeping timbre stable between
    Spanish and English spans in code-switching.
    """

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config
        self._model = None
        self._sr: int | None = None
        self._latent_cache: dict[tuple[str, int], tuple[object, object]] = {}
        self._en_terms = frozenset(DEFAULT_BJJ_EN_TERMS) | frozenset(
            t.lower() for t in config.xtts_en_terms_extra
        )

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

    def load_model(self) -> None:
        """Lazy-load XTTS-v2, triggering download on first use."""
        if self._model is not None:
            return

        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config_path, ckpt_dir = self._ensure_model_downloaded()

        xtts_config = XttsConfig()
        xtts_config.load_json(config_path)
        self._model = Xtts.init_from_config(xtts_config)
        self._model.load_checkpoint(
            xtts_config,
            checkpoint_dir=ckpt_dir,
            use_deepspeed=self.cfg.xtts_use_deepspeed,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            self._model.cuda()
        logger.info("XTTS-v2 loaded on %s (sr=%d Hz)", device, _XTTS_SAMPLE_RATE)
        self._sr = _XTTS_SAMPLE_RATE

    def _ensure_model_downloaded(self) -> tuple[str, str]:
        """Return (config_path, checkpoint_dir), downloading if needed.

        Respects explicit overrides in config; otherwise delegates to the
        Coqui ModelManager which is idempotent (cached on second run).
        """
        cfg_path = self.cfg.xtts_config_path
        ckpt_dir = self.cfg.xtts_checkpoint_dir
        if cfg_path and ckpt_dir:
            return cfg_path, ckpt_dir

        from TTS.utils.manage import ModelManager

        mm = ModelManager()
        model_path, config_path, _ = mm.download_model(self.cfg.xtts_model_name)
        # ModelManager returns: (model_path=checkpoint_dir, config_path, model_item)
        return cfg_path or config_path, ckpt_dir or model_path

    def _get_latents(self, reference_wav: Path) -> tuple[object, object]:
        key = (str(reference_wav.resolve()), reference_wav.stat().st_mtime_ns)
        cached = self._latent_cache.get(key)
        if cached is not None:
            return cached
        latents = self._model.get_conditioning_latents(
            audio_path=[str(reference_wav)]
        )
        self._latent_cache[key] = latents
        return latents

    def generate(
        self,
        text: str,
        reference_wav: Path,
        speed: float | None = None,
    ) -> AudioSegment:
        """Synthesize *text* cloning voice from *reference_wav*."""
        if not text.strip():
            return AudioSegment.silent(duration=100)

        if speed is None:
            speed = self.cfg.tts_speed

        _ = self.model  # lazy-load
        gpt_cond, spk_emb = self._get_latents(reference_wav)

        if self.cfg.xtts_code_switching:
            spans = split_by_language(text, self._en_terms)
        else:
            spans = [("es", text)]

        segments: list[AudioSegment] = []
        for lang, span in spans:
            if not span.strip():
                continue
            for chunk in self._split_long_text(span, self.cfg.tts_char_limit):
                seg = self._synthesize_chunk(chunk, lang, gpt_cond, spk_emb, speed)
                if len(seg) > 0:
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

        return self._normalize(result, target_dbfs=-18.0)

    def _synthesize_chunk(
        self,
        text: str,
        language: str,
        gpt_cond,
        spk_emb,
        speed: float,
    ) -> AudioSegment:
        """Synthesize one language-homogeneous chunk and return an AudioSegment."""
        import numpy as np
        import torch

        try:
            out = self._model.inference(
                text,
                language=language,
                gpt_cond_latents=gpt_cond,
                speaker_embedding=spk_emb,
                temperature=self.cfg.tts_temperature,
                repetition_penalty=self.cfg.tts_repetition_penalty,
                top_p=self.cfg.tts_top_p,
                speed=speed,
            )
        except Exception:
            logger.exception(
                "XTTS inference failed for %s span (%d chars); substituting silence",
                language, len(text),
            )
            return AudioSegment.silent(duration=200)

        wav = out["wav"]
        # Normalise to a flat float32 numpy array regardless of input type
        if isinstance(wav, torch.Tensor):
            samples: np.ndarray = wav.cpu().float().numpy().flatten()
        else:
            samples = np.asarray(wav, dtype=np.float32).flatten()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="xtts_", delete=False)
        tmp.close()
        try:
            import soundfile as sf
            sf.write(tmp.name, samples, self.sample_rate, subtype="PCM_16")
            return AudioSegment.from_wav(tmp.name)
        finally:
            import os as _os
            try:
                _os.remove(tmp.name)
            except OSError:
                pass

    def _split_long_text(self, text: str, limit: int) -> list[str]:
        """Split text into chunks up to *limit* chars at natural boundaries."""
        if len(text) <= limit:
            return [text]

        parts: list[str] = []
        remaining = text
        while len(remaining) > limit:
            best = -1
            for ch in [". ", ", ", "; ", " "]:
                idx = remaining.rfind(ch, 0, limit)
                if idx != -1:
                    best = idx + len(ch)
                    break
            if best == -1:
                best = limit
            parts.append(remaining[:best].strip())
            remaining = remaining[best:].strip()

        if remaining:
            parts.append(remaining)
        return parts

    @staticmethod
    def _normalize(audio: AudioSegment, target_dbfs: float = -18.0) -> AudioSegment:
        if audio.dBFS == float("-inf"):
            return audio
        delta = target_dbfs - audio.dBFS
        delta = max(-12.0, min(12.0, delta))
        return audio.apply_gain(delta)
