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
        raise NotImplementedError("Implemented in Task 5")
