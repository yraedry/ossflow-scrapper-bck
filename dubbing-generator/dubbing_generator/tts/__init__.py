"""TTS subpackage: voice synthesis and cloning."""

from __future__ import annotations

from ..config import DubbingConfig


def build_synthesizer(cfg: DubbingConfig):
    """Return the synthesizer implementation selected by ``cfg.tts_engine``."""
    engine = cfg.tts_engine
    if engine == "xttsv2":
        from .synthesizer_xttsv2 import SynthesizerXTTSv2
        return SynthesizerXTTSv2(cfg)
    if engine == "chatterbox":
        from .synthesizer import Synthesizer
        return Synthesizer(cfg)
    raise ValueError(f"Unknown tts_engine: {engine!r}")
