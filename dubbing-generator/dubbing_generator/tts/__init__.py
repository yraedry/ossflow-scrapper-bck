"""TTS subpackage: voice synthesis backends."""

from __future__ import annotations

from ..config import DubbingConfig


def build_synthesizer(cfg: DubbingConfig):
    """Return the configured synthesizer instance.

    Supported engines:
    - ``elevenlabs`` (cloud, voice cloning, paid)
    - ``piper`` (local ONNX, no cloning, free, fast)
    - ``kokoro`` (local StyleTTS2, no cloning, free, GPU, better prosody)
    """
    engine = cfg.tts_engine
    if engine == "elevenlabs":
        from .synthesizer_elevenlabs import SynthesizerElevenLabs
        return SynthesizerElevenLabs(cfg)
    if engine == "piper":
        from .synthesizer_piper import SynthesizerPiper
        return SynthesizerPiper(cfg)
    if engine == "kokoro":
        from .synthesizer_kokoro import SynthesizerKokoro
        return SynthesizerKokoro(cfg)
    raise ValueError(
        f"Unsupported tts_engine: {engine!r} "
        f"(supported: 'elevenlabs', 'piper', 'kokoro')"
    )
