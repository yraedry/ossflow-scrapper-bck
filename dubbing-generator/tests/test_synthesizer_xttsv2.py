"""Unit tests for SynthesizerXTTSv2 — no model load, heavy mocking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dubbing_generator.config import DubbingConfig
from dubbing_generator.tts.synthesizer_xttsv2 import SynthesizerXTTSv2


def _write_wav_stub(path: Path, content: bytes = b"RIFFxxxxWAVEfmt ") -> None:
    path.write_bytes(content)


def test_latent_cache_hits_on_same_path_and_mtime(tmp_path):
    cfg = DubbingConfig()
    synth = SynthesizerXTTSv2(cfg)
    fake_latents = (object(), object())
    synth._model = MagicMock()
    synth._model.get_conditioning_latents.return_value = fake_latents

    ref = tmp_path / "ref.wav"
    _write_wav_stub(ref)

    first = synth._get_latents(ref)
    second = synth._get_latents(ref)

    assert first is fake_latents
    assert second is fake_latents
    synth._model.get_conditioning_latents.assert_called_once_with(audio_path=[str(ref)])


def test_latent_cache_misses_when_mtime_changes(tmp_path):
    cfg = DubbingConfig()
    synth = SynthesizerXTTSv2(cfg)
    synth._model = MagicMock()
    synth._model.get_conditioning_latents.side_effect = [
        (object(), object()),
        (object(), object()),
    ]

    ref = tmp_path / "ref.wav"
    _write_wav_stub(ref)
    synth._get_latents(ref)

    import os
    new_mtime = ref.stat().st_mtime_ns + 10_000_000_000
    os.utime(ref, ns=(new_mtime, new_mtime))

    synth._get_latents(ref)
    assert synth._model.get_conditioning_latents.call_count == 2


import numpy as np
import torch
from pydub import AudioSegment


def _fake_inference_result(duration_s: float = 0.5) -> dict:
    samples = int(duration_s * _XTTS_SAMPLE_RATE_FOR_TESTS)
    return {"wav": np.zeros(samples, dtype=np.float32)}


_XTTS_SAMPLE_RATE_FOR_TESTS = 24000


def _install_model_mock(synth: SynthesizerXTTSv2) -> MagicMock:
    model = MagicMock()
    model.get_conditioning_latents.return_value = (object(), object())
    model.inference.return_value = _fake_inference_result(0.4)
    synth._model = model
    synth._sr = _XTTS_SAMPLE_RATE_FOR_TESTS
    return model


def test_generate_mono_spanish_single_inference(tmp_path):
    cfg = DubbingConfig()
    cfg.xtts_code_switching = False
    synth = SynthesizerXTTSv2(cfg)
    model = _install_model_mock(synth)

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFxxxxWAVEfmt ")

    audio = synth.generate("hola mundo", ref, speed=1.0)

    assert isinstance(audio, AudioSegment)
    assert model.inference.call_count == 1
    args, kwargs = model.inference.call_args
    assert args[1] == "es"  # language is positional arg #2
    assert kwargs["temperature"] == cfg.tts_temperature
    assert kwargs["repetition_penalty"] == cfg.tts_repetition_penalty
    assert kwargs["top_p"] == cfg.tts_top_p
    assert kwargs["speed"] == 1.0


def test_generate_code_switching_splits_by_language(tmp_path):
    cfg = DubbingConfig()
    cfg.xtts_code_switching = True
    synth = SynthesizerXTTSv2(cfg)
    model = _install_model_mock(synth)

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFxxxxWAVEfmt ")

    synth.generate("aplicamos un two on one desde la guard", ref)

    langs = [call.args[1] for call in model.inference.call_args_list]
    assert langs == ["es", "en", "es", "en"]


def test_generate_uses_default_speed_from_config(tmp_path):
    cfg = DubbingConfig()
    synth = SynthesizerXTTSv2(cfg)
    model = _install_model_mock(synth)

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFxxxxWAVEfmt ")

    synth.generate("hola", ref)

    assert model.inference.call_args.kwargs["speed"] == cfg.tts_speed


def test_generate_empty_text_returns_short_silence(tmp_path):
    cfg = DubbingConfig()
    synth = SynthesizerXTTSv2(cfg)
    _install_model_mock(synth)

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFxxxxWAVEfmt ")

    audio = synth.generate("", ref)
    assert len(audio) > 0
    assert len(audio) <= 200


def test_build_synthesizer_xttsv2_default():
    from dubbing_generator.tts import build_synthesizer
    cfg = DubbingConfig()
    synth = build_synthesizer(cfg)
    assert isinstance(synth, SynthesizerXTTSv2)


def test_build_synthesizer_chatterbox_fallback():
    from dubbing_generator.tts import build_synthesizer
    from dubbing_generator.tts.synthesizer import Synthesizer
    cfg = DubbingConfig()
    cfg.tts_engine = "chatterbox"
    synth = build_synthesizer(cfg)
    assert isinstance(synth, Synthesizer)


def test_build_synthesizer_unknown_engine_raises():
    from dubbing_generator.tts import build_synthesizer
    cfg = DubbingConfig()
    cfg.tts_engine = "nope"
    with pytest.raises(ValueError, match="tts_engine"):
        build_synthesizer(cfg)
