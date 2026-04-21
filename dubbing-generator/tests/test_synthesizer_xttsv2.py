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
