"""GPU-backed integration tests for SynthesizerXTTSv2.

Skipped automatically when CUDA is unavailable. Run explicitly with:
    pytest tests/test_synthesizer_xttsv2_gpu.py -v -m gpu
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dubbing_generator.config import DubbingConfig
from dubbing_generator.tts.synthesizer_xttsv2 import SynthesizerXTTSv2

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

REF_WAV = Path(__file__).resolve().parents[1] / "voices" / "luis_posada_clean.wav"


@pytest.fixture(scope="module")
def synth():
    if not REF_WAV.exists():
        pytest.skip(f"reference wav missing: {REF_WAV}")
    cfg = DubbingConfig()
    s = SynthesizerXTTSv2(cfg)
    s.load_model()
    return s


def test_generate_mono_spanish_produces_audio(synth):
    audio = synth.generate("Hola, esto es una prueba en español.", REF_WAV)
    assert len(audio) > 500
    assert audio.frame_rate == 24000
    assert audio.dBFS > -40.0


def test_generate_code_switching_produces_audio(synth):
    audio = synth.generate(
        "aplicamos un two on one desde la guard cerrada",
        REF_WAV,
    )
    assert len(audio) > 800
    assert audio.dBFS > -40.0
