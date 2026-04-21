# XTTS-v2 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Chatterbox Multilingual with Coqui XTTS-v2 inside `dubbing-generator`, preserving the entire sync/mix/QA pipeline, and add ES/EN code-switching for BJJ terms.

**Architecture:** New `SynthesizerXTTSv2` class in a parallel file uses the low-level Coqui `Xtts` API with pre-computed speaker latents cached per reference WAV. A factory `build_synthesizer(cfg)` in `tts/__init__.py` picks engine by `DubbingConfig.tts_engine` (default `"xttsv2"`). Code-switching is handled by a pure `split_by_language` helper that splits input against a curated BJJ English-term set, then synthesizes each span with the matching `language` argument under the same shared latents.

**Tech Stack:** Python 3.11, `coqui-tts>=0.27.0` (TTS lib), `torch==2.6.0`, `torchaudio==2.6.0`, `pydub`, `pytest`. CUDA 12.4.

**Spec:** `docs/superpowers/specs/2026-04-21-xtts-v2-migration-design.md`

---

## File Structure

### New
- `dubbing-generator/dubbing_generator/tts/bjj_en_terms.py` — frozen default list + public `DEFAULT_BJJ_EN_TERMS` constant.
- `dubbing-generator/dubbing_generator/tts/lang_split.py` — pure `split_by_language(text, en_terms) -> list[tuple[str, str]]`.
- `dubbing-generator/dubbing_generator/tts/synthesizer_xttsv2.py` — `SynthesizerXTTSv2` class with lazy load, latent cache, code-switching.
- `dubbing-generator/tests/test_lang_split.py` — unit tests for splitter (no GPU).
- `dubbing-generator/tests/test_synthesizer_xttsv2.py` — unit tests for cache + factory (no GPU, heavy mocking).
- `dubbing-generator/tests/test_synthesizer_xttsv2_gpu.py` — integration tests marked `@pytest.mark.gpu`.

### Modified
- `dubbing-generator/dubbing_generator/tts/__init__.py` — export `build_synthesizer`.
- `dubbing-generator/dubbing_generator/config.py` — add XTTS fields, flip default `tts_engine`.
- `dubbing-generator/dubbing_generator/pipeline.py` — line 122 swap.
- `dubbing-generator/app.py` — line 299 swap.
- `dubbing-generator/requirements.txt` — drop Chatterbox stack, add `coqui-tts`.

### Untouched (guardrails from spec §2)
`sync/aligner.py`, `sync/words_index.py`, `audio/mixer.py`, `audio/stretcher.py`, `tts/synthesizer.py` (rollback), `tts/voice_cloner.py`, QA module, drift corrector, overlap resolver.

---

## Task 1: Add XTTS-v2 config fields

**Files:**
- Modify: `dubbing-generator/dubbing_generator/config.py`
- Test: `dubbing-generator/tests/test_config.py` (add cases)

- [ ] **Step 1: Write the failing test**

Append to `dubbing-generator/tests/test_config.py`:

```python
def test_xtts_defaults():
    cfg = DubbingConfig()
    assert cfg.tts_engine == "xttsv2"
    assert cfg.xtts_model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
    assert cfg.xtts_config_path == ""
    assert cfg.xtts_checkpoint_dir == ""
    assert cfg.xtts_use_deepspeed is False
    assert cfg.xtts_code_switching is True
    assert cfg.xtts_en_terms_extra == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dubbing-generator && pytest tests/test_config.py::test_xtts_defaults -v`
Expected: FAIL with `AttributeError: 'DubbingConfig' object has no attribute 'tts_engine'`.

- [ ] **Step 3: Add the fields to `DubbingConfig`**

In `dubbing-generator/dubbing_generator/config.py`, immediately after the `tts_model_name` field (currently the last `tts_*` field in the TTS block, around line 61), insert:

```python
    tts_engine: str = "xttsv2"

    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_config_path: str = ""
    xtts_checkpoint_dir: str = ""
    xtts_use_deepspeed: bool = False
    xtts_code_switching: bool = True
    xtts_en_terms_extra: tuple[str, ...] = ()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dubbing-generator && pytest tests/test_config.py::test_xtts_defaults -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dubbing-generator/dubbing_generator/config.py dubbing-generator/tests/test_config.py
git commit -m "feat(dubbing): add XTTS-v2 config fields"
```

---

## Task 2: Add default BJJ English-terms list

**Files:**
- Create: `dubbing-generator/dubbing_generator/tts/bjj_en_terms.py`
- Test: `dubbing-generator/tests/test_lang_split.py` (stub first)

- [ ] **Step 1: Write the failing test**

Create `dubbing-generator/tests/test_lang_split.py` with:

```python
"""Unit tests for the ES/EN language splitter and BJJ term list."""

from dubbing_generator.tts.bjj_en_terms import DEFAULT_BJJ_EN_TERMS


def test_default_bjj_terms_is_frozen_set():
    assert isinstance(DEFAULT_BJJ_EN_TERMS, frozenset)
    assert len(DEFAULT_BJJ_EN_TERMS) >= 25


def test_default_bjj_terms_are_lowercase():
    for term in DEFAULT_BJJ_EN_TERMS:
        assert term == term.lower(), f"non-lowercase term: {term!r}"


def test_default_bjj_terms_core_coverage():
    core = {"underhook", "overhook", "guard", "two on one", "lapel", "mount"}
    assert core <= DEFAULT_BJJ_EN_TERMS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dubbing-generator && pytest tests/test_lang_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dubbing_generator.tts.bjj_en_terms'`.

- [ ] **Step 3: Create the module**

Create `dubbing-generator/dubbing_generator/tts/bjj_en_terms.py`:

```python
"""Default BJJ English terms that must be pronounced with English phonology.

All terms are lowercase. Multi-word terms are stored with single spaces.
The splitter matches case-insensitively on word boundaries; callers may
extend the set via DubbingConfig.xtts_en_terms_extra.
"""

from __future__ import annotations

DEFAULT_BJJ_EN_TERMS: frozenset[str] = frozenset({
    # Grips and arm controls
    "underhook", "underhooks",
    "overhook", "overhooks",
    "two on one",
    "wrist control",
    "collar tie",
    "russian tie",
    "lapel", "lapels",
    "sleeve",
    "grip",
    # Positions
    "guard",
    "closed guard",
    "open guard",
    "half guard",
    "full mount",
    "mount",
    "side control",
    "back control",
    "north south",
    "turtle",
    # Submissions
    "armbar",
    "triangle",
    "kimura",
    "americana",
    "heel hook",
    "rear naked choke",
    "guillotine",
    "bow and arrow",
    # Actions
    "sweep",
    "pass",
    "takedown",
    "shoot",
    "sprawl",
})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dubbing-generator && pytest tests/test_lang_split.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dubbing-generator/dubbing_generator/tts/bjj_en_terms.py dubbing-generator/tests/test_lang_split.py
git commit -m "feat(dubbing): add default BJJ English-terms list"
```

---

## Task 3: Implement `split_by_language` pure helper

**Files:**
- Create: `dubbing-generator/dubbing_generator/tts/lang_split.py`
- Test: `dubbing-generator/tests/test_lang_split.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `dubbing-generator/tests/test_lang_split.py`:

```python
from dubbing_generator.tts.lang_split import split_by_language


def test_split_all_spanish():
    result = split_by_language("esto es todo en español", frozenset({"guard"}))
    assert result == [("es", "esto es todo en español")]


def test_split_empty_terms_returns_single_es_span():
    result = split_by_language("aplicamos un two on one", frozenset())
    assert result == [("es", "aplicamos un two on one")]


def test_split_embedded_english_term():
    terms = frozenset({"two on one", "guard"})
    result = split_by_language(
        "aplicamos un two on one desde la guard",
        terms,
    )
    assert result == [
        ("es", "aplicamos un "),
        ("en", "two on one"),
        ("es", " desde la "),
        ("en", "guard"),
    ]


def test_split_english_at_start_and_end():
    terms = frozenset({"underhook"})
    result = split_by_language("underhook y luego underhook", terms)
    assert result == [
        ("en", "underhook"),
        ("es", " y luego "),
        ("en", "underhook"),
    ]


def test_split_word_boundary_no_substring_match():
    """'underhooks' must match as itself, not produce 'underhook' + 's'."""
    terms = frozenset({"underhook", "underhooks"})
    result = split_by_language("los underhooks son clave", terms)
    assert result == [
        ("es", "los "),
        ("en", "underhooks"),
        ("es", " son clave"),
    ]


def test_split_case_insensitive_match_preserves_original_casing():
    terms = frozenset({"guard"})
    result = split_by_language("la Guard cerrada", terms)
    assert result == [
        ("es", "la "),
        ("en", "Guard"),
        ("es", " cerrada"),
    ]


def test_split_prefers_longer_match():
    """'two on one' must win over 'one' when both are in the set."""
    terms = frozenset({"one", "two on one"})
    result = split_by_language("aplicamos two on one", terms)
    assert result == [
        ("es", "aplicamos "),
        ("en", "two on one"),
    ]


def test_split_empty_string():
    assert split_by_language("", frozenset({"guard"})) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dubbing-generator && pytest tests/test_lang_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dubbing_generator.tts.lang_split'`.

- [ ] **Step 3: Implement `split_by_language`**

Create `dubbing-generator/dubbing_generator/tts/lang_split.py`:

```python
"""Split a mixed ES/EN sentence into per-language spans."""

from __future__ import annotations

import re


def split_by_language(
    text: str,
    en_terms: frozenset[str] | set[str],
) -> list[tuple[str, str]]:
    """Return [(lang, span), ...] preserving order and original whitespace.

    - ``lang`` is ``"es"`` (default) or ``"en"`` (matched term).
    - Matching is case-insensitive on word boundaries; original casing is
      preserved in the returned span.
    - Longer terms take precedence (``"two on one"`` beats ``"one"``).
    - Empty input returns an empty list.
    """
    if not text:
        return []
    if not en_terms:
        return [("es", text)]

    sorted_terms = sorted(en_terms, key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in sorted_terms) + r")\b",
        re.IGNORECASE,
    )

    spans: list[tuple[str, str]] = []
    cursor = 0
    for match in pattern.finditer(text):
        start, end = match.start(), match.end()
        if start > cursor:
            spans.append(("es", text[cursor:start]))
        spans.append(("en", text[start:end]))
        cursor = end
    if cursor < len(text):
        spans.append(("es", text[cursor:]))
    return spans
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dubbing-generator && pytest tests/test_lang_split.py -v`
Expected: PASS (11 tests total).

- [ ] **Step 5: Commit**

```bash
git add dubbing-generator/dubbing_generator/tts/lang_split.py dubbing-generator/tests/test_lang_split.py
git commit -m "feat(dubbing): add pure split_by_language helper"
```

---

## Task 4: Implement `SynthesizerXTTSv2` skeleton with latent cache

**Files:**
- Create: `dubbing-generator/dubbing_generator/tts/synthesizer_xttsv2.py`
- Test: `dubbing-generator/tests/test_synthesizer_xttsv2.py`

- [ ] **Step 1: Write the failing test for the latent cache key**

Create `dubbing-generator/tests/test_synthesizer_xttsv2.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dubbing-generator && pytest tests/test_synthesizer_xttsv2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dubbing_generator.tts.synthesizer_xttsv2'`.

- [ ] **Step 3: Create the skeleton**

Create `dubbing-generator/dubbing_generator/tts/synthesizer_xttsv2.py`:

```python
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
        gpt_cond, spk_emb = self._model.get_conditioning_latents(
            audio_path=[str(reference_wav)]
        )
        self._latent_cache[key] = (gpt_cond, spk_emb)
        return gpt_cond, spk_emb

    def generate(
        self,
        text: str,
        reference_wav: Path,
        speed: float | None = None,
    ) -> AudioSegment:
        raise NotImplementedError("Implemented in Task 5")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dubbing-generator && pytest tests/test_synthesizer_xttsv2.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dubbing-generator/dubbing_generator/tts/synthesizer_xttsv2.py dubbing-generator/tests/test_synthesizer_xttsv2.py
git commit -m "feat(dubbing): XTTS-v2 synthesizer skeleton with latent cache"
```

---

## Task 5: Implement `generate` with per-span synthesis and crossfade

**Files:**
- Modify: `dubbing-generator/dubbing_generator/tts/synthesizer_xttsv2.py`
- Test: `dubbing-generator/tests/test_synthesizer_xttsv2.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `dubbing-generator/tests/test_synthesizer_xttsv2.py`:

```python
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
    _, kwargs = model.inference.call_args
    assert kwargs["language"] == "es"
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

    langs = [call.kwargs["language"] for call in model.inference.call_args_list]
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dubbing-generator && pytest tests/test_synthesizer_xttsv2.py -v`
Expected: FAIL with `NotImplementedError: Implemented in Task 5`.

- [ ] **Step 3: Implement `generate` and helpers**

Replace the `generate` stub in `dubbing-generator/dubbing_generator/tts/synthesizer_xttsv2.py` (and add helpers) with:

```python
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
        import torchaudio as ta
        import torch

        try:
            out = self._model.inference(
                text,
                language,
                gpt_cond,
                spk_emb,
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
        tensor = torch.as_tensor(wav).unsqueeze(0) if getattr(wav, "ndim", 1) == 1 else torch.as_tensor(wav)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="xtts_", delete=False)
        tmp.close()
        try:
            ta.save(tmp.name, tensor.cpu().float(), self.sample_rate)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dubbing-generator && pytest tests/test_synthesizer_xttsv2.py -v`
Expected: PASS (6 tests total).

- [ ] **Step 5: Commit**

```bash
git add dubbing-generator/dubbing_generator/tts/synthesizer_xttsv2.py dubbing-generator/tests/test_synthesizer_xttsv2.py
git commit -m "feat(dubbing): XTTS-v2 generate with code-switching and crossfade"
```

---

## Task 6: Add `build_synthesizer` factory

**Files:**
- Modify: `dubbing-generator/dubbing_generator/tts/__init__.py`
- Test: `dubbing-generator/tests/test_synthesizer_xttsv2.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `dubbing-generator/tests/test_synthesizer_xttsv2.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dubbing-generator && pytest tests/test_synthesizer_xttsv2.py::test_build_synthesizer_xttsv2_default -v`
Expected: FAIL with `ImportError: cannot import name 'build_synthesizer'`.

- [ ] **Step 3: Implement the factory**

Replace `dubbing-generator/dubbing_generator/tts/__init__.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dubbing-generator && pytest tests/test_synthesizer_xttsv2.py -v`
Expected: PASS (9 tests total).

- [ ] **Step 5: Commit**

```bash
git add dubbing-generator/dubbing_generator/tts/__init__.py dubbing-generator/tests/test_synthesizer_xttsv2.py
git commit -m "feat(dubbing): add build_synthesizer factory"
```

---

## Task 7: Wire factory into pipeline and app

**Files:**
- Modify: `dubbing-generator/dubbing_generator/pipeline.py:22,122`
- Modify: `dubbing-generator/app.py:299`

- [ ] **Step 1: Update `pipeline.py` import and constructor**

In `dubbing-generator/dubbing_generator/pipeline.py`, line 22, replace:

```python
from .tts.synthesizer import Synthesizer
```

with:

```python
from .tts import build_synthesizer
```

Then at line 122, replace:

```python
        self.synthesizer = Synthesizer(config)
```

with:

```python
        self.synthesizer = build_synthesizer(config)
```

- [ ] **Step 2: Update `app.py` constructor**

In `dubbing-generator/app.py`, locate:

```python
from dubbing_generator.tts.synthesizer import Synthesizer
```

and replace with:

```python
from dubbing_generator.tts import build_synthesizer
```

At line 299, replace:

```python
    synth = Synthesizer(cfg)
```

with:

```python
    synth = build_synthesizer(cfg)
```

- [ ] **Step 3: Static import check**

Run: `cd dubbing-generator && python -c "from dubbing_generator.pipeline import DubbingPipeline; from dubbing_generator.config import DubbingConfig; print('ok')"`
Expected: prints `ok` with no traceback.

Run: `cd dubbing-generator && python -c "import app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Re-run full test suite**

Run: `cd dubbing-generator && pytest tests/ -v --ignore=tests/test_synthesizer_xttsv2_gpu.py`
Expected: all existing tests pass; no regressions.

- [ ] **Step 5: Commit**

```bash
git add dubbing-generator/dubbing_generator/pipeline.py dubbing-generator/app.py
git commit -m "feat(dubbing): wire build_synthesizer into pipeline and app"
```

---

## Task 8: Swap dependencies (drop Chatterbox, add coqui-tts)

**Files:**
- Modify: `dubbing-generator/requirements.txt`

- [ ] **Step 1: Replace the TTS stack lines**

Open `dubbing-generator/requirements.txt` and replace its current content with:

```
# Dubbing-generator deps.
# Uses Coqui XTTS-v2 (Idiap fork of coqui-ai/TTS, MPL-2.0 code, CPML weights).
# Chatterbox file remains on disk as rollback but requires a manual install
# of chatterbox-tts/transformers>=5.2 to re-enable.
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.6.0
torchaudio==2.6.0
coqui-tts>=0.27.0

# Audio stack
demucs
pydub
# librosa powers the QA boundary report (spectral centroid + pyin F0).
# Soft dep of the qa module — if missing the sidecar falls back to MOS-only.
librosa>=0.10.1

# Test-only
pytest>=7.4
pytest-asyncio>=0.23
```

- [ ] **Step 2: Install and smoke-test**

Run (inside the dubbing-generator venv/container):
```
pip install -r dubbing-generator/requirements.txt
```
Expected: installs cleanly; `coqui-tts` pulls a compatible `transformers`.

Run: `cd dubbing-generator && python -c "from TTS.tts.configs.xtts_config import XttsConfig; from TTS.tts.models.xtts import Xtts; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Re-run CPU test suite**

Run: `cd dubbing-generator && pytest tests/ -v --ignore=tests/test_synthesizer_xttsv2_gpu.py`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add dubbing-generator/requirements.txt
git commit -m "build(dubbing): swap Chatterbox stack for coqui-tts"
```

---

## Task 9: GPU integration tests (opt-in)

**Files:**
- Create: `dubbing-generator/tests/test_synthesizer_xttsv2_gpu.py`

- [ ] **Step 1: Create the integration test file**

Create `dubbing-generator/tests/test_synthesizer_xttsv2_gpu.py`:

```python
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
```

- [ ] **Step 2: Register the `gpu` marker in pytest config**

Check `dubbing-generator/pyproject.toml` or `pytest.ini` or `setup.cfg` for an existing `markers` entry. If there is one, add `"gpu: requires CUDA"`. If there is none, add to `dubbing-generator/pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "gpu: requires CUDA",
]
```

If `pyproject.toml` does not exist in `dubbing-generator/`, create it with only that section.

- [ ] **Step 3: Confirm the tests are discovered and skipped without GPU**

Run: `cd dubbing-generator && pytest tests/test_synthesizer_xttsv2_gpu.py -v`
Expected: both tests reported as `SKIPPED` with reason `CUDA required` or `reference wav missing`.

- [ ] **Step 4: On a GPU host, run them for real**

Run: `cd dubbing-generator && pytest tests/test_synthesizer_xttsv2_gpu.py -v`
Expected: both PASS. First run downloads the XTTS-v2 checkpoint (~1.8 GB).

- [ ] **Step 5: Commit**

```bash
git add dubbing-generator/tests/test_synthesizer_xttsv2_gpu.py dubbing-generator/pyproject.toml
git commit -m "test(dubbing): XTTS-v2 GPU integration tests"
```

---

## Task 10: End-to-end validation against baseline

**Files:** none (validation only, no code changes).

- [ ] **Step 1: Locate the baseline chapter**

The baseline in the spec is S01E02 Seated Guard processed with Chatterbox (MOS 1.40 / 21 hard cuts). Locate its input video + `.es.srt` under the library root referenced by `processor-api` settings. If unsure, ask the operator for the paths.

- [ ] **Step 2: Run the XTTS-v2 dub**

Trigger the dubbing job (via `processor-api` or CLI) with `tts_engine="xttsv2"` (default) against the same chapter. Let it run end-to-end.

- [ ] **Step 3: Capture the QA JSON**

After the run, locate the QA sidecar (`*.qa.json`) produced by the QA module. Record:
- Mean MOS
- Hard cut count
- RMS jump boundaries

- [ ] **Step 4: Compare against baseline**

Success criteria (from spec §1):
- Mean MOS >= 3.5 (stretch: >= 3.8)
- Hard cuts << 21
- Auditory sampling: "two on one", "underhook", "guard" pronounced with English phonology (not Spanish-ified)

- [ ] **Step 5: Record the result**

If the criteria are met: remove `synthesizer.py` (Chatterbox) and the Chatterbox config fields (`tts_exaggeration`, `tts_min_p`, `tts_model_name`, the chatterbox branch in the factory) in a follow-up PR titled `chore(dubbing): remove Chatterbox rollback after XTTS-v2 validation`.

If not met: log concrete failure modes (token repetition? timbre drift? EN span latency?) and open a tuning branch adjusting `tts_temperature`, `tts_repetition_penalty`, `tts_top_p`, `tts_speed`. Do NOT revert; tune first.

No commit for this task — it produces a decision, not code.

---

## Self-Review Pass

- **Spec §1 (motivation / success criteria):** covered by Task 10 validation.
- **Spec §2 (non-goals):** plan touches only files listed in spec §7. `sync/*`, `audio/*`, QA, overlap resolver untouched (verified — no step modifies them).
- **Spec §3.1 (coexistence + factory + default xttsv2):** Tasks 1 (config), 6 (factory), 7 (wiring).
- **Spec §3.2 (public contract `generate`/`sample_rate`/`load_model`/`model`):** Tasks 4, 5.
- **Spec §3.3 (low-level Xtts + bootstrap D1 + latent cache + param mapping + retry removal):** Task 4 (skeleton + `_ensure_model_downloaded` + `_get_latents`) + Task 5 (`inference` with exact param list, no retry).
- **Spec §3.4 (code-switching M1):** Tasks 2 (terms), 3 (splitter), 5 (per-span generate loop + crossfade reuse).
- **Spec §3.5 (errors):** Task 5 handles inference exceptions; Task 4 auto-downloads; empty spans filtered in Task 5 loop.
- **Spec §4 (requirements swap + model cache):** Task 8.
- **Spec §5 (config fields):** Task 1.
- **Spec §6 (unit + integration + manual validation):** Tasks 2, 3, 4, 5, 6 (units); Task 9 (GPU); Task 10 (manual).
- **Spec §7 (files touched):** matches plan's File Structure 1:1.
- **Spec §8 (rollback):** preserved — `synthesizer.py` untouched in all tasks; factory still accepts `"chatterbox"` (Task 6).

Type/signature consistency:
- `split_by_language(text, en_terms) -> list[tuple[str, str]]` — used in Task 3 definition and Task 5 call (match).
- `_get_latents(reference_wav) -> (gpt_cond, spk_emb)` — Task 4 definition, Task 5 consumer (match).
- `DEFAULT_BJJ_EN_TERMS: frozenset[str]` — Task 2 declaration, Task 4 constructor consumer (match).
- `build_synthesizer(cfg)` signature — Task 6 def, Task 7 callers (match).

No `TBD` / `TODO` / `similar to Task N` placeholders found. Every code step has runnable code; every test step has an expected PASS/FAIL.
