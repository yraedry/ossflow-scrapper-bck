# XTTS-v2 Migration Design

**Date:** 2026-04-21
**Scope:** `dubbing-generator` only
**Status:** Approved — ready for implementation plan

## 1. Motivation

Current engine Chatterbox Multilingual produces Spanish dubbing at MOS ~1.4 (validated on S01E02 Jozef Chen with clean reference `luis_posada_clean.wav`). Two hard blockers:

- **Token repetition bug:** tokens 6405/6324/4137/1034 trigger false-positive repetition detection → premature EOS → truncated phrases → retries, RMS jumps between segments, audible stitching.
- **No code-switching:** BJJ English terms embedded in Spanish translations ("two on one", "underhook", "lapel") come out phonetically broken ("to on guane"), because the model forces Spanish phonology across the full utterance.

XTTS-v2 (Coqui) has Spanish MOS ~3.8-4.1 and accepts a per-inference `language` parameter, enabling clean code-switching with a single shared speaker embedding.

**Success criteria** (manual validation on the baseline chapter):
- MOS >= 3.5 (target 3.8+)
- Hard cuts << 21 (baseline)
- English BJJ terms audibly native EN, not Spanish phonology

## 2. Non-goals

- No changes to `sync/aligner.py`, `sync/words_index.py`.
- No changes to `pipeline.py` compact-synthetic-silence rail (`compact_synthetic_max_drift_ms=400`, `original_start_ms` anchors, `compact_synthetic_min_speech_ratio=0.70`).
- No changes to `audio/mixer.py` (ducking, crossfades, sustain), `audio/stretcher.py`, overlap resolver, QA module.
- No toggle exposed via the dynamic settings API (personal use, single user).
- No translator-side markup (`<en>...</en>`) — left as TODO for future M2 phase.

## 3. Architecture

### 3.1 Coexistence (opt-in engine)

A new file `dubbing_generator/tts/synthesizer_xttsv2.py` is added next to the existing `synthesizer.py` (Chatterbox). Chatterbox stays on disk as rollback but is removed from `requirements.txt` (lazy imports keep the file importable without the package installed, as long as `load_model()` is never called on it).

A factory in `dubbing_generator/tts/__init__.py`:

```python
def build_synthesizer(cfg: DubbingConfig):
    if cfg.tts_engine == "xttsv2":
        from .synthesizer_xttsv2 import SynthesizerXTTSv2
        return SynthesizerXTTSv2(cfg)
    from .synthesizer import Synthesizer
    return Synthesizer(cfg)
```

Callers (`pipeline.py:122`, `app.py:299`) switch from `Synthesizer(config)` to `build_synthesizer(config)`. Default `tts_engine="xttsv2"`.

### 3.2 `SynthesizerXTTSv2` public contract

Identical surface to the current `Synthesizer` so downstream code (drift corrector, stretcher, mixer) needs zero changes:

- `generate(text: str, reference_wav: Path, speed: float | None = None) -> AudioSegment`
- `sample_rate: int` property — returns 24000 (XTTS-v2 fixed)
- `load_model()` — idempotent lazy load
- `model` property — triggers load on first access

### 3.3 Internal design (low-level Xtts)

Rationale for low-level `Xtts` over the high-level `TTS` facade: pre-computed speaker latents shared across all chunks of a chapter → stable timbre across ES/EN spans in code-switching, and native exposure of sampling parameters already tuned in `DubbingConfig`.

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

class SynthesizerXTTSv2:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self._sr = None
        self._latent_cache = {}   # (abs_path, mtime_ns) -> (gpt_cond, spk_emb)
        self._en_terms = set(DEFAULT_BJJ_EN_TERMS) | set(cfg.xtts_en_terms_extra)

    def load_model(self):
        if self._model is not None:
            return
        config_path, ckpt_dir = self._ensure_model_downloaded()
        config = XttsConfig()
        config.load_json(config_path)
        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(
            config, checkpoint_dir=ckpt_dir,
            use_deepspeed=self.cfg.xtts_use_deepspeed,
        )
        if torch.cuda.is_available():
            self._model.cuda()
        self._sr = 24000
```

#### 3.3.1 Model bootstrap (D1)

First load triggers download via the high-level `ModelManager` once, then low-level loader reads config/checkpoint from the resulting cache directory:

```python
def _ensure_model_downloaded(self):
    from TTS.utils.manage import ModelManager
    mm = ModelManager()
    model_path, config_path, _ = mm.download_model(self.cfg.xtts_model_name)
    return config_path, model_path  # model_path is the checkpoint dir
```

Overridable via `cfg.xtts_config_path` / `cfg.xtts_checkpoint_dir` (both empty by default → auto-resolve).

#### 3.3.2 Latent cache

Keyed by `(abs_path, mtime_ns)` of the reference WAV so a swap of the reference file on disk invalidates the cache cleanly:

```python
def _get_latents(self, reference_wav: Path):
    key = (str(reference_wav.resolve()), reference_wav.stat().st_mtime_ns)
    cached = self._latent_cache.get(key)
    if cached is not None:
        return cached
    gpt_cond, spk_emb = self._model.get_conditioning_latents(
        audio_path=[str(reference_wav)]
    )
    self._latent_cache[key] = (gpt_cond, spk_emb)
    return gpt_cond, spk_emb
```

One latent computation per chapter run (the cloner hands over the same `luis_posada_clean.wav` for every block).

#### 3.3.3 Parameter mapping

| DubbingConfig field | XTTS-v2 param | Notes |
|---|---|---|
| `tts_temperature` (0.65) | `temperature` | |
| `tts_repetition_penalty` (1.45) | `repetition_penalty` | |
| `tts_top_p` (1.0) | `top_p` | |
| `tts_speed` (0.88) | `speed` | Native support — no `cfg_weight` math needed |
| — | `top_k` (default 50) | Not currently tuned, leave at library default |
| — | `length_penalty` (default 1.0) | Not currently tuned, leave at library default |

Dropped (Chatterbox-only, no XTTS equivalent):
- `cfg_weight`
- `tts_exaggeration`
- `tts_min_p`

Dropped behavior:
- **Escalated-retry loop** (temperature/penalty bumps on truncation). The bug it works around is Chatterbox-specific. A single defensive minimum-duration check remains; on failure, log and return best available — no re-sampling.

### 3.4 Code-switching (Fase 2 integrated, M1 detector)

A pure helper `split_by_language(text, en_terms) -> list[tuple[lang, span]]` scans the input and splits on word-boundary matches against a curated BJJ EN-term set. The set lives in a new module `dubbing_generator/tts/bjj_en_terms.py` (starts at ~30 core terms, extendable via `cfg.xtts_en_terms_extra`).

Example: `"aplicamos un two on one desde la guard"` →
`[("es", "aplicamos un "), ("en", "two on one"), ("es", " desde la "), ("en", "guard")]`

Each span is synthesized with the shared latents (timbre stability) and its own `language` argument. Segments are concatenated with the existing `tts_crossfade_ms` (120 ms) join path — no new crossfade logic needed.

When `cfg.xtts_code_switching=False` the helper returns `[("es", text)]` unconditionally → single-pass mono-ES path.

Word-boundary rule: EN term must match as a whole word (regex `\b<term>\b`, case-insensitive). Multi-word terms (`"two on one"`) are treated as a single atomic phrase. `"underhooks"` does NOT match `"underhook"` to avoid splitting Spanish plurals/suffixes.

**TODO (future M2 phase):** accept translator-emitted markers like `<en>two on one</en>` and skip the detector when markers are present. Deferred — translator currently emits none.

### 3.5 Error handling

- **Model not downloaded:** `ModelManager.download_model()` is idempotent; runs on first `load_model()`.
- **Invalid reference WAV** (SR <16 kHz, duration <3 s): log warning, pass through — XTTS tolerates short refs.
- **Empty EN span after split:** filter before synthesis.
- **`model.inference()` exception on a span:** log error, substitute `AudioSegment.silent(200ms)`, continue with remaining spans. No full-phrase failure.
- **VRAM exhaustion:** hard fail with clear message (XTTS-v2 needs ~4.5 GB GPU). No silent CPU fallback.

## 4. Dependencies

### 4.1 `requirements.txt` changes

Remove:
```
transformers>=5.2.0
chatterbox-tts>=0.1.7
resemble-perth>=1.0.1
setuptools<81
```

Add:
```
coqui-tts>=0.27.0
```

Keep unchanged: `torch==2.6.0`, `torchaudio==2.6.0`, `demucs`, `pydub`, `librosa>=0.10.1`, pytest deps.

`coqui-tts` pins its own compatible `transformers`. Chatterbox file (`synthesizer.py`) remains importable in isolation (lazy imports) but calling `load_model()` on it after the swap would raise `ImportError` — acceptable, that's the rollback path.

### 4.2 Model cache

Default cache: `~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/` (Windows: `%LOCALAPPDATA%\tts\...`). First run downloads ~1.8 GB. Subsequent runs are offline.

## 5. Config changes

New fields in `DubbingConfig`:

```python
# Engine selection
tts_engine: str = "xttsv2"  # "chatterbox" | "xttsv2"

# XTTS-v2 model bootstrap
xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
xtts_config_path: str = ""        # empty → auto-resolve from ModelManager cache
xtts_checkpoint_dir: str = ""     # empty → idem
xtts_use_deepspeed: bool = False  # opt-in perf; requires deepspeed install

# Code-switching
xtts_code_switching: bool = True
xtts_en_terms_extra: tuple[str, ...] = ()
```

Existing TTS fields (`tts_speed`, `tts_temperature`, `tts_repetition_penalty`, `tts_top_p`, `tts_crossfade_ms`, `tts_char_limit`, `avg_ms_per_char`) are reused as-is. Chatterbox-only fields (`tts_exaggeration`, `tts_min_p`, `tts_model_name`) are kept for backward compat with `synthesizer.py` but unused when `tts_engine="xttsv2"`.

## 6. Testing

### 6.1 Unit tests (CPU, no model load)

New file `dubbing-generator/tests/test_synthesizer_xttsv2.py`:

- `test_split_by_language_pure_es` — 100% Spanish → single ES span.
- `test_split_by_language_embedded_en` — `"aplicamos un two on one desde la guard"` → 4 spans in order, whitespace preserved.
- `test_split_by_language_edge_positions` — EN term at start and at end → correct boundaries.
- `test_split_by_language_word_boundary` — `"underhooks"` does NOT match `"underhook"` term.
- `test_split_by_language_empty_terms` — empty set → single ES span.
- `test_default_bjj_terms_snapshot` — snapshot of default list, catches accidental mutations.
- `test_build_synthesizer_factory` — with engine toggle, asserts correct class (mock imports to avoid loading models).
- `test_latent_cache_key_stable` — same path + same mtime → hit; mtime change → miss.

### 6.2 Integration tests (`@pytest.mark.gpu`, skipped without CUDA)

- `test_xtts_generate_mono_es` — synthesize "Hola, esto es una prueba" with real reference. Assert duration >0 and `frame_rate==24000`.
- `test_xtts_generate_code_switching` — synthesize "aplicamos un two on one desde la guard". Assert duration >0 and no silent spans (RMS > -40 dBFS).

### 6.3 End-to-end validation (manual)

Re-dub chapter S01E02 Seated Guard (baseline Chatterbox: MOS 1.40 / 21 hard cuts). Compare QA JSON side-by-side:
- MOS target >= 3.5
- Hard cuts target << 21
- Auditory check: "two on one", "underhook" pronounced in English phonology

## 7. Files touched

### New
- `dubbing-generator/dubbing_generator/tts/synthesizer_xttsv2.py`
- `dubbing-generator/dubbing_generator/tts/bjj_en_terms.py`
- `dubbing-generator/tests/test_synthesizer_xttsv2.py`

### Modified
- `dubbing-generator/dubbing_generator/tts/__init__.py` — add `build_synthesizer` factory
- `dubbing-generator/dubbing_generator/config.py` — add XTTS fields, default `tts_engine="xttsv2"`
- `dubbing-generator/dubbing_generator/pipeline.py` — swap `Synthesizer(config)` for `build_synthesizer(config)` at line 122
- `dubbing-generator/app.py` — same swap at line 299
- `dubbing-generator/requirements.txt` — remove Chatterbox stack, add `coqui-tts`

### Untouched (guardrails)
- `dubbing-generator/dubbing_generator/tts/synthesizer.py` — kept as rollback
- `dubbing-generator/dubbing_generator/sync/aligner.py`
- `dubbing-generator/dubbing_generator/sync/words_index.py`
- `dubbing-generator/dubbing_generator/audio/mixer.py`
- `dubbing-generator/dubbing_generator/audio/stretcher.py`
- All QA, overlap resolver, drift corrector code

## 8. Rollback

Edit `DubbingConfig.tts_engine = "chatterbox"` and reinstall Chatterbox stack:
```
pip install "transformers>=5.2.0" "chatterbox-tts>=0.1.7" "resemble-perth>=1.0.1" "setuptools<81"
```
No git revert required.
