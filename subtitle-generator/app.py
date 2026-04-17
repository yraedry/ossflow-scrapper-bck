"""FastAPI entrypoint for subtitle-generator backend.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_KIT_PARENT = Path(__file__).resolve().parent.parent
if str(_KIT_PARENT) not in sys.path:
    sys.path.insert(0, str(_KIT_PARENT))

from bjj_service_kit import JobEvent, RunRequest, create_app, emit_logs  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402


SERVICE_NAME = "subtitle-generator"


class ValidateRequest(BaseModel):
    srt_path: str


class RegenerateRequest(BaseModel):
    srt_path: str
    segment_idx: int
    context_seconds: float = 1.0
    video_path: str | None = None
    model: str = "large-v3"
    language: str = "en"


class ApplyRequest(BaseModel):
    srt_path: str
    segment_idx: int
    text: str
    start: float | None = None
    end: float | None = None


class TranslateRequest(BaseModel):
    srt_path: str
    target_lang: str = "ES"
    source_lang: str = "EN"
    provider: str = "openai"
    model: str | None = None
    formality: str | None = None
    api_key: str | None = None
    fallback_provider: str | None = None
    fallback_api_key: str | None = None
    out_path: str | None = None


class AnalyzeRequest(BaseModel):
    video_path: str
    language: str = "en"
    model: str = "large-v3"


_regenerator: object | None = None


def _get_regenerator(model: str, language: str):
    global _regenerator
    from subtitle_generator.config import TranscriptionConfig  # type: ignore
    from subtitle_generator.segment_regen import SegmentRegenerator  # type: ignore

    if _regenerator is None:
        _regenerator = SegmentRegenerator(
            TranscriptionConfig(model_name=model, language=language)
        )
    return _regenerator


def _resolve_input(path: Path) -> Path:
    """Validate input path exists. Returns the path as-is (file or directory)."""
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def _run_subtitle_generator(req: RunRequest, emit) -> None:
    """Bridge RunRequest -> subtitle_generator.pipeline.SubtitlePipeline.

    When ``options.translate_only=True`` runs SRT translation (EN→ES via DeepL)
    instead of transcription — reusing the same job/SSE contract.
    """
    opts = req.options or {}

    if opts.get("translate_only"):
        _run_translate_directory(req, emit)
        return

    input_path = _resolve_input(Path(req.input_path))

    level = logging.DEBUG if opts.get("verbose") else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    with emit_logs(emit, level=level):
        from subtitle_generator.config import (  # type: ignore
            DEFAULT_INITIAL_PROMPT,
            SubtitleConfig,
            TranscriptionConfig,
            generate_prompt,
        )
        from subtitle_generator.cuda_setup import setup_nvidia_dlls, setup_pytorch_safety  # type: ignore
        from subtitle_generator.pipeline import SubtitlePipeline  # type: ignore

        setup_nvidia_dlls()
        setup_pytorch_safety()

        if opts.get("prompt") is not None:
            initial_prompt = opts["prompt"]
        elif opts.get("instructor") or opts.get("topic"):
            initial_prompt = generate_prompt(instructor=opts.get("instructor"), topic=opts.get("topic"))
        else:
            initial_prompt = DEFAULT_INITIAL_PROMPT

        t_config = TranscriptionConfig(
            model_name=opts.get("model", "large-v3"),
            language=opts.get("language", "en"),
            batch_size=int(opts.get("batch_size", 4)),
            initial_prompt=initial_prompt,
        )
        s_config = SubtitleConfig()

        force = bool(opts.get("force", False))
        emit(JobEvent(type="log", data={"message":
            f"starting subtitle-generator on {input_path}"
            + (" (force overwrite)" if force else "")
        }))
        pipeline = SubtitlePipeline(t_config, s_config)
        pipeline.load_models()
        if input_path.is_file():
            pipeline.process_file(input_path, force=force)
        else:
            pipeline.process_directory(input_path, force=force)
        emit(JobEvent(type="progress", data={"pct": 100}))


def _build_translator_with_fallback(opts: dict):
    """Build (primary, fallback) translators from job options.

    Primary provider defaults to ``openai``. Fallback is used only if the
    primary raises at translate time. Fallback key is read from
    ``fallback_api_key`` or, for deepl fallback, ``DEEPL_API_KEY`` env.
    """
    from subtitle_generator.translator import make_translator  # type: ignore

    provider = (opts.get("provider") or "openai").lower()
    primary_key = opts.get("api_key")
    if not primary_key:
        if provider == "openai":
            primary_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "deepl":
            primary_key = os.environ.get("DEEPL_API_KEY")

    primary = make_translator(
        provider,
        api_key=primary_key,
        source_lang=opts.get("source_lang", "EN"),
        target_lang=opts.get("target_lang", "ES"),
        model=opts.get("model"),
        formality=opts.get("formality"),
    )

    fb_name = (opts.get("fallback_provider") or "").lower() or None
    fb = None
    if fb_name and fb_name != provider:
        fb_key = opts.get("fallback_api_key")
        if not fb_key:
            if fb_name == "openai":
                fb_key = os.environ.get("OPENAI_API_KEY")
            elif fb_name == "deepl":
                fb_key = os.environ.get("DEEPL_API_KEY")
        try:
            fb = make_translator(
                fb_name,
                api_key=fb_key,
                source_lang=opts.get("source_lang", "EN"),
                target_lang=opts.get("target_lang", "ES"),
                model=opts.get("fallback_model"),
                formality=opts.get("formality"),
            )
        except ValueError:
            fb = None
    return primary, fb


def _run_translate_directory(req: RunRequest, emit) -> None:
    """Translate every .srt under input_path to *_ES.srt."""
    opts = req.options or {}
    level = logging.DEBUG if opts.get("verbose") else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    root = Path(req.input_path)
    if root.is_file():
        if root.suffix.lower() == ".srt":
            srts = [root]
        else:
            # Video file — look for the matching .srt next to it
            candidate = root.with_suffix(".srt")
            srts = [candidate] if candidate.exists() else []
        root = root.parent
    elif root.is_dir():
        srts = sorted(root.rglob("*.srt"))
    else:
        raise FileNotFoundError(f"input_path not found: {root}")

    srts = [s for s in srts if not s.stem.endswith("_ES") and not s.stem.endswith("_ESP_DUB")]

    with emit_logs(emit, level=level):
        primary, fallback = _build_translator_with_fallback(opts)
        provider_name = (opts.get("provider") or "openai").lower()
        fb_name = (opts.get("fallback_provider") or "").lower() or None

        total = len(srts)
        emit(JobEvent(type="log", data={"message":
            f"translating {total} SRT file(s) in {root} via {provider_name}"
            + (f" (fallback: {fb_name})" if fallback else "")
        }))
        if total == 0:
            emit(JobEvent(type="progress", data={"pct": 100}))
            return

        for i, srt in enumerate(srts, 1):
            out = srt.with_name(f"{srt.stem}_ES.srt")
            if out.exists():
                emit(JobEvent(type="log", data={"message": f"skip (exists): {out.name}"}))
            else:
                try:
                    primary.translate_srt(srt, out)
                    emit(JobEvent(type="log", data={"message": f"translated: {srt.name} -> {out.name}"}))
                except Exception as exc:
                    emit(JobEvent(type="log", data={"message":
                        f"{provider_name} failed on {srt.name}: {exc}"
                    }))
                    if fallback is not None:
                        try:
                            fallback.translate_srt(srt, out)
                            emit(JobEvent(type="log", data={"message":
                                f"translated via fallback {fb_name}: {srt.name}"
                            }))
                        except Exception as exc2:
                            emit(JobEvent(type="log", data={"message":
                                f"ERROR fallback {fb_name} also failed on {srt.name}: {exc2}"
                            }))
            pct = int(i * 100 / total)
            emit(JobEvent(type="progress", data={"pct": pct}))


def _hf_cache_root() -> Path:
    return Path(
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or "/models/huggingface"
    )


def _clear_hf_locks() -> dict:
    """Delete stale .lock files inside the HuggingFace hub cache.

    Returns a summary dict with counts. Safe: only touches files under
    `<cache>/hub/.locks/` (or legacy `<cache>/.locks/`).
    """
    root = _hf_cache_root()
    candidates = [root / "hub" / ".locks", root / ".locks"]
    removed, errors = 0, []
    for base in candidates:
        if not base.exists():
            continue
        for lock in base.rglob("*.lock"):
            try:
                lock.unlink()
                removed += 1
            except Exception as exc:
                errors.append(f"{lock}: {exc}")
    return {"removed": removed, "errors": errors, "roots": [str(c) for c in candidates]}


# Best-effort cleanup at import/startup — frees locks left by a killed worker.
try:
    _startup_clean = _clear_hf_locks()
    if _startup_clean["removed"]:
        logging.getLogger(SERVICE_NAME).warning(
            "cleared %d stale HF lock(s) at startup", _startup_clean["removed"]
        )
except Exception as _exc:  # noqa: BLE001
    logging.getLogger(SERVICE_NAME).warning("HF lock cleanup at startup failed: %s", _exc)


app = create_app(service_name=SERVICE_NAME, task_fn=_run_subtitle_generator)


@app.post("/maintenance/clear-hf-locks")
def clear_hf_locks() -> dict:
    """Manually clear stale HuggingFace hub locks."""
    return _clear_hf_locks()


@app.post("/maintenance/restart")
def restart_service() -> dict:
    """Graceful self-restart: libera VRAM y reinicia el proceso.

    Docker reinicia el contenedor automáticamente (restart: unless-stopped).
    Útil para liberar VRAM tras un OOM o cancelación de job.
    """
    import threading, os, signal
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    return {"ok": True, "message": "Reiniciando subtitle-generator…"}


@app.post("/validate")
def validate(req: ValidateRequest) -> dict:
    from pathlib import Path as _P
    from subtitle_generator.config import SubtitleConfig  # type: ignore
    from subtitle_generator.quality_checker import SubtitleQualityChecker  # type: ignore
    from subtitle_generator.segment_regen import find_sibling_video  # type: ignore
    from subtitle_generator.srt_io import parse_srt  # type: ignore

    srt = _P(req.srt_path)
    if not srt.exists():
        raise HTTPException(status_code=404, detail=f"SRT no existe: {req.srt_path}")
    subs = parse_srt(srt)
    report = SubtitleQualityChecker(SubtitleConfig()).check(subs)
    sibling = find_sibling_video(srt)
    report["video_path"] = str(sibling) if sibling else None
    report["srt_path"] = str(srt)
    return report


@app.post("/regenerate-segment")
def regenerate_segment(req: RegenerateRequest) -> dict:
    from pathlib import Path as _P
    from subtitle_generator.cuda_setup import setup_nvidia_dlls, setup_pytorch_safety  # type: ignore
    from subtitle_generator.segment_regen import find_sibling_video  # type: ignore
    from subtitle_generator.srt_io import parse_srt  # type: ignore

    srt = _P(req.srt_path)
    if not srt.exists():
        raise HTTPException(status_code=404, detail=f"SRT no existe: {req.srt_path}")
    subs = parse_srt(srt)
    if req.segment_idx < 1 or req.segment_idx > len(subs):
        raise HTTPException(status_code=400, detail="segment_idx fuera de rango")

    video = _P(req.video_path) if req.video_path else find_sibling_video(srt)
    if not video or not video.exists():
        raise HTTPException(
            status_code=400,
            detail="No se encontró video hermano del SRT (pasa video_path)",
        )

    target = subs[req.segment_idx - 1]
    setup_nvidia_dlls()
    setup_pytorch_safety()
    regen = _get_regenerator(req.model, req.language)
    try:
        result = regen.regenerate(
            video, target["start"], target["end"], req.context_seconds
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Regeneración falló: {exc}")

    # Clamp timestamps so they don't overlap with neighbours
    idx = req.segment_idx - 1
    prev_end = subs[idx - 1]["end"] if idx > 0 else 0.0
    next_start = subs[idx + 1]["start"] if idx + 1 < len(subs) else float("inf")
    result["start"] = max(result["start"], prev_end)
    result["end"] = min(result["end"], next_start)

    return {
        "segment_idx": req.segment_idx,
        "old": {"text": target["text"], "start": target["start"], "end": target["end"]},
        "new": result,
        "video_path": str(video),
    }


@app.post("/apply-segment")
def apply_segment(req: ApplyRequest) -> dict:
    from pathlib import Path as _P
    from subtitle_generator.srt_io import parse_srt, write_srt_with_backup  # type: ignore

    srt = _P(req.srt_path)
    if not srt.exists():
        raise HTTPException(status_code=404, detail=f"SRT no existe: {req.srt_path}")
    subs = parse_srt(srt)
    if req.segment_idx < 1 or req.segment_idx > len(subs):
        raise HTTPException(status_code=400, detail="segment_idx fuera de rango")

    target = subs[req.segment_idx - 1]
    target["text"] = req.text.strip()
    if req.start is not None:
        target["start"] = float(req.start)
    if req.end is not None:
        target["end"] = float(req.end)

    write_srt_with_backup(srt, subs)
    return {"ok": True, "segment_idx": req.segment_idx, "backup": str(srt) + ".bak"}


@app.post("/translate")
def translate(req: TranslateRequest) -> dict:
    from pathlib import Path as _P

    srt = _P(req.srt_path)
    if not srt.exists():
        raise HTTPException(status_code=404, detail=f"SRT no existe: {req.srt_path}")

    opts = {
        "provider": req.provider,
        "model": req.model,
        "api_key": req.api_key,
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
        "formality": req.formality,
        "fallback_provider": req.fallback_provider,
        "fallback_api_key": req.fallback_api_key,
    }
    try:
        primary, fallback = _build_translator_with_fallback(opts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    out = _P(req.out_path) if req.out_path else None
    used = req.provider
    try:
        written = primary.translate_srt(srt, out)
    except Exception as exc:
        if fallback is None:
            raise HTTPException(status_code=502, detail=f"{req.provider}: {exc}")
        try:
            written = fallback.translate_srt(srt, out)
            used = req.fallback_provider or "fallback"
        except Exception as exc2:
            raise HTTPException(
                status_code=502,
                detail=f"{req.provider}: {exc} | fallback {req.fallback_provider}: {exc2}",
            )
    return {"ok": True, "out_path": str(written), "source": str(srt), "provider": used}


@app.post("/analyze")
def analyze_video(req: AnalyzeRequest) -> dict:
    """Deep diagnostic analysis of a video for subtitle generation debugging.

    Returns detailed info about audio energy, VAD decisions, transcription
    segments, and what filters would drop.
    """
    from pathlib import Path as _P
    import numpy as np

    from subtitle_generator.config import (
        DEFAULT_INITIAL_PROMPT,
        DEFAULT_HOTWORDS,
        SubtitleConfig,
        TranscriptionConfig,
    )
    from subtitle_generator.cuda_setup import setup_nvidia_dlls, setup_pytorch_safety
    from subtitle_generator.hallucination_filter import HallucinationFilter

    vp = _P(req.video_path)
    if not vp.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")

    setup_nvidia_dlls()
    setup_pytorch_safety()

    import whisperx

    t_config = TranscriptionConfig(
        model_name=req.model,
        language=req.language,
        initial_prompt=DEFAULT_INITIAL_PROMPT,
    )
    s_config = SubtitleConfig()

    # 1. Load audio
    audio = whisperx.load_audio(str(vp))
    sr = 16000
    duration = len(audio) / sr

    # 2. Energy map — RMS per second
    energy_map = []
    for sec in range(int(duration) + 1):
        start_s = sec * sr
        end_s = min((sec + 1) * sr, len(audio))
        chunk = audio[start_s:end_s]
        if len(chunk) == 0:
            energy_map.append({"sec": sec, "rms_db": -100.0})
            continue
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms_db = float(20 * np.log10(rms + 1e-10))
        energy_map.append({"sec": sec, "rms_db": round(rms_db, 1)})

    # 3. Denoise and compare
    try:
        import noisereduce as nr
        denoised = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)
        denoised = denoised.astype(np.float32)
        has_denoise = True
    except ImportError:
        denoised = audio
        has_denoise = False

    energy_map_denoised = []
    if has_denoise:
        for sec in range(int(duration) + 1):
            start_s = sec * sr
            end_s = min((sec + 1) * sr, len(audio))
            chunk = denoised[start_s:end_s]
            if len(chunk) == 0:
                energy_map_denoised.append({"sec": sec, "rms_db": -100.0})
                continue
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_db = float(20 * np.log10(rms + 1e-10))
            energy_map_denoised.append({"sec": sec, "rms_db": round(rms_db, 1)})

    # 4. Transcribe (raw, no denoise) to see what Whisper produces
    hotwords = DEFAULT_HOTWORDS
    asr_options = {
        "initial_prompt": t_config.initial_prompt,
        "hotwords": hotwords,
        "beam_size": t_config.beam_size,
        "condition_on_previous_text": t_config.condition_on_previous_text,
    }

    model = whisperx.load_model(
        t_config.model_name,
        t_config.device,
        compute_type=t_config.compute_type,
        asr_options=asr_options,
        vad_options={
            "vad_onset": t_config.vad_onset,
            "vad_offset": t_config.vad_offset,
        },
    )

    # 4a. Transcribe original audio
    result_raw = model.transcribe(audio, batch_size=t_config.batch_size, language=req.language)
    raw_segments = [
        {"start": round(s.get("start", 0), 3), "end": round(s.get("end", 0), 3), "text": s.get("text", "")}
        for s in result_raw.get("segments", [])
    ]

    # 4b. Transcribe denoised audio
    result_dn = model.transcribe(denoised, batch_size=t_config.batch_size, language=req.language)
    dn_segments = [
        {"start": round(s.get("start", 0), 3), "end": round(s.get("end", 0), 3), "text": s.get("text", "")}
        for s in result_dn.get("segments", [])
    ]

    # 5. Detect gaps in denoised transcription
    gaps = []
    sorted_segs = sorted(dn_segments, key=lambda s: s["start"])
    if sorted_segs and sorted_segs[0]["start"] > 2.0:
        gaps.append({"start": 0.0, "end": round(sorted_segs[0]["start"], 3), "duration": round(sorted_segs[0]["start"], 3)})
    for i in range(len(sorted_segs) - 1):
        gap_start = sorted_segs[i]["end"]
        gap_end = sorted_segs[i + 1]["start"]
        gap_dur = gap_end - gap_start
        if gap_dur > 2.0:
            # Check energy in gap
            g_start_s = int(gap_start * sr)
            g_end_s = int(gap_end * sr)
            g_chunk = denoised[g_start_s:g_end_s]
            g_rms = float(np.sqrt(np.mean(g_chunk ** 2))) if len(g_chunk) > 0 else 0
            g_rms_db = float(20 * np.log10(g_rms + 1e-10))
            gaps.append({
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "duration": round(gap_dur, 3),
                "energy_db": round(g_rms_db, 1),
                "likely_speech": bool(g_rms_db > -40.0),
            })
    if sorted_segs and (duration - sorted_segs[-1]["end"]) > 2.0:
        gaps.append({"start": round(sorted_segs[-1]["end"], 3), "end": round(duration, 3), "duration": round(duration - sorted_segs[-1]["end"], 3)})

    # 6. Run hallucination filter on denoised segments to see what gets dropped
    h_filter = HallucinationFilter(s_config, initial_prompt=t_config.initial_prompt)
    # Simulate — need word-level data for some filters, so do alignment first
    align_model, align_meta = whisperx.load_align_model(language_code=req.language, device=t_config.device)
    aligned_result = whisperx.align(
        result_dn.get("segments", []),
        align_model,
        align_meta,
        denoised,
        device=t_config.device,
        return_char_alignments=False,
    )
    aligned_segs = aligned_result.get("segments", [])

    # Extract aligned segments (pre-filter) for display
    aligned_display = [
        {"start": round(s.get("start", 0), 3), "end": round(s.get("end", 0), 3), "text": s.get("text", "")}
        for s in aligned_segs
    ]

    pre_filter_count = len(aligned_segs)
    filtered, dropped_segs = h_filter.filter_all(aligned_segs, audio_path=vp, return_dropped=True)
    post_filter_count = len(filtered)

    # Extract filtered segments for display
    filtered_display = [
        {"start": round(s.get("start", 0), 3), "end": round(s.get("end", 0), 3), "text": s.get("text", "")}
        for s in filtered
    ]

    # Detect gaps in FILTERED segments (what actually ends up in the SRT)
    gaps_filtered = []
    sorted_filtered = sorted(filtered_display, key=lambda s: s["start"])
    if sorted_filtered and sorted_filtered[0]["start"] > 2.0:
        gaps_filtered.append({
            "start": 0.0, "end": round(sorted_filtered[0]["start"], 3),
            "duration": round(sorted_filtered[0]["start"], 3),
        })
    for i in range(len(sorted_filtered) - 1):
        gf_start = sorted_filtered[i]["end"]
        gf_end = sorted_filtered[i + 1]["start"]
        gf_dur = gf_end - gf_start
        if gf_dur > 2.0:
            g_start_s = int(gf_start * sr)
            g_end_s = int(gf_end * sr)
            g_chunk = denoised[g_start_s:g_end_s]
            g_rms = float(np.sqrt(np.mean(g_chunk ** 2))) if len(g_chunk) > 0 else 0
            g_rms_db = float(20 * np.log10(g_rms + 1e-10))
            gaps_filtered.append({
                "start": round(gf_start, 3),
                "end": round(gf_end, 3),
                "duration": round(gf_dur, 3),
                "energy_db": round(g_rms_db, 1),
                "likely_speech": bool(g_rms_db > -40.0),
            })
    if sorted_filtered and (duration - sorted_filtered[-1]["end"]) > 2.0:
        gaps_filtered.append({
            "start": round(sorted_filtered[-1]["end"], 3),
            "end": round(duration, 3),
            "duration": round(duration - sorted_filtered[-1]["end"], 3),
        })

    # 7. Read existing SRT from disk (ground truth for gap detection)
    srt_path = vp.with_suffix(".srt")
    segments_srt: list[dict] = []
    gaps_srt: list[dict] = []
    if srt_path.exists():
        from subtitle_generator.srt_io import parse_srt
        srt_blocks = parse_srt(srt_path)
        segments_srt = [
            {"start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"]}
            for s in srt_blocks
        ]
        # Detect gaps between consecutive SRT blocks
        sorted_srt = sorted(segments_srt, key=lambda s: s["start"])
        if sorted_srt and sorted_srt[0]["start"] > 2.0:
            gaps_srt.append({
                "start": 0.0, "end": round(sorted_srt[0]["start"], 3),
                "duration": round(sorted_srt[0]["start"], 3),
            })
        for i in range(len(sorted_srt) - 1):
            gs = sorted_srt[i]["end"]
            ge = sorted_srt[i + 1]["start"]
            gd = ge - gs
            if gd > 1.0:  # 1s threshold for SRT gaps (blocks are shorter than segments)
                g_start_s = int(gs * sr)
                g_end_s = int(ge * sr)
                g_chunk = denoised[g_start_s:g_end_s]
                g_rms = float(np.sqrt(np.mean(g_chunk ** 2))) if len(g_chunk) > 0 else 0
                g_rms_db = float(20 * np.log10(g_rms + 1e-10))
                gaps_srt.append({
                    "start": round(gs, 3),
                    "end": round(ge, 3),
                    "duration": round(gd, 3),
                    "energy_db": round(g_rms_db, 1),
                    "likely_speech": bool(g_rms_db > -40.0),
                })
        if sorted_srt and (duration - sorted_srt[-1]["end"]) > 2.0:
            gaps_srt.append({
                "start": round(sorted_srt[-1]["end"], 3),
                "end": round(duration, 3),
                "duration": round(duration - sorted_srt[-1]["end"], 3),
            })

    # Cleanup GPU
    import gc, torch
    del model, align_model, align_meta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _sanitize(obj):
        """Convert numpy types to Python natives for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return _sanitize({
        "video_path": str(vp),
        "duration_seconds": round(duration, 2),
        "has_denoise": has_denoise,
        "vad_params": {
            "onset": t_config.vad_onset,
            "offset": t_config.vad_offset,
        },
        "transcription": {
            "raw_segments": len(raw_segments),
            "denoised_segments": len(dn_segments),
            "improvement": len(dn_segments) - len(raw_segments),
        },
        "segments_raw": raw_segments,
        "segments_denoised": dn_segments,
        "segments_aligned": aligned_display,
        "segments_filtered": filtered_display,
        "segments_dropped": dropped_segs,
        "segments_srt": segments_srt,
        "gaps": gaps,
        "gaps_filtered": gaps_filtered,
        "gaps_srt": gaps_srt,
        "hallucination_filter": {
            "input_segments": pre_filter_count,
            "output_segments": post_filter_count,
            "dropped": pre_filter_count - post_filter_count,
            "stats": h_filter.stats,
        },
        "energy_map": energy_map,
        "energy_map_denoised": energy_map_denoised,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
