"""Pipeline endpoint: execute multiple processing steps sequentially.

Each step delegates to a backend microservice over HTTP (see
``api.backend_client``). If any step fails, the pipeline stops and reports
the error. Progress is streamed via SSE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from api.event_normalizer import normalize
from api.paths import to_container_path
from api.settings import get_library_path
from api.backend_client import (
    BackendClient,
    BackendError,
    dubbing_client,
    splitter_client,
    subs_client,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

BASE_DIR = Path(__file__).parent.parent  # processor-api root

from api.settings import CONFIG_DIR as _CONFIG_DIR
HISTORY_FILE = _CONFIG_DIR / "pipeline_history.json"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class StepInfo:
    name: str
    status: StepStatus = StepStatus.PENDING
    progress: float = 0.0
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    diff: Optional[dict] = None
    # Per-step options snapshot (e.g. dubbing_mode=True for translate runs
    # that generate .dub.es.srt). Lets the UI distinguish "Traducción" from
    # "Guion doblaje" even though both are the same backend step.
    options: dict = field(default_factory=dict)


@dataclass
class PipelineInfo:
    pipeline_id: str
    path: str
    steps: list[StepInfo]
    options: dict = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    current_step: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    # After `chapters` succeeds we redirect subsequent steps here — the Season
    # folder containing the freshly-split chapters. None until chapters runs.
    chained_path: Optional[str] = None
    log_buffer: list[dict] = field(default_factory=list)
    # Monotonic sequence counter for events (used for client-side dedupe).
    event_seq: int = 0


# In-memory store
_pipelines: dict[str, PipelineInfo] = {}
# Per-pipeline list of subscriber queues. Fan-out: each SSE client gets its
# own queue so multiple consumers (e.g. StrictMode double-mount, reconnect
# while a previous EventSource is still draining) do NOT steal events from
# each other. A single shared queue caused "missing live logs" in LogPanel.
_pipeline_subscribers: dict[str, list[asyncio.Queue]] = {}
_pipeline_tasks: dict[str, asyncio.Task] = {}
_pipeline_cancel: dict[str, bool] = {}


def _subscribe(pipeline_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _pipeline_subscribers.setdefault(pipeline_id, []).append(q)
    return q


def _unsubscribe(pipeline_id: str, q: asyncio.Queue) -> None:
    subs = _pipeline_subscribers.get(pipeline_id)
    if not subs:
        return
    try:
        subs.remove(q)
    except ValueError:
        pass
    if not subs:
        _pipeline_subscribers.pop(pipeline_id, None)


def _serialize(p: PipelineInfo) -> dict:
    return {
        "pipeline_id": p.pipeline_id,
        "path": p.path,
        "options": p.options,
        "status": p.status.value,
        "current_step": p.current_step,
        "created_at": p.created_at,
        "completed_at": p.completed_at,
        "steps": [
            {
                "name": s.name, "status": s.status.value, "progress": s.progress,
                "message": s.message, "started_at": s.started_at,
                "completed_at": s.completed_at, "diff": s.diff,
            } for s in p.steps
        ],
    }


import threading
import time

# Debounce: coalesce bursts of _save_history calls (common during a run with
# many SSE events) into a single write at most every _SAVE_MIN_INTERVAL
# seconds. A trailing write is always scheduled so the final state lands on
# disk. Write runs in a daemon thread so the asyncio loop is never blocked.
_SAVE_MIN_INTERVAL = 2.0
_save_lock = threading.Lock()
_save_last_write = 0.0
_save_timer: Optional[threading.Timer] = None


def _write_history_sync() -> None:
    """Actual disk write (runs off the event loop)."""
    global _save_last_write
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(_pipelines.values(), key=lambda p: p.created_at, reverse=True)[:200]
        payload = json.dumps([_serialize(p) for p in items], indent=2, ensure_ascii=False)
        HISTORY_FILE.write_text(payload, encoding="utf-8")
        _save_last_write = time.monotonic()
    except OSError as exc:
        log.warning("Failed to save pipeline history: %s", exc)


def _save_history() -> None:
    """Non-blocking, debounced history save.

    Decision: combined *fire-and-forget thread* + *debounce (2 s)*. A single
    background daemon thread performs the JSON serialization and write, so
    the event loop is never stalled (diagnosis section 2). Bursty calls are
    coalesced by scheduling a trailing threading.Timer if the previous write
    happened < 2 s ago; this collapses dozens of step_progress-triggered
    saves into one write while guaranteeing the final state reaches disk.
    """
    global _save_timer
    now = time.monotonic()
    with _save_lock:
        elapsed = now - _save_last_write
        if elapsed >= _SAVE_MIN_INTERVAL:
            # Cancel any pending trailing write — this one supersedes it.
            if _save_timer is not None:
                _save_timer.cancel()
                _save_timer = None
            threading.Thread(target=_write_history_sync, daemon=True).start()
        else:
            # Schedule a single trailing write; drop redundant schedules.
            if _save_timer is None or not _save_timer.is_alive():
                delay = _SAVE_MIN_INTERVAL - elapsed
                t = threading.Timer(delay, _write_history_sync)
                t.daemon = True
                _save_timer = t
                t.start()


def _load_history() -> None:
    if not HISTORY_FILE.exists():
        return
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to load pipeline history: %s", exc)
        return
    for d in data:
        steps = [
            StepInfo(
                name=s["name"],
                status=StepStatus(s.get("status", "pending")),
                progress=s.get("progress", 0.0),
                message=s.get("message", ""),
                started_at=s.get("started_at"),
                completed_at=s.get("completed_at"),
                diff=s.get("diff"),
            ) for s in d.get("steps", [])
        ]
        p = PipelineInfo(
            pipeline_id=d["pipeline_id"],
            path=d["path"],
            steps=steps,
            options=d.get("options", {}),
            status=StepStatus(d.get("status", "pending")),
            current_step=d.get("current_step", 0),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            completed_at=d.get("completed_at"),
        )
        # Mark stale running pipelines as failed (server restarted mid-run)
        if p.status in (StepStatus.RUNNING, StepStatus.PENDING):
            p.status = StepStatus.FAILED
            p.completed_at = p.completed_at or datetime.now(timezone.utc).isoformat()
            for s in p.steps:
                if s.status == StepStatus.RUNNING:
                    s.status = StepStatus.FAILED
        _pipelines[p.pipeline_id] = p


_load_history()

# Valid step names
VALID_STEPS = {"chapters", "subtitles", "translate", "dubbing"}
# Canonical order for sorting a user-supplied list.
STEP_ORDER = ["chapters", "subtitles", "translate", "dubbing"]


# ---------------------------------------------------------------------------
# Step execution helpers
# ---------------------------------------------------------------------------

SIDECAR_NAME = ".bjj-meta.json"


def _load_oracle_for_path(host_path: str) -> Optional[dict]:
    """Read oracle data from the instructional's .bjj-meta.json sidecar."""
    p = Path(host_path)
    folder = p if p.is_dir() else p.parent
    sidecar = folder / SIDECAR_NAME
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data.get("oracle") if isinstance(data, dict) else None
    except (OSError, ValueError) as exc:
        log.warning("Failed to read oracle from %s: %s", sidecar, exc)
        return None


def _load_voice_profile_for_path(host_path: str) -> str:
    """Walk up from video path looking for .bjj-meta.json with voice_profile.

    Videos live deep inside the tree (Season_NN / chapter files), so the
    sidecar is typically 1-2 levels up. Empty string means "clone instructor".
    """
    p = Path(host_path)
    current = p if p.is_dir() else p.parent
    for _ in range(4):  # at most 4 levels up (Season / chapters / file)
        sidecar = current / SIDECAR_NAME
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    vp = data.get("voice_profile")
                    if isinstance(vp, str) and vp:
                        return vp
            except (OSError, ValueError) as exc:
                log.warning("Failed to read voice_profile from %s: %s", sidecar, exc)
        if current.parent == current:
            break
        current = current.parent
    return ""


def _client_and_payload(
    step_name: str,
    path: str,
    options: dict,
    chained_path: Optional[str] = None,
) -> tuple[BackendClient, dict, bool]:
    """Map a step name to (client, payload, use_oracle) for the target microservice.

    ``chained_path`` — cuando chapters ya creó la Season_NN/, los pasos
    posteriores (subs/translate/dubbing) deben operar sobre esa carpeta, no
    sobre el original, porque los capítulos son los ficheros reales a
    subtitular/doblar. Chapters siempre usa el ``path`` original.

    When ``options["mode"] == "oracle"`` and step is ``chapters``, reads the
    oracle data from ``.bjj-meta.json`` and returns a payload for ``/run-oracle``.
    The third element of the returned tuple is ``True`` in that case.
    """
    effective = path if step_name == "chapters" else (chained_path or path)
    lib = get_library_path() or ""
    container_path = to_container_path(effective, lib) if lib else effective
    video_dir = container_path.rsplit("/", 1)[0] if "." in container_path.rsplit("/", 1)[-1] else container_path
    if not video_dir:
        video_dir = "/library"
    # User-supplied output_dir overrides the default (allows non-destructive runs)
    user_out = options.get("output_dir")
    if user_out:
        out_dir = to_container_path(user_out, lib) if lib else user_out
    else:
        out_dir = video_dir
    base = {"input_path": container_path, "output_dir": out_dir}

    if step_name == "chapters":
        # Oracle mode: read oracle data and use /run-oracle endpoint
        if options.get("mode") == "oracle":
            oracle_data = _load_oracle_for_path(path)
            if oracle_data:
                return splitter_client(), {
                    "path": container_path,
                    "oracle": oracle_data,
                    "output_dir": out_dir,
                }, True
            raise ValueError(
                f"Oracle mode requested but no oracle data found for '{path}'. "
                "Run Oracle first from the instructional detail page."
            )

        return splitter_client(), {
            **base,
            "options": {
                "dry_run": bool(options.get("dry_run", False)),
                "verbose": True,
            },
        }, False
    if step_name == "subtitles":
        sub_opts: dict = {"verbose": True}
        if options.get("force"):
            sub_opts["force"] = True
        return subs_client(), {**base, "options": sub_opts}, False
    if step_name == "translate":
        from api.settings import get_setting

        provider = (options.get("provider") or get_setting("translation_provider") or "openai").lower()
        fallback = (
            options.get("fallback_provider")
            or get_setting("translation_fallback_provider")
            or ""
        ).lower() or None
        model = options.get("model") or get_setting("translation_model")

        topts: dict = {
            "translate_only": True,
            "verbose": True,
            "target_lang": options.get("target_lang", "ES"),
            "source_lang": options.get("source_lang", "EN"),
            "provider": provider,
        }
        if model:
            topts["model"] = model
        if options.get("formality"):
            topts["formality"] = options["formality"]

        # Dubbing-adapted track (.dub.es.srt). Explicit option wins; else
        # pull from global setting. Required so the dubbing step can pick
        # the speech-anchored SRT (nivel 3) instead of the reading one.
        if "dubbing_mode" in options:
            dub_on = bool(options["dubbing_mode"])
        else:
            dub_on = bool(get_setting("translation_dubbing_mode"))
        if dub_on:
            topts["dubbing_mode"] = True
            cps = options.get("dubbing_cps") or get_setting("translation_dubbing_cps")
            if cps:
                topts["dubbing_cps"] = float(cps)

        key = options.get("api_key") or (
            get_setting("openai_api_key") if provider == "openai"
            else get_setting("deepl_api_key") if provider == "deepl"
            else None
        )
        if key:
            topts["api_key"] = key

        if fallback and fallback != provider:
            fb_key = options.get("fallback_api_key") or (
                get_setting("openai_api_key") if fallback == "openai"
                else get_setting("deepl_api_key") if fallback == "deepl"
                else None
            )
            if fb_key:
                topts["fallback_provider"] = fallback
                topts["fallback_api_key"] = fb_key

        return subs_client(), {**base, "options": topts}, False
    if step_name == "dubbing":
        opts: dict = {"skip_translation": True}
        # Explicit voice_profile in options wins; else fall back to the one
        # stored in the instructional's sidecar. Empty = clone instructor.
        vp = options.get("voice_profile") or _load_voice_profile_for_path(path)
        if vp:
            opts["voice_profile"] = vp
            opts["use_model_voice"] = True
        elif options.get("use_model_voice", False):
            opts["use_model_voice"] = True
        return dubbing_client(), {**base, "options": opts}, False
    raise ValueError(f"Unknown step: {step_name}")


def _target_dir(pipeline: PipelineInfo) -> Optional[Path]:
    """Host directory to snapshot for diffing (output_dir or video dir)."""
    out = pipeline.options.get("output_dir")
    if out:
        p = Path(out)
    else:
        pp = Path(pipeline.path)
        p = pp if pp.is_dir() else pp.parent
    try:
        if p.exists() and p.is_dir():
            return p
    except OSError:
        return None
    return None


def _snapshot_dir(base: Optional[Path]) -> dict[str, tuple[int, float]]:
    """Return {relative_path: (size, mtime)} for all files under base."""
    if base is None:
        return {}
    out: dict[str, tuple[int, float]] = {}
    try:
        for f in base.rglob("*"):
            try:
                if not f.is_file():
                    continue
                st = f.stat()
                rel = f.relative_to(base).as_posix()
                out[rel] = (st.st_size, st.st_mtime)
            except OSError:
                continue
    except OSError:
        return {}
    return out


def _compute_diff(
    before: dict[str, tuple[int, float]],
    after: dict[str, tuple[int, float]],
    limit: int = 200,
) -> dict:
    added_all = [p for p in after if p not in before]
    removed_all = [p for p in before if p not in after]
    modified_all = [
        p for p in after
        if p in before and (
            before[p][0] != after[p][0] or abs(before[p][1] - after[p][1]) > 0.001
        )
    ]
    def _trunc(lst):
        return lst[:limit], len(lst) > limit
    added, a_t = _trunc(sorted(added_all))
    modified, m_t = _trunc(sorted(modified_all))
    removed, r_t = _trunc(sorted(removed_all))
    return {
        "added": added,
        "modified": modified,
        "removed": removed,
        "truncated": a_t or m_t or r_t,
    }


_VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
_SEASON_DIR_RE = re.compile(r"^Season\s*\d+$", re.IGNORECASE)


def _detect_season_folder(target: Optional[Path], added: list[str]) -> Optional[str]:
    """Given the diff ``added`` list (paths relative to target_dir), return the
    absolute host path of the Season folder where new chapters landed.

    Heuristic: group added video files by their immediate parent directory;
    prefer parents whose name matches ``Season NN``; break ties by file count.
    """
    if target is None or not added:
        return None
    from collections import Counter
    parents: Counter = Counter()
    for rel in added:
        if not rel.lower().endswith(_VIDEO_EXTS):
            continue
        parts = rel.rsplit("/", 1)
        if len(parts) != 2:
            continue
        parents[parts[0]] += 1
    if not parents:
        return None
    # Prefer "Season NN" directories.
    season_like = {p: c for p, c in parents.items() if _SEASON_DIR_RE.match(Path(p).name)}
    chosen = max(season_like or parents, key=lambda k: (season_like or parents)[k])
    return str(target / chosen)


async def _emit(pipeline: PipelineInfo, queue: asyncio.Queue, event: dict) -> None:
    """Append to the persistent buffer (capped) and broadcast to all live
    subscribers.

    The ``queue`` argument is kept for backwards compatibility with call sites
    but is ignored — we always fan-out to every subscriber registered for
    this pipeline, so no consumer can "steal" events from another.
    Each event is tagged with a monotonic ``seq`` for client-side dedupe
    across reconnects (buffer replay would otherwise re-deliver events the
    client already has).
    """
    pipeline.event_seq += 1
    event = {**event, "seq": pipeline.event_seq}
    pipeline.log_buffer.append(event)
    if len(pipeline.log_buffer) > 2000:
        del pipeline.log_buffer[: len(pipeline.log_buffer) - 2000]
    for q in list(_pipeline_subscribers.get(pipeline.pipeline_id, [])):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover - unbounded queues
            pass


async def _run_step(
    pipeline: PipelineInfo,
    step_index: int,
    queue: asyncio.Queue,
) -> bool:
    """Run a single pipeline step via its backend microservice."""
    step = pipeline.steps[step_index]
    step.status = StepStatus.RUNNING
    step.started_at = datetime.now(timezone.utc).isoformat()
    pipeline.current_step = step_index

    # Snapshot target dir BEFORE running (for post-run diff)
    target = _target_dir(pipeline)
    before_snap = _snapshot_dir(target)
    diff_emitted = {"done": False}

    async def _finalize_diff() -> None:
        if diff_emitted["done"]:
            return
        diff_emitted["done"] = True
        after_snap = _snapshot_dir(target)
        diff = _compute_diff(before_snap, after_snap)
        step.diff = diff
        # Chain: tras chapters, si detectamos una Season_NN/ creada, los
        # pasos siguientes operarán sobre esa carpeta (donde viven los
        # capítulos reales a subtitular/doblar).
        if step.name == "chapters" and step.status == StepStatus.COMPLETED:
            season_path = _detect_season_folder(target, diff.get("added", []))
            if season_path:
                pipeline.chained_path = season_path
                await _emit(pipeline, queue, {
                    "type": "log",
                    "data": {"message": f"Pipeline chained to: {season_path}"},
                })
        await _emit(pipeline, queue, {
            "type": "step_diff",
            "step": step.name,
            "step_index": step_index,
            **diff,
        })

    await _emit(pipeline, queue,{
        "type": "step_started",
        "step": step.name,
        "step_index": step_index,
        "total_steps": len(pipeline.steps),
        "progress": 0,
    })

    try:
        client, payload, use_oracle = _client_and_payload(
            step.name, pipeline.path, pipeline.options, pipeline.chained_path
        )
        log.info("[pipeline:%s] Delegating %s to %s%s", pipeline.pipeline_id, step.name, client.base_url,
                 " (oracle)" if use_oracle else "")

        remote_id = await (client.run_oracle(payload) if use_oracle else client.run(payload))

        async for evt in client.stream(remote_id):
            # evt is a NormalizedEvent. Be tolerant of test doubles.
            if isinstance(evt, dict):
                evt = normalize(evt)
            msg = evt.message or ""
            if msg:
                step.message = msg
            if evt.progress is not None:
                step.progress = evt.progress

            if evt.kind == "error":
                step.status = StepStatus.FAILED
                step.completed_at = datetime.now(timezone.utc).isoformat()
                step.message = evt.message or "backend error"
                await _emit(pipeline, queue,{
                    "type": "step_failed",
                    "step": step.name,
                    "step_index": step_index,
                    "message": step.message,
                })
                return False

            if evt.kind == "done":
                step.status = StepStatus.COMPLETED
                step.progress = 100.0
                step.completed_at = datetime.now(timezone.utc).isoformat()
                await _emit(pipeline, queue,{
                    "type": "step_completed",
                    "step": step.name,
                    "step_index": step_index,
                    "progress": 100,
                })
                return True

            if msg or evt.progress is not None:
                await _emit(pipeline, queue,{
                    "type": "step_progress",
                    "step": step.name,
                    "step_index": step_index,
                    "message": msg,
                    "progress": step.progress,
                })

        # Stream ended cleanly w/o terminal event -> treat as success
        step.status = StepStatus.COMPLETED
        step.progress = 100.0
        step.completed_at = datetime.now(timezone.utc).isoformat()
        await _emit(pipeline, queue,{
            "type": "step_completed",
            "step": step.name,
            "step_index": step_index,
            "progress": 100,
        })
        return True

    except asyncio.CancelledError:
        step.status = StepStatus.CANCELLED
        step.completed_at = datetime.now(timezone.utc).isoformat()
        step.message = "cancelled by user"
        await _emit(pipeline, queue,{
            "type": "step_failed",
            "step": step.name,
            "step_index": step_index,
            "message": "cancelled by user",
        })
        raise
    except BackendError as exc:
        step.status = StepStatus.FAILED
        step.completed_at = datetime.now(timezone.utc).isoformat()
        step.message = f"backend error: {exc}"
        await _emit(pipeline, queue,{
            "type": "step_failed",
            "step": step.name,
            "step_index": step_index,
            "message": step.message,
        })
        return False
    except Exception as exc:
        step.status = StepStatus.FAILED
        step.completed_at = datetime.now(timezone.utc).isoformat()
        step.message = str(exc)
        await _emit(pipeline, queue,{
            "type": "step_failed",
            "step": step.name,
            "step_index": step_index,
            "message": str(exc),
        })
        return False
    finally:
        try:
            await _finalize_diff()
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to compute step diff: %s", exc)


async def _flush_gpu_after_step(pipeline: PipelineInfo, queue: asyncio.Queue) -> None:
    """Restart subtitle-generator to free VRAM after a GPU-heavy step."""
    import httpx
    subs_url = subs_client().base_url
    log.info("[pipeline:%s] Flushing GPU (restarting subtitle-generator)…", pipeline.pipeline_id)
    await _emit(pipeline, queue, {"type": "log", "message": "Liberando VRAM…"})
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{subs_url}/maintenance/restart")
    except Exception:
        pass  # service kills itself before responding
    for _ in range(30):
        await asyncio.sleep(2.0)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{subs_url}/health")
                if r.status_code == 200:
                    log.info("[pipeline:%s] subtitle-generator healthy again.", pipeline.pipeline_id)
                    return
        except Exception:
            pass
    log.warning("[pipeline:%s] subtitle-generator did not recover in 60s after GPU flush.", pipeline.pipeline_id)


async def _run_pipeline(pipeline: PipelineInfo, queue: asyncio.Queue) -> None:
    """Execute all steps in a pipeline sequentially."""
    pipeline.status = StepStatus.RUNNING
    await _emit(pipeline, queue,{"type": "pipeline_started", "pipeline_id": pipeline.pipeline_id})

    try:
        for i, step in enumerate(pipeline.steps):
            if _pipeline_cancel.get(pipeline.pipeline_id):
                for j in range(i, len(pipeline.steps)):
                    pipeline.steps[j].status = StepStatus.CANCELLED
                pipeline.status = StepStatus.CANCELLED
                pipeline.completed_at = datetime.now(timezone.utc).isoformat()
                await _emit(pipeline, queue,{
                    "type": "pipeline_failed",
                    "pipeline_id": pipeline.pipeline_id,
                    "message": "cancelled by user",
                })
                return
            success = await _run_step(pipeline, i, queue)
            # After subtitles/dubbing step, flush GPU before next step to prevent OOM
            if success and step.name in ("subtitles", "dubbing"):
                await _flush_gpu_after_step(pipeline, queue)
            if not success:
                for j in range(i + 1, len(pipeline.steps)):
                    pipeline.steps[j].status = StepStatus.SKIPPED
                pipeline.status = StepStatus.FAILED
                pipeline.completed_at = datetime.now(timezone.utc).isoformat()
                await _emit(pipeline, queue,{
                    "type": "pipeline_failed",
                    "pipeline_id": pipeline.pipeline_id,
                    "failed_step": step.name,
                    "message": step.message,
                })
                return

        pipeline.status = StepStatus.COMPLETED
        pipeline.completed_at = datetime.now(timezone.utc).isoformat()
        await _emit(pipeline, queue,{
            "type": "pipeline_completed",
            "pipeline_id": pipeline.pipeline_id,
        })
    except asyncio.CancelledError:
        pipeline.status = StepStatus.CANCELLED
        pipeline.completed_at = datetime.now(timezone.utc).isoformat()
        for s in pipeline.steps:
            if s.status in (StepStatus.PENDING, StepStatus.RUNNING):
                s.status = StepStatus.CANCELLED
        await _emit(pipeline, queue,{
            "type": "pipeline_failed",
            "pipeline_id": pipeline.pipeline_id,
            "message": "cancelled by user",
        })
    finally:
        _pipeline_cancel.pop(pipeline.pipeline_id, None)
        _pipeline_tasks.pop(pipeline.pipeline_id, None)
        _save_history()
        # Refresh scan cache so the UI reflects new/changed files
        _refresh_scan_cache_for(pipeline.path)


def _refresh_scan_cache_for(pipeline_path: str) -> None:
    """Re-discover videos in the instructional folder and update the scan cache."""
    try:
        from api.scan_cache import ScanCache
        from api.settings import CONFIG_DIR
        from api.library_refresh import rediscover_instructional

        cache = ScanCache(CONFIG_DIR / "library.json")
        data = cache.load()
        if not data:
            return
        items = data.get("instructionals", []) if isinstance(data, dict) else []

        # Find the instructional that contains this path
        p = Path(pipeline_path)
        folder = p if p.is_dir() else p.parent
        # Walk up to find the instructional root (direct child of library path)
        lib = get_library_path()
        if lib:
            lib_p = Path(lib)
            while folder.parent != lib_p and folder.parent != folder:
                folder = folder.parent

        folder_str = str(folder)
        match = next(
            (it for it in items if it.get("path") and
             (it["path"] == folder_str or Path(it["path"]).resolve() == Path(folder_str).resolve())),
            None,
        )
        if not match:
            folder_name = folder.name
            match = next((it for it in items if it.get("name") == folder_name), None)

        if match:
            rediscover_instructional(match)
            cache.save(items)
            log.info("Scan cache refreshed for %s after pipeline", match.get("name"))
    except Exception:
        log.warning("Failed to refresh scan cache after pipeline", exc_info=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _duration_seconds(started: Optional[str], completed: Optional[str]) -> Optional[float]:
    if not started or not completed:
        return None
    try:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(completed)
        delta = (t1 - t0).total_seconds()
        return delta if delta > 0 else None
    except (ValueError, TypeError):
        return None


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def _total_video_duration(path_str: str) -> Optional[float]:
    """Resolve total video duration in seconds for a file or a directory."""
    try:
        from api.app import get_video_info  # lazy to avoid circular import
    except Exception:  # noqa: BLE001
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    if p.is_file():
        info = get_video_info(str(p))
        d = info.get("duration") if isinstance(info, dict) else 0
        return float(d) if d else None
    exts = {".mkv", ".mp4", ".avi", ".mov", ".webm"}
    total = 0.0
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            info = get_video_info(str(f))
            d = info.get("duration") if isinstance(info, dict) else 0
            if d:
                total += float(d)
    return total if total > 0 else None


@router.post("/flush-gpu")
async def flush_gpu():
    """Restart subtitle-generator to free VRAM, then wait until healthy (max 60s)."""
    import httpx

    subs_url = subs_client().base_url

    # Fire restart — service dies before responding, so ignore errors
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{subs_url}/maintenance/restart")
    except Exception:
        pass  # expected: service kills itself mid-response

    # Poll /health until up (max 60s)
    for _ in range(30):
        await asyncio.sleep(2.0)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{subs_url}/health")
                if r.status_code == 200:
                    return {"ok": True, "message": "subtitle-generator restarted and healthy"}
        except Exception:
            pass

    return JSONResponse({"ok": False, "message": "subtitle-generator did not recover in 60s"}, status_code=503)


@router.get("/eta")
async def pipeline_eta(
    steps: str = "",
    video_duration_sec: Optional[float] = None,
    path: Optional[str] = None,
):
    """Estimate per-step and total ETA from historical completed pipelines."""
    requested = [s.strip() for s in steps.split(",") if s.strip()]
    if not requested:
        requested = sorted(VALID_STEPS)
    invalid = [s for s in requested if s not in VALID_STEPS]
    if invalid:
        return JSONResponse(
            {"error": f"Invalid steps: {invalid}. Valid: {sorted(VALID_STEPS)}"},
            status_code=422,
        )

    if video_duration_sec is None and path:
        video_duration_sec = _total_video_duration(path)

    MIN_SAMPLES = 3
    WINDOW = 20

    by_step: dict[str, list[float]] = {s: [] for s in requested}
    ordered = sorted(_pipelines.values(), key=lambda p: p.created_at, reverse=True)
    for pipe in ordered:
        for s in pipe.steps:
            if s.name not in by_step:
                continue
            if s.status != StepStatus.COMPLETED:
                continue
            dur = _duration_seconds(s.started_at, s.completed_at)
            if dur is None:
                continue
            by_step[s.name].append(dur)

    per_step: dict[str, Optional[float]] = {}
    total = 0.0
    total_known = True
    for name in requested:
        samples = by_step[name][:WINDOW]
        if len(samples) < MIN_SAMPLES:
            per_step[name] = None
            total_known = False
            continue
        est = _median(samples)
        per_step[name] = est
        total += est

    return {
        "per_step": per_step,
        "total_seconds": total if total_known else None,
        "video_duration_sec": video_duration_sec,
        "sample_counts": {k: len(by_step[k]) for k in requested},
    }


@router.post("")
async def create_pipeline(request: Request):
    """Create and launch a processing pipeline.

    Body::

        {
            "path": "/data/instructionals/Some Instructional/video.mkv",
            "steps": ["chapters", "subtitles", "translate", "dubbing"],
            "options": { "dry_run": false, "voice_profile": "gordon_ryan" }
        }
    """
    body = await request.json()
    path = body.get("path", "")
    steps_raw = body.get("steps", [])
    options = body.get("options", {})

    if not path:
        return JSONResponse({"error": "Missing 'path'"}, status_code=400)
    if not Path(path).exists():
        return JSONResponse(
            {"error": f"Path not accessible: {path}"},
            status_code=422,
        )
    if not steps_raw:
        return JSONResponse({"error": "Missing 'steps' list"}, status_code=400)

    # Validate step names
    invalid = [s for s in steps_raw if s not in VALID_STEPS]
    if invalid:
        return JSONResponse(
            {"error": f"Invalid steps: {invalid}. Valid: {sorted(VALID_STEPS)}"},
            status_code=422,
        )

    # Reject if a GPU step is already running (prevents VRAM OOM)
    GPU_STEPS = {"subtitles", "dubbing"}
    requested_gpu = GPU_STEPS.intersection(steps_raw)
    if requested_gpu:
        for p in _pipelines.values():
            if p.status == StepStatus.RUNNING:
                active_gpu = GPU_STEPS.intersection(s.name for s in p.steps)
                if active_gpu:
                    return JSONResponse(
                        {
                            "error": "GPU ocupada",
                            "detail": f"Pipeline {p.pipeline_id} ya está usando GPU ({', '.join(sorted(active_gpu))}). Espera a que termine.",
                            "active_pipeline_id": p.pipeline_id,
                        },
                        status_code=409,
                    )

    pipeline_id = str(uuid.uuid4())[:8]
    steps = [StepInfo(name=s) for s in steps_raw]
    pipeline = PipelineInfo(
        pipeline_id=pipeline_id,
        path=path,
        steps=steps,
        options=options,
    )
    _pipelines[pipeline_id] = pipeline
    # Producer does not need its own queue: _emit broadcasts to subscribers.
    queue: asyncio.Queue = asyncio.Queue()

    # Launch pipeline in background
    task = asyncio.create_task(_run_pipeline(pipeline, queue))
    _pipeline_tasks[pipeline_id] = task
    _save_history()

    return {
        "pipeline_id": pipeline_id,
        "steps": [s.name for s in steps],
        "status": pipeline.status.value,
    }


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get the current state of a pipeline."""
    pipeline = _pipelines.get(pipeline_id)
    if not pipeline:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)
    return {
        "pipeline_id": pipeline.pipeline_id,
        "path": pipeline.path,
        "status": pipeline.status.value,
        "current_step": pipeline.current_step,
        "steps": [
            {
                "name": s.name,
                "status": s.status.value,
                "progress": s.progress,
                "message": s.message,
                "started_at": s.started_at,
                "completed_at": s.completed_at,
                "diff": s.diff,
            }
            for s in pipeline.steps
        ],
        "options": pipeline.options,
        "created_at": pipeline.created_at,
        "completed_at": pipeline.completed_at,
    }


@router.get("/{pipeline_id}/events")
async def pipeline_events(pipeline_id: str):
    """SSE endpoint for real-time pipeline progress."""
    pipeline = _pipelines.get(pipeline_id)
    if not pipeline:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)

    # Each SSE consumer gets its own queue (fan-out). This prevents multiple
    # clients (or a stale retry) from stealing events from each other.
    queue = _subscribe(pipeline_id)

    # Snapshot existing buffer so a reconnecting client gets full history.
    # Replayed events carry their original ``seq`` so the client can dedupe.
    replay = list(pipeline.log_buffer)
    replay_max_seq = max((e.get("seq", 0) for e in replay), default=0)
    terminal = pipeline.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.CANCELLED)

    async def event_stream():
        try:
            for evt in replay:
                yield f"data: {json.dumps(evt)}\n\n"
            if terminal:
                return
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                # Skip events that were already flushed as part of the replay
                # snapshot (producer emitted them before we subscribed but the
                # broadcast still enqueued them into our fresh subscriber
                # queue in the tiny window between snapshot and subscribe).
                if data.get("seq", 0) and data["seq"] <= replay_max_seq:
                    continue
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("type") in ("pipeline_completed", "pipeline_failed"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            _unsubscribe(pipeline_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{pipeline_id}/cancel")
async def cancel_pipeline(pipeline_id: str):
    """Request cancellation of a running pipeline."""
    pipeline = _pipelines.get(pipeline_id)
    if not pipeline:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)
    if pipeline.status not in (StepStatus.RUNNING, StepStatus.PENDING):
        return JSONResponse(
            {"error": f"Pipeline is {pipeline.status.value}, cannot cancel"},
            status_code=409,
        )
    _pipeline_cancel[pipeline_id] = True
    task = _pipeline_tasks.get(pipeline_id)
    if task and not task.done():
        task.cancel()
    return {"pipeline_id": pipeline_id, "status": "cancelling"}


@router.post("/{pipeline_id}/retry")
async def retry_pipeline(pipeline_id: str):
    """Re-run a finished pipeline using its original path/steps/options."""
    src = _pipelines.get(pipeline_id)
    if not src:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)
    if src.status in (StepStatus.RUNNING, StepStatus.PENDING):
        return JSONResponse(
            {"error": "Pipeline still active, cannot retry"},
            status_code=409,
        )
    new_id = str(uuid.uuid4())[:8]
    new_steps = [StepInfo(name=s.name) for s in src.steps]
    new_pipe = PipelineInfo(
        pipeline_id=new_id,
        path=src.path,
        steps=new_steps,
        options=dict(src.options),
    )
    _pipelines[new_id] = new_pipe
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(_run_pipeline(new_pipe, queue))
    _pipeline_tasks[new_id] = task
    return {
        "pipeline_id": new_id,
        "retried_from": pipeline_id,
        "steps": [s.name for s in new_steps],
        "status": new_pipe.status.value,
    }


@router.get("")
async def list_pipelines(
    response: Response,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
):
    """List pipelines (summary payload, sorted by created_at desc).

    Query params:
        limit   — max entries to return (default 50).
        offset  — pagination offset (default 0).
        status  — optional filter (pending/running/completed/failed/cancelled).

    Response includes an ``X-Total-Count`` header with the *filtered* total.
    Only summary fields are returned per pipeline — for the complete payload
    (step ``diff``, ``message``, timestamps) use ``GET /api/pipeline/{id}``.
    """
    # Clamp
    if limit < 1:
        limit = 1
    if limit > 500:
        limit = 500
    if offset < 0:
        offset = 0

    ordered = sorted(_pipelines.values(), key=lambda p: p.created_at, reverse=True)
    if status:
        ordered = [p for p in ordered if p.status.value == status]
    total = len(ordered)
    page = ordered[offset : offset + limit]

    response.headers["X-Total-Count"] = str(total)
    return {
        "pipelines": [
            {
                "pipeline_id": p.pipeline_id,
                "path": p.path,
                "status": p.status.value,
                "created_at": p.created_at,
                "completed_at": p.completed_at,
                "steps": [
                    {
                        "name": s.name,
                        "status": s.status.value,
                        "progress": s.progress,
                    }
                    for s in p.steps
                ],
            }
            for p in page
        ]
    }
