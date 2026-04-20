"""FastAPI application factory for BJJ backends."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections import deque
from typing import Deque

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .events import sse_generator
from .runner import BaseRunner, TaskFn
from .schemas import RunRequest


class RingBufferHandler(logging.Handler):
    """In-memory log ring used by the /logs endpoint."""

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__(level=logging.DEBUG)
        self.buffer: Deque[dict] = deque(maxlen=capacity)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append({
                "timestamp": self.formatTime(record) if False else record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            })
        except Exception:  # pragma: no cover
            pass


def _install_ring_buffer() -> RingBufferHandler:
    root = logging.getLogger()
    # Avoid double-attach on reload
    for h in list(root.handlers):
        if isinstance(h, RingBufferHandler):
            _ensure_uvicorn_propagates()
            return h
    handler = RingBufferHandler()
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    _ensure_uvicorn_propagates()
    return handler


class _HealthAccessFilter(logging.Filter):
    """Drop uvicorn.access lines for noisy health/metrics endpoints.

    The pipeline UI polls /health every few seconds; without this filter the
    ring buffer and stdout drown real pipeline logs in access noise.
    """

    _NOISY_PATHS = (" /health ", " /gpu ", " /logs ")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(p in msg for p in self._NOISY_PATHS)


def _ensure_uvicorn_propagates() -> None:
    # Uvicorn loggers are non-propagating by default; make them feed the root
    # handler so our ring buffer captures request/access/error lines.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.propagate = True
        if lg.level == logging.NOTSET:
            lg.setLevel(logging.INFO)
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _HealthAccessFilter) for f in access.filters):
        access.addFilter(_HealthAccessFilter())


def _query_gpus() -> list[dict]:
    """Return list of GPUs seen by nvidia-smi, or [] if unavailable."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    gpus: list[dict] = []
    for line in (result.stdout or "").splitlines():
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 5:
            continue
        name, util, mem_used, mem_total, temp = parts[:5]
        try:
            gpus.append({
                "name": name,
                "util_percent": float(util),
                "mem_used_mb": float(mem_used),
                "mem_total_mb": float(mem_total),
                "temp_c": float(temp),
            })
        except ValueError:
            continue
    return gpus


def create_app(service_name: str, task_fn: TaskFn, *, runner: BaseRunner | None = None) -> FastAPI:
    """Build a FastAPI app with standard /health, /run, /events/{job_id} endpoints.

    Parameters
    ----------
    service_name: identifier for the backend (returned by /health).
    task_fn: the backend-specific callable (see runner.TaskFn signature).
    runner: optional pre-built BaseRunner (useful for dependency injection in tests).
    """
    app = FastAPI(title=f"BJJ {service_name}")
    app.state.service_name = service_name
    app.state.runner = runner or BaseRunner(task_fn=task_fn)
    app.state.log_buffer = _install_ring_buffer()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": service_name}

    @app.get("/gpu")
    def gpu() -> dict:
        """Expose local GPU visibility for the metrics aggregator."""
        return {"service": service_name, "gpus": _query_gpus()}

    @app.get("/logs")
    def logs(level: str | None = None, tail: int = 500) -> dict:
        """Return the in-memory log ring. Filter by level if provided."""
        buf = list(app.state.log_buffer.buffer)
        if level and level.upper() != "ALL":
            lvl = level.upper()
            buf = [r for r in buf if r.get("level") == lvl]
        if tail and tail > 0:
            buf = buf[-tail:]
        return {"service": service_name, "lines": buf, "truncated": False}

    @app.post("/run")
    def run(req: RunRequest) -> dict[str, str]:
        try:
            job_id = app.state.runner.submit(req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"job_id": job_id}

    @app.get("/events/{job_id}")
    def events(job_id: str):
        q = app.state.runner.registry.get(job_id)
        if q is None:
            raise HTTPException(status_code=404, detail="job_id not found")
        return StreamingResponse(sse_generator(q), media_type="text/event-stream")

    return app
