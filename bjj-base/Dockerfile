# syntax=docker/dockerfile:1.7
# Base compartida BJJ: CUDA + Python + deps comunes (torch, transformers, fastapi...)
# Build una vez:
#   docker build -t bjj-base:latest -f python/bjj-base/Dockerfile python/
# Reutilizada por chapter-splitter, subtitle-generator, dubbing-generator.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg curl ca-certificates \
        build-essential git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" VIRTUAL_ENV=/opt/venv

# Deps comunes pesadas — una sola vez para los 3 backends
# Usamos --extra-index-url con variante cu124 — evita que torch traiga sus
# propias copias de las libs CUDA ya presentes en el base image.
RUN --mount=type=cache,target=/root/.cache/pip,id=bjj-pip,sharing=locked \
    pip install --upgrade pip setuptools wheel \
 && pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch torchaudio \
 && pip install \
        transformers sentencepiece \
        numpy pydub \
        fastapi "uvicorn[standard]>=0.27" "sse-starlette>=2.0" \
        "pydantic>=2.5" "httpx>=0.27"

# Prune post-install: __pycache__, *.pyc, docs/tests/examples dentro de
# site-packages. Ahorra ~500MB sin afectar ejecución.
RUN find /opt/venv -depth -type d -name "__pycache__" -exec rm -rf {} + \
 && find /opt/venv -type f -name "*.pyc" -delete \
 && find /opt/venv/lib/python3.11/site-packages -maxdepth 3 -type d \
      \( -name "tests" -o -name "test" -o -name "docs" -o -name "examples" \) \
      -exec rm -rf {} + 2>/dev/null || true

# Cache de modelos fuera de la imagen — se montan como volúmenes en runtime
ENV HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    TRANSFORMERS_CACHE=/models/huggingface \
    XDG_CACHE_HOME=/models/cache
RUN mkdir -p /models/huggingface /models/torch /models/cache

WORKDIR /app
