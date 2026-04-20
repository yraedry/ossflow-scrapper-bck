# syntax=docker/dockerfile:1.7
# Build context: ../python
FROM bjj-base:latest

# rubberband-cli: pitch-preserving high-quality time-stretch for dubbing
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        rubberband-cli

COPY bjj_service_kit /app/bjj_service_kit
COPY dubbing-generator/requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt \
 && pip install -e /app/bjj_service_kit \
 && pip uninstall -y pytest pytest-asyncio || true \
 && find /opt/venv -depth -type d -name "__pycache__" -exec rm -rf {} + \
 && find /opt/venv -type f -name "*.pyc" -delete \
 && find /opt/venv/lib/python3.11/site-packages -maxdepth 3 -type d \
      \( -name "tests" -o -name "test" -o -name "docs" -o -name "examples" \) \
      -exec rm -rf {} + 2>/dev/null || true

COPY dubbing-generator /app/dubbing-generator
WORKDIR /app/dubbing-generator

# Chatterbox downloads model weights to HF cache on first use (~2GB)
ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface

EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8003/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
