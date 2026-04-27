# syntax=docker/dockerfile:1.7
# Build context: ../python

# ----- Stage 1: s2.cpp Vulkan builder -----------------------------------
# Ubuntu 24.04 (noble) ships Vulkan-Headers 1.3.275 + matching Vulkan-Hpp;
# 22.04 (jammy)'s 1.3.204 is too old and breaks ggml-vulkan compilation
# (vk::PhysicalDeviceProperties2 → VkPhysicalDeviceProperties2 conversion
# was removed). Builder image bump only — the compiled binary links against
# libvulkan.so.1 which is forward-compatible at runtime.
FROM ubuntu:24.04 AS s2cpp-builder
ARG S2CPP_REPO=https://github.com/rodrigomatta/s2.cpp
ARG S2CPP_REF=e48ce8e02d8335bd9a0ba94679f605724b31d123
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libvulkan-dev \
        glslang-tools glslc spirv-tools \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recurse-submodules ${S2CPP_REPO} /src/s2.cpp \
 && cd /src/s2.cpp && git checkout ${S2CPP_REF} \
 && git submodule update --init --recursive

WORKDIR /src/s2.cpp
# -DS2_VULKAN=ON is the s2.cpp option name; CMakeLists forwards it to
# GGML_VULKAN. Verified locally on 2026-04-27 against SHA e48ce8e0.
# CMAKE_RUNTIME_OUTPUT_DIRECTORY pins the s2 executable to build/bin/
# alongside the ggml shared libs (which ggml's CMakeLists already routes
# there via the same variable). On Linux single-config builds, without
# this, the s2 binary lands in build/ (root) and the COPY below misses it.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/src/s2.cpp/build/bin \
 && cmake --build build --parallel "$(nproc)" \
 && ls -la /src/s2.cpp/build/bin/

# ----- Stage 2: dubbing-generator runtime -------------------------------
FROM bjj-base:latest

# Vulkan loader only. The NVIDIA driver (passed via nvidia-container-toolkit
# with NVIDIA_DRIVER_CAPABILITIES=graphics,...) supplies its own ICD JSON.
# DO NOT install mesa-vulkan-drivers — it can preempt the NVIDIA ICD and
# force a CPU fallback that hangs synthesis silently.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        rubberband-cli wget espeak-ng \
        libvulkan1 vulkan-tools \
 && rm -rf /var/lib/apt/lists/*

COPY --from=s2cpp-builder /src/s2.cpp/build/bin/s2 /usr/local/bin/s2
# GGML is shared (4 libs verified locally on 2026-04-27 against SHA e48ce8e0):
# libggml.so, libggml-base.so, libggml-cpu.so, libggml-vulkan.so. CMake emits
# them under build/bin/ on Linux single-config builds. If a future s2.cpp
# revision flips to static linking, this glob will fail-loud at build and
# the implementer should remove this line + verify with
# `docker run --rm --entrypoint ldd <image> /usr/local/bin/s2 | grep ggml`.
COPY --from=s2cpp-builder /src/s2.cpp/build/bin/libggml*.so /usr/local/lib/
RUN ldconfig

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

RUN mkdir -p /models/piper && cd /models/piper && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json

# Pre-cache Kokoro-82M (kept as fallback engine, not the default).
RUN python3 -c "from kokoro import KPipeline; KPipeline(lang_code='e')" || true

COPY dubbing-generator /app/dubbing-generator
WORKDIR /app/dubbing-generator

# HF cache still used by torch hub (silero-vad en VoiceCloner) y demucs.
ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface

EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8003/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
