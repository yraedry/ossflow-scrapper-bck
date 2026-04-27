# syntax=docker/dockerfile:1.7
# Build context: ../python

# ----- Stage 1: s2.cpp Vulkan builder -----------------------------------
# MUST match the runtime base's GLIBC (bjj-base = Ubuntu 22.04 / CUDA 12.4
# = GLIBC 2.35). Ubuntu 24.04 builds break at runtime with `GLIBC_2.38 not
# found` because the resulting binary links symbols introduced after 2.35.
# We stay on jammy and patch the single ggml-vulkan line that breaks with
# jammy's older Vulkan-Hpp (1.3.204): `vk::PhysicalDeviceProperties2 →
# VkPhysicalDeviceProperties2` implicit conversion. `auto props = …` works
# with both old and new Vulkan-Hpp.
FROM ubuntu:22.04 AS s2cpp-builder
ARG S2CPP_REPO=https://github.com/rodrigomatta/s2.cpp
ARG S2CPP_REF=e48ce8e02d8335bd9a0ba94679f605724b31d123
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libvulkan-dev glslang-tools spirv-tools \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recurse-submodules ${S2CPP_REPO} /src/s2.cpp \
 && cd /src/s2.cpp && git checkout ${S2CPP_REF} \
 && git submodule update --init --recursive

WORKDIR /src/s2.cpp
# Patch ggml-vulkan.cpp for compatibility with jammy's Vulkan-Hpp 1.3.204.
# The unique line `VkPhysicalDeviceProperties2 props = vkdev.getProperties2();`
# fails because:
# - Vulkan-Hpp 1.3.204 returns vk::PhysicalDeviceProperties2 (no implicit
#   conversion to the C struct)
# - the pointer overload `getProperties2(&p)` doesn't exist in 1.3.204
#   either — it interprets &p as a Dispatch parameter.
# Use the raw C API instead: `static_cast<VkPhysicalDevice>(vkdev)` peels
# the C++ wrapper to call the C function directly. Works on every Vulkan
# header version.
RUN sed -i \
        -e 's|VkPhysicalDeviceProperties2 props = vkdev.getProperties2();|VkPhysicalDeviceProperties2 props{}; props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2; vkGetPhysicalDeviceProperties2(static_cast<VkPhysicalDevice>(vkdev), \&props);|' \
        /src/s2.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp \
 && grep -n "ggml_vk_intel_shader_core_count" /src/s2.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp | head -3

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON \
 && cmake --build build --parallel "$(nproc)" \
 && echo "=== Build artifacts ===" \
 && find /src/s2.cpp/build -maxdepth 4 -name "s2" -type f -executable -exec ls -la {} + \
 && find /src/s2.cpp/build -name "libggml*.so*" -exec ls -la {} +

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

# s2 binary lands in build/bin/ on Linux (the ggml subproject sets
# CMAKE_RUNTIME_OUTPUT_DIRECTORY for both itself and the parent target).
COPY --from=s2cpp-builder /src/s2.cpp/build/bin/s2 /usr/local/bin/s2
# GGML shared libs land in build/bin/ alongside s2 (libggml.so,
# libggml-base.so, libggml-cpu.so, libggml-vulkan.so + their .0 symlinks).
# Use `*` to catch all variants (libggml.so, libggml.so.0, etc).
COPY --from=s2cpp-builder /src/s2.cpp/build/bin/libggml*.so* /usr/local/lib/
RUN ldconfig \
 && echo "=== Verifying s2 binary linkage ===" \
 && ls -la /usr/local/bin/s2 /usr/local/lib/libggml*.so* \
 && ldd /usr/local/bin/s2 | grep -E "ggml|not found" \
 && /usr/local/bin/s2 --help > /dev/null && echo "s2 binary OK"

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
