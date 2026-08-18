# Qwen3.8-27B (AWQ-INT4) + DSpark speculative decoding on RTX 3090s
# Validated on 1x / 2x / 4x RTX 3090 (Ampere sm_86), driver 610.x, CUDA 13 runtime (pip wheels)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# python 3.12 (sglang requirement), build toolchain for JIT kernels, ffmpeg 6.x libs for torchcodec (video)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      curl git ca-certificates \
      build-essential ninja-build pkg-config \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy

# sglang pinned to the exact commit validated for Qwen3.8-27B (hybrid GDN arch) + DSpark
# torch/triton/flashinfer pinned to the validated stack
ENV SGLANG_BUILD_RUST_EXTS=none
RUN uv venv /opt/venv --python 3.12 && \
    uv pip install --python /opt/venv/bin/python \
      "sglang[all] @ git+https://github.com/sgl-project/sglang.git@2e7c85da6#subdirectory=python" \
      torch==2.13.0 triton==3.7.1 flashinfer-python==0.6.17 \
      "huggingface_hub[cli]" nvidia-cuda-nvcc==13.3.73

ENV PATH="/opt/venv/bin:/opt/cuda/bin:$PATH" VIRTUAL_ENV=/opt/venv

# ---- CUDA_HOME shim from the pip nvcc wheels (kernel JIT needs nvcc) ----
RUN SP=/opt/venv/lib/python3.12/site-packages/nvidia/cu13 && \
    mkdir -p /opt/cuda && \
    for d in bin include lib nvvm; do ln -sfn $SP/$d /opt/cuda/$d; done && \
    ln -sfn $SP/lib /opt/cuda/lib64 && \
    ln -sf $SP/lib/libcudart.so.13 $SP/lib/libcudart.so && \
    /opt/cuda/bin/nvcc --version

# ---- flashinfer: relax over-conservative cccl version guard vs nvcc 13.3 ----
RUN F=/opt/venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h && \
    sed -i 's|^\( *# *error .*incompatible.*\)|// PATCHED \1|' $F && \
    grep -c PATCHED $F

ENV CUDA_HOME=/opt/cuda \
    NVCC_APPEND_FLAGS=-allow-unsupported-compiler \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LD_LIBRARY_PATH=/opt/cuda/lib64:/usr/lib/x86_64-linux-gnu

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---- Hermes-2-Pro XML tool-call parser (fixes structured tool calls) ----
# Qwen3.8-27B's chat template renders tools as XML (<function=name><parameter=k>v</parameter>),
# but sglang's built-in parsers expect JSON inside <tool_call> and silently drop
# XML calls (empty content, tool_calls=[]). Install + register the XML detector.
COPY hermes_xml_detector.py /opt/venv/lib/python3.12/site-packages/sglang/srt/function_call/hermes_xml_detector.py
RUN P=/opt/venv/lib/python3.12/site-packages/sglang/srt/function_call/function_call_parser.py && \
    grep -q hermes_xml_detector "$P" || sed -i \
      -e 's|from sglang.srt.function_call.hermes_detector import HermesDetector|from sglang.srt.function_call.hermes_detector import HermesDetector\nfrom sglang.srt.function_call.hermes_xml_detector import HermesXMLDetector|' \
      -e 's|"hermes": HermesDetector,|"hermes": HermesDetector,\n        "hermes_xml": HermesXMLDetector,|' "$P" && \
    grep -c hermes_xml "$P"

VOLUME /models
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
