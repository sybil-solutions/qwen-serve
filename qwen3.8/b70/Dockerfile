# Qwen3.8-27B on 2x Intel Arc Pro B70 — one-command llama.cpp SYCL TP2 server
#
# Build:  docker compose build
# Run:    docker compose up -d     (downloads the pinned model on first run)
#
# Source: mndodd/llama.cpp @ 4302fb5 + steveseguin/b70-optimization-lab TP2 stack
# (both patches SHA-256-verified against the lab's published digests).
FROM intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
        python3-pip python3-venv ffmpeg \
    && pip3 install --break-system-packages "huggingface_hub[cli]" \
    && rm -rf /var/lib/apt/lists/*

# Patched source: base fork + full TP2 stack + Q4K increment
WORKDIR /build
RUN git clone -q https://github.com/mndodd/llama.cpp llama.cpp \
    && cd llama.cpp \
    && git checkout -q 4302fb59969a5d8cf9f8e5f55fdd4506d0ed2126
COPY patches/ /patches/
RUN cd /build/llama.cpp \
    && git apply --check /patches/tp2-full-stack.patch \
    && git apply /patches/tp2-full-stack.patch \
    && git apply --check /patches/q4k-increment.patch \
    && git apply /patches/q4k-increment.patch \
    && git diff --check

# SYCL JIT build. Note: under oneAPI 2026.1.x, UR device enumeration works once
# /opt/intel/oneapi/umf/1.0/lib (providing libumf.so.1) is on LD_LIBRARY_PATH
# or registered via ldconfig, but the bmg_g31 AOT reordered MMVQ path hangs in
# check_Prefill_Speed on the current dp4a2 + 26.05.37020.4 stack; 2025.3.3 JIT
# remains the verified-working path and reaches decode parity with the lab record.
SHELL ["/bin/bash", "-c"]
RUN cd /build/llama.cpp \
    && . /opt/intel/oneapi/setvars.sh --force \
    && cmake -G "Unix Makefiles" -S . -B build-sycl \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/cc \
        -DCMAKE_CXX_COMPILER="$(command -v icpx)" \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
        -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
        -DGGML_SYCL_F16=ON -DGGML_SYCL_GRAPH=OFF -DGGML_SYCL_DNN=OFF \
        -DGGML_SYCL_HOST_MEM_FALLBACK=OFF \
        -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON \
    && cmake --build build-sycl --target llama-server -j"$(nproc)"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Model pins (SHA-256 verified against b70-optimization-lab repro digest)
ENV MODEL_REPO=ggml-org/Qwen3.8-27B-GGUF \
    MODEL_REV=0669b98607d47046c7c2b3f801011d54a08cfccf \
    MODEL_FILE=Qwen3.8-27B-Q4_K_M.gguf \
    MODEL_SHA256=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34 \
    MTP_FILE=mtp-Qwen3.8-27B-Q4_0.gguf \
    MMPROJ_FILE=mmproj-Qwen3.8-27B-Q8_0.gguf

EXPOSE 8010
ENTRYPOINT ["/entrypoint.sh"]
