#!/bin/bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"

export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"

export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_MAMBA_SSM_DTYPE=float32
export SGLANG_XPU_GDN_FAST_PATH="${SGLANG_XPU_GDN_FAST_PATH:-1}"

export SGLANG_XPU_ESIMD_DECODE="${SGLANG_XPU_ESIMD_DECODE:-1}"
export SGLANG_XPU_ESIMD_MOE="${SGLANG_XPU_ESIMD_MOE:-1}"
export SGLANG_XPU_ESIMD_MOE_PREFILL="${SGLANG_XPU_ESIMD_MOE_PREFILL:-1}"
export SGLANG_XPU_FA_ESIMD_QKV="${SGLANG_XPU_FA_ESIMD_QKV:-1}"
export SGLANG_XPU_GDN_ESIMD="${SGLANG_XPU_GDN_ESIMD:-0}"
export SGLANG_XPU_GDN_EXTEND_ESIMD="${SGLANG_XPU_GDN_EXTEND_ESIMD:-0}"
export SGLANG_XPU_PREFILL_DPAS="${SGLANG_XPU_PREFILL_DPAS:-1}"
export SGLANG_XPU_ENABLE_GRAPH="${SGLANG_XPU_ENABLE_GRAPH:-1}"

export SGL_XPU_ESIMD_DECODE="$SGLANG_XPU_ESIMD_DECODE"
export SGL_XPU_ESIMD_MOE="$SGLANG_XPU_ESIMD_MOE"
export SGL_XPU_ESIMD_MOE_PREFILL="$SGLANG_XPU_ESIMD_MOE_PREFILL"
export SGL_XPU_FA_ESIMD_QKV="$SGLANG_XPU_FA_ESIMD_QKV"
export SGL_XPU_GDN_ESIMD="$SGLANG_XPU_GDN_ESIMD"
export SGL_XPU_GDN_EXTEND_ESIMD="$SGLANG_XPU_GDN_EXTEND_ESIMD"
export SGL_XPU_PREFILL_DPAS="$SGLANG_XPU_PREFILL_DPAS"
export SGL_XPU_ENABLE_GRAPH="$SGLANG_XPU_ENABLE_GRAPH"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP_SIZE}" \
    --dtype float16 \
    --quantization fp8 \
    --load-format "${LOAD_FORMAT:-layered_fp8}" \
    --attention-backend intel_xpu \
    --trust-remote-code \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-mamba-cache-size 64 \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer \
    --reasoning-parser qwen3 \
    --enable-cache-report \
    --enable-metrics \
    --host "${HOST}" \
    --port "${PORT}"
