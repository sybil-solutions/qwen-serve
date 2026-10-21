#!/usr/bin/env bash
# Launch SGLang server for Qwen3.6-35B-A3B with native MTP speculative
# decoding on Intel BMG, TP=2 FP8.
#
# This enables NEXTN (native Multi-Token Prediction) — the MTP module built
# into the Qwen3.6 checkpoint (mtp_num_hidden_layers=1, 19 weight keys).
# NEXTN resolves to EAGLE internally; the draft model is the MTP module loaded
# from the same checkpoint via Qwen3_5ForCausalLMMTP.
#
# Auto-tuned speculative params for this arch: num_steps=3, eagle_topk=1,
# num_draft_tokens=4. Override via env vars if needed.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"

# --- device selection ---
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"

# --- GDN SSM state dtype: fp32 to avoid !!! corruption ---
export SGLANG_MAMBA_CONV_DTYPE=float16
# SGLANG_MAMBA_SSM_DTYPE intentionally NOT set — let model's float32 stand.

# --- ESIMD fast-path gates ---
export SGLANG_XPU_ESIMD_DECODE="${SGLANG_XPU_ESIMD_DECODE:-1}"
export SGLANG_XPU_ESIMD_MOE="${SGLANG_XPU_ESIMD_MOE:-1}"
export SGLANG_XPU_ESIMD_MOE_PREFILL="${SGLANG_XPU_ESIMD_MOE_PREFILL:-1}"
export SGLANG_XPU_FA_ESIMD_QKV="${SGLANG_XPU_FA_ESIMD_QKV:-1}"
# GDN ESIMD DISABLED: fp16 SSM state drift → NaN → !!! output
export SGLANG_XPU_GDN_ESIMD="${SGLANG_XPU_GDN_ESIMD:-0}"
export SGLANG_XPU_GDN_EXTEND_ESIMD="${SGLANG_XPU_GDN_EXTEND_ESIMD:-0}"
export SGLANG_XPU_PREFILL_DPAS="${SGLANG_XPU_PREFILL_DPAS:-1}"

# --- XPU Graph ---
export SGLANG_XPU_ENABLE_GRAPH="${SGLANG_XPU_ENABLE_GRAPH:-0}"

# --- Speculative decoding params (overridable via env) ---
SPEC_ALGO="${SPEC_ALGO:-NEXTN}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
SPEC_NUM_DRAFT="${SPEC_NUM_DRAFT:-4}"

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
    --speculative-algorithm "${SPEC_ALGO}" \
    --speculative-draft-model-path "${MODEL_PATH}" \
    --speculative-num-steps "${SPEC_NUM_STEPS}" \
    --speculative-eagle-topk "${SPEC_EAGLE_TOPK}" \
    --speculative-num-draft-tokens "${SPEC_NUM_DRAFT}" \
    --speculative-draft-attention-backend triton \
    --enable-cache-report \
    --enable-metrics \
    --host "${HOST}" \
    --port "${PORT}"
