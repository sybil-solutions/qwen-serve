#!/usr/bin/env bash
# Launch SGLang server for Qwen3.6-35B-A3B with native MTP speculative
# decoding on 4× RTX 3090 (CUDA, bf16, TP=4).
#
# Native MTP (NEXTN → EAGLE) uses the MTP module built into the Qwen3.6
# checkpoint (mtp_num_hidden_layers=1). The draft model is loaded from
# the same checkpoint via Qwen3_5ForCausalLMMTP.
#
# On CUDA, all required sgl_kernel ops (build_tree_kernel_efficient,
# verify_tree_greedy) are available natively.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8020}"
TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Speculative decoding params (overridable via env)
SPEC_ALGO="${SPEC_ALGO:-NEXTN}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
SPEC_NUM_DRAFT="${SPEC_NUM_DRAFT:-4}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP_SIZE}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --host "${HOST}" --port "${PORT}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --context-length 2048 --max-running-requests 2 \
    --attention-backend flashinfer \
    --schedule-conservativeness 2.0 \
    --max-mamba-cache-size 32 \
    --mamba-scheduler-strategy extra_buffer \
    --speculative-algorithm "${SPEC_ALGO}" \
    --speculative-draft-model-path "${MODEL_PATH}" \
    --speculative-num-steps "${SPEC_NUM_STEPS}" \
    --speculative-eagle-topk "${SPEC_EAGLE_TOPK}" \
    --speculative-num-draft-tokens "${SPEC_NUM_DRAFT}"
