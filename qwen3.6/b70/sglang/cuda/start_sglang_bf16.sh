#!/bin/bash
# sglang CUDA bf16 server for Qwen3.6-35B-A3B on 4x RTX 3090 (TP=4)
#
# Tested working: ~14 tok/s decode, fully coherent output
# VRAM: ~17.5 GB per GPU (35B bf16 / 4 = 17.5GB weights + KV cache)
#
# Usage:
#   docker run -d --name sglang-cuda-bf16 --gpus all --shm-size=4g \
#     --entrypoint "" \
#     -v /home/sero/models:/models \
#     -p 8020:8020 \
#     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     ghcr.io/0xsero/qwen38-3090-sglang:latest \
#     bash /sglang/cuda/start_sglang_bf16.sh

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
PORT="${PORT:-8020}"
TP="${TP:-4}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp "$TP" \
  --dtype bfloat16 \
  --trust-remote-code \
  --host 0.0.0.0 --port "$PORT" \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CONTEXT_LENGTH" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --attention-backend flashinfer \
  --schedule-conservativeness 2.0 \
  --max-mamba-cache-size 32 \
  --mamba-scheduler-strategy extra_buffer
