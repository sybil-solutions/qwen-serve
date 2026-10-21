#!/bin/bash
set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec vllm serve /model \
  --dtype float16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --tensor-parallel-size 1 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --port 8000 \
  --host 0.0.0.0 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":5,"model":"/model"}'
