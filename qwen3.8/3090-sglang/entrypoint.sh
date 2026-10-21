#!/bin/bash
# Qwen3.8-27B AWQ-INT4 + DSpark on RTX 3090s (validated presets)
# Env: TP (1|2|4, default 1), PORT (default 8000), DSPARK (1|0, default 1), MODELS_DIR (default /models)
set -euo pipefail
TP=${TP:-1}
PORT=${PORT:-8000}
DSPARK=${DSPARK:-1}
MODELS=${MODELS_DIR:-/models}
AWQ=$MODELS/Qwen3.8-27B-AWQ-INT4
DRAFT=$MODELS/Qwen3.8-27B-DSpark
AWQ_REPO=cyankiwi/Qwen3.8-27B-AWQ-INT4
AWQ_REVISION=63768c10df38c0395e12ef49edac1bd539eaeeea
DRAFT_REPO=RadixArk/Qwen3.8-27B-DSpark
DRAFT_REVISION=85ef153be924f17ce4bf62726954eeaa4a73e854

download_pinned_model() {
  local repository=$1
  local revision=$2
  local model_dir=$3
  local required_file=$4
  local revision_file="$model_dir/.inference-index-revision"
  local expected_revision="$repository@$revision"

  if [[ -f $required_file && -f $revision_file && $(<"$revision_file") == "$expected_revision" ]]; then
    echo "[entrypoint] using pinned cache $expected_revision"
    return
  fi

  if [[ -f $required_file && ( ! -f $revision_file || "${HF_HUB_OFFLINE:-0}" == "1" || "${TRANSFORMERS_OFFLINE:-0}" == "1" ) ]]; then
    echo "[entrypoint] using existing local model files in $model_dir"
    return
  fi

  if [[ "${HF_HUB_OFFLINE:-0}" == "1" || "${TRANSFORMERS_OFFLINE:-0}" == "1" ]]; then
    echo "[entrypoint] offline mode enabled and required model file is missing: $required_file" >&2
    exit 1
  fi

  echo "[entrypoint] downloading pinned model $expected_revision"
  mkdir -p "$model_dir"
  if ! hf download "$repository" --revision "$revision" --local-dir "$model_dir"; then
    if [[ -f $required_file ]]; then
      echo "[entrypoint] warning: hf download failed; falling back to existing local files in $model_dir" >&2
      return
    fi
    echo "[entrypoint] failed to download $expected_revision and $required_file is not present locally" >&2
    exit 1
  fi
  [[ -f $required_file ]] || {
    echo "[entrypoint] required model file is missing after download: $required_file" >&2
    exit 1
  }
  printf '%s\n' "$expected_revision" >"$revision_file.tmp"
  mv "$revision_file.tmp" "$revision_file"
}

# DSpark on TP1 leaves almost no KV cache (21.7 GB weights on 24 GB card) — auto-disable
if [ "$TP" = "1" ] && [ "$DSPARK" = "1" ]; then
  echo "[entrypoint] DSpark auto-disabled on TP=1 (insufficient VRAM for draft model; would leave <1 GB KV cache)"
  DSPARK=0
fi

echo "[entrypoint] TP=$TP DSPARK=$DSPARK PORT=$PORT models=$MODELS"

# --- download exact weight revisions if the pinned cache is absent ---
download_pinned_model "$AWQ_REPO" "$AWQ_REVISION" "$AWQ" "$AWQ/model.safetensors.index.json"
if [ "$DSPARK" = "1" ]; then
  download_pinned_model "$DRAFT_REPO" "$DRAFT_REVISION" "$DRAFT" "$DRAFT/model.safetensors"
fi

# Prevent SGLang / Transformers / huggingface_hub from making unguarded Hub API calls once local files are verified
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- validated presets per GPU count ---
# GDN hybrid memory tuning: TP1 needs a small mamba state pool to fit one 24 GB card
case "$TP:$DSPARK" in
  1:1) TPFLAGS="--mem-fraction-static 0.92 --max-mamba-cache-size 4 --disable-prefill-cuda-graph --cuda-graph-max-bs 4" ;;
  1:0) TPFLAGS="--mem-fraction-static 0.92 --max-mamba-cache-size 8 --disable-prefill-cuda-graph --cuda-graph-max-bs 8" ;;
  2:1) TPFLAGS="--mem-fraction-static 0.85 --max-mamba-cache-size 16 --disable-prefill-cuda-graph --cuda-graph-max-bs 16" ;;
  2:0) TPFLAGS="--mem-fraction-static 0.85 --max-mamba-cache-size 32 --disable-prefill-cuda-graph --cuda-graph-max-bs 16" ;;
  4:1) TPFLAGS="--mem-fraction-static 0.72" ;;
  4:0) TPFLAGS="--mem-fraction-static 0.80" ;;
  *) echo "TP must be 1, 2 or 4"; exit 1 ;;
esac

SPECFLAGS=""
if [ "$DSPARK" = "1" ]; then
  SPECFLAGS="--speculative-algorithm DSPARK --speculative-draft-model-path $DRAFT --speculative-dspark-block-size 7 --speculative-draft-model-quantization unquant"
fi

echo "[entrypoint] launching sglang…"
exec python3 -m sglang.launch_server \
  --model-path "$AWQ" \
  --tp "$TP" \
  --chunked-prefill-size 4096 \
  --mamba-radix-cache-strategy extra_buffer \
  $TPFLAGS $SPECFLAGS \
  --host 0.0.0.0 --port "$PORT"
