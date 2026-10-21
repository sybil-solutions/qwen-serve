#!/bin/bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT

mkdir -p "$TEST_DIR/bin" "$TEST_DIR/models"

printf '%s\n' \
  '#!/bin/bash' \
  'set -euo pipefail' \
  'printf "%s\n" "$*" >>"$TEST_HF_LOG"' \
  'repository=$2' \
  'model_dir=${!#}' \
  'mkdir -p "$model_dir"' \
  'if [[ $repository == cyankiwi/* ]]; then' \
  '  touch "$model_dir/model.safetensors.index.json"' \
  'else' \
  '  touch "$model_dir/model.safetensors"' \
  'fi' >"$TEST_DIR/bin/hf"

printf '%s\n' \
  '#!/bin/bash' \
  'set -euo pipefail' \
  'printf "%s\n" "$*" >"$TEST_PYTHON_LOG"' >"$TEST_DIR/bin/python3"

chmod +x "$TEST_DIR/bin/hf" "$TEST_DIR/bin/python3"
export PATH="$TEST_DIR/bin:$PATH"
export TEST_HF_LOG="$TEST_DIR/hf.log"
export TEST_PYTHON_LOG="$TEST_DIR/python.log"

bash -n "$REPO_DIR/entrypoint.sh"
MODELS_DIR="$TEST_DIR/models" TP=4 DSPARK=1 PORT=8123 "$REPO_DIR/entrypoint.sh"

grep -Fq 'cyankiwi/Qwen3.8-27B-AWQ-INT4 --revision 63768c10df38c0395e12ef49edac1bd539eaeeea' "$TEST_HF_LOG"
grep -Fq 'RadixArk/Qwen3.8-27B-DSpark --revision 85ef153be924f17ce4bf62726954eeaa4a73e854' "$TEST_HF_LOG"
grep -Fqx 'cyankiwi/Qwen3.8-27B-AWQ-INT4@63768c10df38c0395e12ef49edac1bd539eaeeea' "$TEST_DIR/models/Qwen3.8-27B-AWQ-INT4/.inference-index-revision"
grep -Fqx 'RadixArk/Qwen3.8-27B-DSpark@85ef153be924f17ce4bf62726954eeaa4a73e854' "$TEST_DIR/models/Qwen3.8-27B-DSpark/.inference-index-revision"
grep -Fq -- '--tp 4' "$TEST_PYTHON_LOG"
grep -Fq -- '--speculative-draft-model-path' "$TEST_PYTHON_LOG"
grep -Fq -- '--host 0.0.0.0 --port 8123' "$TEST_PYTHON_LOG"

: >"$TEST_HF_LOG"
MODELS_DIR="$TEST_DIR/models" TP=4 DSPARK=1 PORT=8123 "$REPO_DIR/entrypoint.sh"
[[ ! -s $TEST_HF_LOG ]]


# Pre-existing local files on filesystem without .inference-index-revision should not call hf
rm -f "$TEST_DIR/models/Qwen3.8-27B-AWQ-INT4/.inference-index-revision"
rm -f "$TEST_DIR/models/Qwen3.8-27B-DSpark/.inference-index-revision"
: >"$TEST_HF_LOG"
MODELS_DIR="$TEST_DIR/models" TP=4 DSPARK=1 PORT=8123 "$REPO_DIR/entrypoint.sh"
[[ ! -s $TEST_HF_LOG ]]

# Stale revision marker in offline environment should fall back to local files when hf fails
printf '%s\n' 'stale-revision' >"$TEST_DIR/models/Qwen3.8-27B-AWQ-INT4/.inference-index-revision"
printf '%s\n' '#!/bin/bash' 'exit 1' >"$TEST_DIR/bin/hf"
chmod +x "$TEST_DIR/bin/hf"
MODELS_DIR="$TEST_DIR/models" TP=4 DSPARK=1 PORT=8123 "$REPO_DIR/entrypoint.sh"

echo 'entrypoint tests passed'
