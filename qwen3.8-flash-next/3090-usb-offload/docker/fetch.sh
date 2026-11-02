#!/bin/bash
# Model fetch + verify, run by `docker compose --profile fetch run --rm fetch <what>` (~50 GB + ~84 GB into ./out/models; nothing is downloaded without an argument).
#   experts   the 45.8 GB EXL3 experts: 48 layers range-fetched from turboderp/Qwen3.8-Flash-Next-exl3 @ 69e33439 (pack_td.py), each layer's sha256 compared with
#             results/pack/exl3_pack.json (a mismatch aborts), then `verify` (6 experts per layer re-fetched and byte-compared) and the Strata marker file.
#             EXPERT_LAYERS="24 25" limits the run to those layers (no marker then).
#   dense     IQ3_S GGUF (2 shards, 83.6 GB) + the BF16 mmproj, sha256 compared with the Hugging Face LFS ids in docker/models.sha256
#             (DENSE_MATCH=mmproj limits it to the paths matching that regex)
#   all       both
# The NEON forward check of `pack_td.py sync` needs libexl3cpu.dylib (macOS); here the sha256 manifest comparison replaces it (the manifest was produced
# with the NEON check on the mini), and ./run.sh re-runs the NEON check on a sample when it installs the pack.
set -euo pipefail
M=${MODELS:-/models}; REF=/opt/exl3_pack.reference.json; PY=/opt/venv/bin/python
fail() { echo "FAILED: $*" >&2; exit 1; }
experts() {
  mkdir -p $M/exl3/pack $M/exl3/td; ln -sfn $M/exl3/pack /opt/exl3/pack; ln -sfn $M/exl3/td /opt/exl3/td
  cd /opt/exl3
  $PY pack_td.py index
  $PY pack_td.py sync --no-neon --expect=$REF ${EXPERT_LAYERS:-}
  $PY pack_td.py verify ${EXPERT_LAYERS:-}
  $PY pack_td.py compare $M/exl3/pack/exl3_pack.json $REF ${EXPERT_LAYERS:-}
  if [ -z "${EXPERT_LAYERS:-}" ]; then $PY pack_exl3.py marker; else echo "partial run (layers $EXPERT_LAYERS): no marker written"; fi
}
dense() {
  mkdir -p $M/qwen38fn; cd $M/qwen38fn
  local rev; rev=$(awk '$1 == "revision" {print $2}' /opt/models.sha256)
  grep -v '^#' /opt/models.sha256 | grep -v '^revision' | grep -E "${DENSE_MATCH:-.}" | while read -r sha size path; do
    local f; f=$(basename $path)
    if [ ! -f $f.ok ] || [ "$(stat -c %s $f 2>/dev/null || echo 0)" != "$size" ]; then
      rm -f $f.ok
      curl -fL --retry 8 --retry-delay 5 -C - -o $f "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/$rev/$path"
      [ "$(stat -c %s $f)" = "$size" ] || fail "$f: size $(stat -c %s $f), expected $size"
      echo "$sha  $f" | sha256sum -c - || { mv $f $f.bad; fail "$f: sha256 mismatch (kept as $f.bad)"; }
      touch $f.ok
    else echo "$f: already verified"; fi
  done
}
case ${1:-} in
  experts) experts ;;
  dense) dense ;;
  all) experts; dense ;;
  *) sed -n '2,12p' /opt/fetch.sh; exit 2 ;;
esac
echo "fetch ${1:-}: done"
