#!/bin/sh
# qwen38-flash-next-b70-offload image entrypoint.
#   pack-store / verify-store      build or check the packed expert store in /nvx (CPU only, needs /models)
#   anything else                  the SGLang server argv (exec'd as PID 1), or one shell string (/bin/sh -c)
# QWEN_B70_MODE picks the tier settings (each can still be overridden by its own env var):
#   nvme16  RAM tier 2 GB, 2-buffer prefill ring, admission-controlled decode tier, 8 decode-fill queues (16 GiB cap)
#   nvme32  RAM tier 7 GB, 5-buffer ring, srv23 tier code path (32 GiB cap; fewer masked decode picks)
set -e
case "$1" in
  pack-store) shift; exec python3 /opt/qwen-b70/serve/pack_store.py pack "$@" ;;
  verify-store) shift; exec python3 /opt/qwen-b70/serve/pack_store.py verify "$@" ;;
esac
case "${QWEN_B70_MODE:-nvme16}" in
  nvme16) : "${EXL3_NVTIER_RAM_GB:=2}" "${EXL3_NVTIER_STAGE_HOSTBUF:=2}" "${EXL3_NVTIER_ADMIT:=1}" "${EXL3_NVTIER_DECODE_FILL_QUEUES:=8}" ;;
  nvme32) : "${EXL3_NVTIER_RAM_GB:=7}" "${EXL3_NVTIER_STAGE_HOSTBUF:=5}" "${EXL3_NVTIER_ADMIT:=0}" "${EXL3_NVTIER_DECODE_FILL_QUEUES:=1}" ;;
  *) echo "QWEN_B70_MODE must be nvme16 or nvme32" >&2; exit 2 ;;
esac
# experimental, off by default: EXL3_NVTIER_VRAM_WARM=7500 warms the VRAM expert cache from the prior at start
: "${EXL3_NVTIER_VRAM_WARM:=0}" "${HF_HUB_OFFLINE:=1}" "${TRANSFORMERS_OFFLINE:=1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE EXL3_NVTIER_RAM_GB EXL3_NVTIER_STAGE_HOSTBUF EXL3_NVTIER_ADMIT EXL3_NVTIER_DECODE_FILL_QUEUES EXL3_NVTIER_VRAM_WARM
if [ "${EXL3_NVTIER:-1}" = 1 ] && [ "$#" -gt 0 ]; then
  python3 /opt/qwen-b70/serve/pack_store.py probe
fi
if [ "$#" -eq 0 ]; then exec python3 -m sglang.launch_server --help; fi
if [ "$#" -eq 1 ]; then exec /bin/sh -c "$1"; fi
exec "$@"
