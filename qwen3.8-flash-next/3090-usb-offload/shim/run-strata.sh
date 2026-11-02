#!/bin/zsh
# Run the macOS Strata engine on the RTX 3090 behind the chestnut USB dock (tgcu shim + tinygrad).
source ~/.chestnut_env
SITE=$(~/tinygrad/.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
export DEV=NV:PTX PYTHONPATH=$HOME/shim:$HOME/tg-strata:$SITE PYTHONNOUSERSITE=1
# EXL3 configs: expert-pool workers must NOT spin (the 20 ms default starves the USB host thread: h2d stall 13-15 ms/layer).
[[ "$*" == *exl3* ]] && export STRATA_POOL_SPIN_US=${STRATA_POOL_SPIN_US:-0}
# one round trip per doorbell layer (host reduces the CPU partial; byte-identical text vs 0, 15-25% faster under spin 0: S1/S2 2026-10-11)
[[ "$*" == *exl3* ]] && export STRATA_EXL3_SLIM=${STRATA_EXL3_SLIM:-1}
# EXL3 transport + cold-expert I/O (HideHandshakeAndPool 2026-10-11): pre-armed gated CE pull of x/ids/w + one gated rows+flag push with the next pull queued behind it (byte-identical text, -1.4..-2 ms per ring), and parallel whole-blob preads of the cold experts before the NEON kernels (mmap page faults were 16 KiB random reads)
[[ "$*" == *exl3* ]] && export STRATA_HS_PRE=${STRATA_HS_PRE:-1} STRATA_HS_DEFER=${STRATA_HS_DEFER:-1} STRATA_MAC_PULL=${STRATA_MAC_PULL:-1}
exec ~/strata-mac/strata "$@"
