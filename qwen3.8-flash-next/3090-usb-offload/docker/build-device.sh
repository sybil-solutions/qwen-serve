#!/bin/bash
# Inside the build container: the device code that has no engine build of its own.
#   - the three EXL3 engine modules (exl3_strata, exl3_strata_recon, exl3_strata_pf) -> sm_86 fatbins, nvcc 13.4.92, the flags of exl3/build.sh
#   - the shim's two prebuilt blobs (tgcu_builtin.fatbin, tgcu_libc.cubin), same flags/tool as the shipped ones
# Every product's sha256 is compared with the deployed one (results/verification/fatbin-sha256.txt, shim/prebuilt/SHA256SUMS); a mismatch fails the build.
# The path /Users/sero/exl3 (= $BUILD_ROOT/exl3) is part of the contract: exl3_strata.fatbin embeds a path-dependent id.
set -euo pipefail
R=${BUILD_ROOT:?}; OUT=${1:?output dir}; NVCC=$R/cuda-13.4/bin/nvcc
mkdir -p $OUT; : > $OUT/SHA256SUMS
fail() { echo "FAILED: $*" >&2; exit 1; }
nv() {   # nv <log> <cmd...>: run, keep the full log, show it only on failure
  local log=$1; shift
  "$@" > $log 2>&1 || { tail -40 $log >&2; fail "$* (full log: $log)"; }
}
expect() {   # expect <file> <expected sha256> <label>
  local got; got=$(sha256sum "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] || fail "sha256 of $3: built $got, deployed $2"
  echo "MATCH deployed: $3 $got"; echo "$got  $3" >> $OUT/SHA256SUMS
}
cd $R/exl3
for m in exl3_strata exl3_strata_recon exl3_strata_pf; do
  nv $OUT/$m.nvcc.log $NVCC -std=c++17 -O3 --fatbin -arch=sm_86 -no-compress -Xptxas -v -I torchstub --expt-relaxed-constexpr \
    "-DCUBLAS_STATUS_MAPPING_ERROR=((cublasStatus_t) 11)" "-DCUBLAS_STATUS_LICENSE_ERROR=((cublasStatus_t) 16)" \
    -x cu $m.cu -o $OUT/$m.fatbin
  expect $OUT/$m.fatbin "$(awk -v f=$m.fatbin '$1 !~ /^#/ && $2 == f {print $1}' $R/verify/fatbin-sha256.txt)" $m.fatbin
done
cd $R/shim
nv $OUT/tgcu_builtin.nvcc.log $NVCC -std=c++17 -O3 -Xptxas -v --fatbin -arch=sm_86 -no-compress -x cu tgcu_builtin.cu -o $OUT/tgcu_builtin.fatbin
expect $OUT/tgcu_builtin.fatbin "$(awk '/tgcu_builtin.fatbin/ {print $1}' $R/verify/prebuilt-SHA256SUMS)" tgcu_builtin.fatbin
nv $OUT/tgcu_libc.ptxas.log $R/cuda-13.4/bin/ptxas -arch=sm_86 -c tgcu_libc.ptx -o $OUT/tgcu_libc.cubin
expect $OUT/tgcu_libc.cubin "$(awk '/tgcu_libc.cubin/ {print $1}' $R/verify/prebuilt-SHA256SUMS)" tgcu_libc.cubin
