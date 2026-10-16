#!/bin/zsh
# Build <module>.fatbin (sm_86 SASS, uncompressed so tgcu can read the cubin) with nvcc 13.4 in the strata-cuda image.
# Usage: build.sh [exl3_m1|exl3_moe|exl3_quant] (default exl3_m1).
# Needs colima running ($HOME is mounted 1:1 in the VM); the same nvcc + flags style as ~/strata-mac/device.ninja.
# ~/cuda-13.4/include/cublas_v2.h is the tgcu shim's minimal cuBLAS header; the vendored util.cuh error-string helper
# names two statuses it lacks, defined here with cuBLAS's values (no cuBLAS call is compiled into these modules).
# torchstub/ stands in for <ATen/Tensor.h> (forward declarations for the vendored exl3_moe_coop.cuh host prototypes).
# --expt-relaxed-constexpr: torch's cpp_extension passes it to every upstream build (COMMON_NVCC_FLAGS); the MoE kernels
# call constexpr host helpers (tile_cols & co.) from device code.
set -e
cd ${0:A:h}
mod=${1:-exl3_m1}
export PATH=~/.local/bin:$PATH
rm -f $mod.fatbin   # a failed nvcc run (e.g. colima stopped) must not leave the old fatbin looking fresh
log=${TMPDIR:-/tmp}/nvcc-$mod.log
# nvcc's exit status decides, not the display filter below
perl -e 'alarm 1800; exec @ARGV' docker run --rm -v $HOME:$HOME -w $PWD strata-cuda:latest \
  $HOME/cuda-13.4/bin/nvcc -std=c++17 -O3 --fatbin -arch=sm_86 -no-compress -Xptxas -v -I torchstub --expt-relaxed-constexpr \
  "-DCUBLAS_STATUS_MAPPING_ERROR=((cublasStatus_t) 11)" "-DCUBLAS_STATUS_LICENSE_ERROR=((cublasStatus_t) 16)" \
  -x cu $mod.cu -o $mod.fatbin > $log 2>&1 || { tail -30 $log; echo "nvcc failed for $mod (full log $log)"; exit 1; }
grep -E "error|warning|Compiling entry|Used [0-9]+ registers" $log | paste - - | grep -E "exl3_recon_k3_cb2|exl3_recon_had_k3_cb2|exl3_had_in|exl3_gemv_kernelILi3ELb0ELi2|exl3_gemm_kernelILi3ELb0ELb0ELi2|exl3_moe_coop|exl3_moe_kernel|exl3s_pf|error|warning" || true
ls -l $mod.fatbin
shasum -a 256 $mod.fatbin
