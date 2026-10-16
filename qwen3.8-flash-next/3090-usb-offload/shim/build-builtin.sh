#!/bin/zsh
# tgcu_builtin.fatbin: the kernels the shim itself needs (memset, device copies, WMMA GEMM for the cuBLAS subset). nvcc 13.4 in the strata-cuda container.
# Usage: build-builtin.sh [OUT_DIR]  (must be under $HOME: the container mounts $HOME 1:1).
# Rebuilt this way the fatbin has the same size as shim/prebuilt/tgcu_builtin.fatbin but different bytes (nvcc embeds a path-dependent id [INFERENCE]; the same
# effect was measured on exl3_strata.fatbin built in three directories), so prebuilt/ keeps the exact files the engine ran with.
# tgcu_libc.cubin (device printf / assert stubs) is only shipped prebuilt: the ptxas flags that made the 3,200-byte deployed file were not recorded
# (plain `ptxas -arch=sm_86 tgcu_libc.ptx` gives a 1,464-byte file with the same functions and no .debug_frame).
set -e
cd ${0:A:h}
OUT=${1:-$PWD}; mkdir -p $OUT
export PATH=$HOME/.local/bin:$PATH
perl -e 'alarm 900; exec @ARGV' docker run --rm -v $HOME:$HOME -w $PWD strata-cuda:latest \
  $HOME/cuda-13.4/bin/nvcc --fatbin -arch=sm_86 -no-compress -x cu tgcu_builtin.cu -o $OUT/tgcu_builtin.fatbin
shasum -a 256 $OUT/tgcu_builtin.fatbin
