#!/bin/zsh
# libcudart.dylib: the CUDA runtime/driver/cuBLAS-subset replacement that Strata links on macOS. Every call goes into tgcu_api.py
# (embedded CPython) -> tgcu.py -> tinygrad NV over USB.  Plain C++ build, no CUDA toolkit libraries; only CUDA 12.8 *headers*.
# Usage: build-libcudart.sh [OUT_DIR]   (default: this directory; the engine finds it through -Wl,-rpath,$HOME/shim)
# Env: PY (python prefix; default the uv-managed CPython 3.12.15 that tinygrad's venv uses), CUDA_HOST (default ~/cuda-12.8), SDK.
set -e
cd ${0:A:h}
OUT=${1:-$PWD}; mkdir -p $OUT
PY=${PY:-$HOME/.local/share/uv/python/cpython-3.12.15-macos-aarch64-none}
CUDA_HOST=${CUDA_HOST:-$HOME/cuda-12.8}
SDK=${SDK:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}
clang++ -std=c++17 -O2 -shared -fPIC -arch arm64 -isysroot $SDK -Wno-everything \
  -I$CUDA_HOST/include -I$PWD/include -I$PY/include/python3.12 \
  tgcudart.cpp tgcublas.cpp -o $OUT/libcudart.dylib -install_name @rpath/libcudart.dylib \
  -L$PY/lib -lpython3.12 -Wl,-rpath,$PY/lib
echo built $OUT/libcudart.dylib
shasum -a 256 $OUT/libcudart.dylib
