#!/bin/bash
# Inside the build container: Strata v0.1.41 + patches, the Linux CMake configure (fetches llama.cpp at Strata's pin), the compile-command
# list, and the device half of the engine (every .cu -> sm_86 fatbin) -- the same steps as docs/reproduce.md section 5, paths as on the mini.
#   build-engine.sh source | configure | device
set -euo pipefail
R=${BUILD_ROOT:?}; PIN=fb58e0dbc8399662c0e47c76578c6e878b14f6cf; JOBS=${JOBS:-4}
fail() { echo "FAILED: $*" >&2; exit 1; }
case ${1:?phase} in
source)    # pristine tag, commit checked, then every patch in patches/strata (in order) and the vision patch
  git clone -q --depth 1 --branch v0.1.41 ${STRATA_GIT:-https://github.com/Niko1221/Strata} $R/strata
  got=$(git -C $R/strata rev-parse HEAD); [ "$got" = $PIN ] || fail "Strata v0.1.41 is $got, expected $PIN"
  cd $R/strata
  for p in /patches/strata/0*.patch; do git apply --check -p1 $p; git apply -p1 $p; echo "applied $(basename $p)"; done
  git apply --check -p1 /patches/vision/0001-strata-vision-metal-macos.patch; git apply -p1 /patches/vision/0001-strata-vision-metal-macos.patch; echo "applied 0001-strata-vision-metal-macos.patch"
  rm -rf .git ;;
configure) # same options as the deployed ~/strata-linux/CMakeCache.txt (read 2026-10-12)
  cmake -S $R/strata -B $R/strata-linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=$R/cuda-13.4/bin/nvcc -DCUDAToolkit_ROOT=$R/cuda-13.4 > $R/configure.log 2>&1 || { tail -40 $R/configure.log >&2; fail "cmake configure"; }
  tail -3 $R/configure.log
  (cd $R/strata-linux && ninja -t commands strata > strata_cmds.txt)
  echo "compile commands: $(wc -l < $R/strata-linux/strata_cmds.txt) lines"
  llama=$(git -C $R/strata-linux/_deps/strata_llamacpp-src rev-parse HEAD 2>/dev/null || true); echo "llama.cpp: ${llama:-?}"
  python3 $R/shim/macbuild.py ;;
device)    # 63 nvcc rules (sm_86, uncompressed fatbins)
  ninja -C $R/strata-mac -f device.ninja -j $JOBS > $R/device.log 2>&1 || { tail -60 $R/device.log >&2; fail "device.ninja"; }
  tail -2 $R/device.log
  echo "fatbins: $(find $R/strata-mac -name '*.fatbin' | wc -l)" ;;
*) fail "unknown phase $1" ;;
esac
