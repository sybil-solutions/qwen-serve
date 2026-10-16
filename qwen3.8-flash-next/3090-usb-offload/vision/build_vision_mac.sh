#!/bin/bash
# Metal strata-vision (llama.cpp mtmd + the mmproj) for macOS arm64. Builds in ~20 s on an M-series Mac with Xcode's Metal toolchain.
#   needs: a Strata v0.1.41 checkout with patches/vision/0001-strata-vision-metal-macos.patch applied (tools/vision), cmake + ninja, and llama.cpp at the
#   commit Strata pins (3cf03257f219afbe7334045ff7c6a06ac68c627d; the Linux configure fetches it to <build>/_deps/strata_llamacpp-src).
# env: SRC (default $HOME/strata/tools/vision), LLAMA (llama.cpp checkout), OUT (build dir), DEST (installed binary)
set -e
SRC=${SRC:-$HOME/strata/tools/vision}
LLAMA=${LLAMA:-$HOME/strata-linux/_deps/strata_llamacpp-src}
OUT=${OUT:-$HOME/strata-mac-vision/build-vision}
DEST=${DEST:-$HOME/strata-mac-vision/strata-vision-metal}
PATH=$HOME/strata/.venv/bin:$PATH
cmake -S "$SRC" -B "$OUT" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$LLAMA" -DSTRATA_PORTABLE=OFF -DSTRATA_VISION_METAL=ON
cmake --build "$OUT" --target strata-vision -j 4
mkdir -p "$(dirname "$DEST")"
cp "$OUT/bin/strata-vision" "$DEST" && codesign -s - -f "$DEST"
echo built "$DEST"; shasum -a 256 "$DEST"
