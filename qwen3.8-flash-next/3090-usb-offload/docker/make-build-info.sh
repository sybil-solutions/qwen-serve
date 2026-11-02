#!/bin/sh
# BUILD_INFO.txt + SHA256SUMS for the bundle in $1
set -eu
B=$1
{
  echo "qwen38-flash-next-3090-usb-offload build bundle"
  echo "platform: $(uname -m) $(. /etc/os-release; echo $PRETTY_NAME)"
  echo "build root in the container: $BUILD_ROOT"
  echo "Strata: v0.1.41 fb58e0dbc8399662c0e47c76578c6e878b14f6cf + patches/strata/0*.patch + patches/vision/0001"
  echo "llama.cpp (Strata FetchContent pin): $(grep -o '[0-9a-f]\{40\}' $B/engine/strata/CMakeLists.txt | head -1)"
  echo "nvcc: 13.4.92 (redist manifest redistrib_13.4.2.json)"
  echo "fatbins/SHA256SUMS: sha256 of the products that were compared with the deployed ones"
  echo "engine/strata-mac-fatbins.sha256: $(wc -l < $B/engine/strata-mac-fatbins.sha256) engine fatbins"
} > $B/BUILD_INFO.txt
(cd $B && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
