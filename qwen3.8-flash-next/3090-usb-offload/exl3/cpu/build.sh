#!/bin/zsh
# Build libexl3cpu.dylib (the NEON EXL3 tier: decode, had_in, GEMV; ctypes C API). Runs on any Apple-silicon Mac.
# Usage: build.sh [OUT_DIR]   (default: this directory).  SDK=<path> overrides the pinned SDK (the 27.0 SDK breaks the link on the mini).
# The output is built as a *relative* name inside OUT_DIR: clang takes the dylib's install name from the -o argument, so an absolute -o path
# changes the bytes. Built this way with SDK 26.5 or 26.0 the result is byte-identical to the dylib the engine ran (sha256 9fe9cab5...).
set -e
SRC=${0:A:h}/exl3_cpu.c
OUT=${1:-${0:A:h}}; mkdir -p $OUT; cd $OUT
SDK=${SDK:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}
clang -O3 -mcpu=apple-m4 -ffp-contract=off -isysroot $SDK -shared -o libexl3cpu.dylib $SRC
echo built $PWD/libexl3cpu.dylib
shasum -a 256 libexl3cpu.dylib
