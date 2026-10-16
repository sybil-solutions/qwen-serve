#!/bin/zsh
# Card-free reproducibility checks, run on the Mac mini from a clone of this repo (the clone must live under $HOME for the docker step).
# Never touches the GPU, the running server, or the live pack: everything is written under $WORK (default /tmp/repro-work).
#   scripts/verify-card-free.sh [patches] [cpu] [metal] [cudart] [fatbins] [pack-layer]     (no argument = all)
# Strict: any failing command, failing build/test (whatever it is piped into) or any hash MISMATCH ends the run with a nonzero exit status and no
# "done:" line.  Full output of every step is kept in $WORK/logs/<step>.log; scripts/test-gates.sh checks that the gates really fail.
# env: STRATA_GIT (default https://github.com/Niko1221/Strata), STRATA_DIR (an existing v0.1.41 checkout to export from instead of cloning),
#      LIBEXL3METAL (prebuilt libexl3metal.dylib for the metal step; built here only if `xcrun -f metal` works), PACK_BIN (read-only experts.bin
#      for the cpu/metal tests, default $HOME/exl3/pack/experts.bin), PACK_LAYER (default 24), PYTHON (default $HOME/strata/.venv/bin/python)
set -euo pipefail
REPO=${0:A:h:h}; WORK=${WORK:-/tmp/repro-work}; PYTHON=${PYTHON:-$HOME/strata/.venv/bin/python}
PACK_BIN=${PACK_BIN:-$HOME/exl3/pack/experts.bin}; PACK_LAYER=${PACK_LAYER:-24}
export PATH=$HOME/.local/bin:$PATH
mkdir -p $WORK/logs
steps=("$@"); [ ${#steps[@]} -gt 0 ] || steps=(patches cpu metal cudart fatbins pack-layer)
say() { echo; echo "=== $*"; }
fail() { echo "FAILED: $*" >&2; exit 1; }
logged() {   # logged <name> <n> <cmd...>: run to $WORK/logs/<name>.log, show its last <n> lines; on failure show them to stderr and exit nonzero
  local name=$1 n=$2; shift 2
  if "$@" > $WORK/logs/$name.log 2>&1; then tail -n $n $WORK/logs/$name.log
  else echo "--- $name failed; last lines of $WORK/logs/$name.log:" >&2; tail -n 30 $WORK/logs/$name.log >&2; exit 1; fi
}
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
expect_sha() {   # expect_sha <file> <sha256> <label>
  local got=$(sha_of $1); [ "$got" = "$2" ] && echo "sha256 $3 $got MATCH" || fail "sha256 $3 MISMATCH: got $got, expected $2"
}

step_patches() {   # every Strata patch applies, in order, to a pristine v0.1.41 export; every other patch applies to its own base
  say "patches: Strata v0.1.41 + patches/strata/000*.patch"
  rm -rf $WORK/strata; mkdir -p $WORK/strata
  if [ -n "${STRATA_DIR:-}" ]; then (cd $STRATA_DIR && git archive v0.1.41) | tar x -C $WORK/strata
  else rm -rf $WORK/strata-clone; git clone -q --depth 1 --branch v0.1.41 ${STRATA_GIT:-https://github.com/Niko1221/Strata} $WORK/strata-clone
       (cd $WORK/strata-clone && git archive HEAD) | tar x -C $WORK/strata
       local pin=$(git -C $WORK/strata-clone rev-parse HEAD)
       [ "$pin" = fb58e0dbc8399662c0e47c76578c6e878b14f6cf ] && echo "pinned commit $pin MATCH" || fail "Strata v0.1.41 resolves to $pin, expected fb58e0dbc8399662c0e47c76578c6e878b14f6cf"; fi
  cd $WORK/strata; git init -q . 2>/dev/null || true
  for p in $REPO/patches/strata/0*.patch; do git apply --check -p1 $p; git apply -p1 $p; echo "applied $(basename $p)"; done
  rm -rf $WORK/vis && mkdir -p $WORK/vis/tools && cp -r $WORK/strata/tools/vision $WORK/vis/tools/ && cd $WORK/vis && git init -q . 2>/dev/null
  git apply --check $REPO/patches/vision/0001-strata-vision-metal-macos.patch; echo "vision patch applies to tools/vision"
}
step_cpu() {       # libexl3cpu from source + its parity tests (decode/had_in bit-exact vs exl3_ref, GEMV gate, 18 real experts from the pack)
  say "cpu: build libexl3cpu, run exl3/test_exl3_cpu.py"
  logged cpu-build 2 zsh $REPO/exl3/cpu/build.sh $WORK/cpu
  expect_sha $WORK/cpu/libexl3cpu.dylib 9fe9cab5491e32e39fcbedfc93043f017f7f1ed60cd65d5211e96a436e42ef3c "libexl3cpu.dylib (the deployed one)"
  cp $WORK/cpu/libexl3cpu.dylib $REPO/exl3/cpu/libexl3cpu.dylib
  EXL3_PACK_BIN=$PACK_BIN EXL3_TEST_NOBENCH=1 logged cpu-test 4 nice -n 10 $PYTHON $REPO/exl3/test_exl3_cpu.py $WORK/cpu_receipt.json
}
step_metal() {     # libexl3metal parity (vs NEON + float64 reference), negatives, determinism, thread safety
  say "metal: libexl3metal test"
  local lib=${LIBEXL3METAL:-}
  if [ -z "$lib" ] && xcrun -f metal >/dev/null 2>&1; then logged metal-build 2 bash $REPO/exl3/metal/build.sh $WORK/metal; lib=$WORK/metal/libexl3metal.dylib; fi
  [ -n "$lib" ] || fail "no libexl3metal: set LIBEXL3METAL=<dylib built with exl3/metal/build.sh on a Mac with the Metal toolchain>"
  logged metal-test 6 nice -n 10 $PYTHON $REPO/exl3/metal/test_exl3_metal.py --lib $lib --cpu-lib $WORK/cpu/libexl3cpu.dylib --pack $PACK_BIN --receipt $WORK/metal_receipt.json
}
step_cudart() {    # the CUDA-runtime shim compiles; exported symbol set printed (deployed one: 136 symbols)
  say "cudart: build shim/libcudart.dylib"
  logged cudart-build 2 zsh $REPO/shim/build-libcudart.sh $WORK/cudart
  local n=$(nm -gU $WORK/cudart/libcudart.dylib | wc -l | tr -d ' ')
  [ "$n" = 136 ] && echo "exported symbols: $n MATCH (deployed: 136)" || fail "libcudart exports $n symbols, the deployed one 136"
}
step_fatbins() {   # sm_86 fatbins of the EXL3 engine kernels (nvcc 13.4 in the strata-cuda container) + the host library around them
  say "fatbins: exl3_strata / _recon / _pf with nvcc 13.4 in the container, then strata_m4/build.sh"
  case $REPO in $HOME/*) ;; *) fail "repo must be under \$HOME (the container mounts \$HOME 1:1)";; esac
  cd $REPO/exl3
  for m in exl3_strata exl3_strata_recon exl3_strata_pf; do
    logged fatbin-$m 2 zsh build.sh $m
    want=$(awk -v f=$m.fatbin '$1 !~ /^#/ && $2 == f {print $1}' $REPO/results/verification/fatbin-sha256.txt)
    if [ $m = exl3_strata ] && [ $PWD != /Users/sero/exl3 ]; then echo "$m.fatbin: its id depends on the build path ($PWD, not /Users/sero/exl3); sha256 not compared"
    else expect_sha $m.fatbin $want $m.fatbin; fi
  done
  (cd strata_m4 && logged strata_m4-build 2 zsh build.sh)
}
step_pack-layer() { # fetch one layer from the pinned HF revision into a TEMP pack, verify, compare its sha256 with results/pack/exl3_pack.json
  say "pack-layer: pack_td.py sync $PACK_LAYER into $WORK/exl3 (needs ~1 GB free + network)"
  rm -rf $WORK/exl3; mkdir -p $WORK/exl3; (cd $REPO && tar cf - exl3) | tar xf - -C $WORK
  cd $WORK/exl3; cp $WORK/cpu/libexl3cpu.dylib cpu/
  logged pack-index 1 $PYTHON pack_td.py index
  logged pack-sync 2 $PYTHON pack_td.py sync $PACK_LAYER
  $PYTHON pack_td.py compare $WORK/exl3/pack/exl3_pack.json $REPO/results/pack/exl3_pack.json $PACK_LAYER   # exits nonzero on MISMATCH
  rm -f $WORK/exl3/pack/experts.bin
}
for s in $steps; do step_$s; done
echo; echo "done: ${steps[*]}"
