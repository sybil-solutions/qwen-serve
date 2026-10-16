#!/bin/bash
# run.sh - the native half of the one-click flow. Runs on the Mac mini (macOS, Apple silicon); consumes the bundle that
#   docker compose run --rm build        (or: make)          -> ./out
# produced, installs it into the layout the engine expects under $HOME, builds the macOS-only pieces, renders the config and starts the server.
#
#   ./run.sh                start (idempotent: every step is skipped when its result is already there)
#   ./run.sh --check        dry run: everything except starting the engine and touching USB (installs + builds + renders, then validates)
#   ./run.sh --build-metal  only build the two Metal binaries (libexl3metal.dylib, strata-vision-metal) into ./out/metal; run this on a Mac that has the
#                           Metal toolchain when the mini does not (Command Line Tools alone have no `metal` compiler), then run ./run.sh on the mini
#   ./run.sh stop           stop the server this script started (pid file $HOME/shim/server.pid)
# options: --force (replace differing installed files / rebuild), --restart (stop a running server first), --verify-pack (sha256 of the whole 45.8 GB
#          pack against the manifest; reads all of it)
# env:     RUN_ROOT (default $HOME: where shim/ exl3/ strata/ ... go; the dry run on a machine that is serving should use a scratch dir),
#          BUNDLE (default ./out), HOST (127.0.0.1), PORT (8080), KEY_FILE ($HOME/.config/strata/flashnext.key), JOBS (host compile jobs, default 4),
#          CLANG_BIN (directory with clang/clang++ for the CUDA host compiles of the engine; needed when `clang` is older than Apple clang 21, see below)
# Never overwrites an installed file that differs from the bundle's (reported, kept) unless --force; never prints the API key.
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
CHECK=0; FORCE=0; RESTART=0; VERIFY_PACK=0; MODE=up
for a in "$@"; do
  case $a in
    --check) CHECK=1 ;; --force) FORCE=1 ;; --restart) RESTART=1 ;; --verify-pack) VERIFY_PACK=1 ;;
    --build-metal) MODE=build-metal ;; stop) MODE=stop ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a (see ./run.sh --help)" >&2; exit 2 ;;
  esac
done

REAL_HOME=$HOME
# --build-metal installs nothing of its own: it works in a scratch root unless RUN_ROOT is given
if [ "$MODE" = build-metal ] && [ -z "${RUN_ROOT:-}" ]; then RUN_ROOT=$REAL_HOME/.cache/qwen38fn-run/metal-root; fi
ROOT=${RUN_ROOT:-$HOME}; mkdir -p "$ROOT"; ROOT=$(cd "$ROOT" && pwd)
export UV_CACHE_DIR=${UV_CACHE_DIR:-$REAL_HOME/.cache/uv} UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$REAL_HOME/.local/share/uv/python}
export HOME=$ROOT                      # every path below (and every child: engine, scripts) sees the install root as $HOME
BUNDLE=${BUNDLE:-$REPO/out}
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8080}; JOBS=${JOBS:-4}
KEY_FILE=${KEY_FILE:-$HOME/.config/strata/flashnext.key}
CONFIG_TMPL=$REPO/configs/strata-exl3-full512-final-g4-lru-ht.json.tmpl
CONFIG=$HOME/shim/qwen38fn-run.json
STAMPS=$HOME/.qwen38fn-run/stamps; mkdir -p "$STAMPS"
CACHE=$REAL_HOME/.cache/qwen38fn-run; mkdir -p "$CACHE"
PYPIN=3.12.15; PYFALLBACK=3.12
BUILD_ROOT=/Users/sero      # the path the bundle was built at (strata_cmds.txt carries it)
export PATH=$HOME/.local/bin:$REAL_HOME/.local/bin:$PATH

NOK=0; NWARN=0; NBAD=0
step() { echo; echo "== $*"; }
ok()   { echo "[ ok ] $*"; NOK=$((NOK + 1)); }
info() { echo "[info] $*"; }
warn() { echo "[warn] $*"; NWARN=$((NWARN + 1)); }
bad()  { echo "[FAIL] $*"; NBAD=$((NBAD + 1)); }
die()  { echo "[FAIL] $*" >&2; exit 1; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
hash_files() { cat "$@" | shasum -a 256 | cut -c1-16; }

# ------------------------------------------------------------------ stamps: rebuild only when inputs changed or the output is gone
need_build() {   # need_build <name> <output> <inputs-hash>   (0 = build)
  local name=$1 out=$2 h=$3
  [ $FORCE -eq 1 ] && return 0
  [ -e "$out" ] || return 0
  [ ! -f "$STAMPS/$name" ] && return 1        # an output we did not build (e.g. the live one on the mini): kept
  [ "$(cat "$STAMPS/$name")" = "$h" ] && return 1
  return 0
}
stamp() { echo "$3" > "$STAMPS/$1"; }

# ------------------------------------------------------------------ copying policy
install_file() {   # install_file <src> <dst>
  mkdir -p "$(dirname "$2")"
  if [ ! -e "$2" ]; then cp -p "$1" "$2"; ok "installed $2"
  elif cmp -s "$1" "$2"; then ok "up to date $2"
  elif [ $FORCE -eq 1 ]; then cp -p "$1" "$2"; ok "replaced $2 (--force)"
  else warn "$2 differs from the bundle's/repo's copy: kept (use --force to replace)"; fi
}
install_tree() {   # install_tree <src-dir> <dst-dir> <label> [exclude...]: add what is missing, report (and with --force replace) what differs, never delete
  local src=$1 dst=$2 label=$3; shift 3
  local ex=() dx=() e
  for e in "$@"; do ex+=(--exclude "$e"); dx+=(-x "$e"); done
  mkdir -p "$dst"
  tar -C "$src" ${ex[@]+"${ex[@]}"} -cf - . | tar -C "$dst" -xkf - 2>/dev/null || true      # -k: keep files that already exist
  local drift; drift=$(diff -rq ${dx[@]+"${dx[@]}"} "$src" "$dst" 2>/dev/null | grep -v '^Only in '"$dst" || true)
  if [ -z "$drift" ]; then ok "$label: $dst up to date"
  elif [ $FORCE -eq 1 ]; then tar -C "$src" ${ex[@]+"${ex[@]}"} -cf - . | tar -C "$dst" -xf -; ok "$label: $dst replaced (--force)"
  else warn "$label: $(echo "$drift" | wc -l | tr -d ' ') file(s) in $dst differ from the bundle's, kept (--force replaces); first: $(echo "$drift" | head -1)"; fi
}

# ------------------------------------------------------------------ preflight
pick_sdk() {
  if [ -n "${SDK:-}" ]; then [ -d "$SDK" ] || die "SDK=$SDK does not exist"; return; fi
  local d=/Library/Developer/CommandLineTools/SDKs c
  if [ -d $d/MacOSX26.5.sdk ]; then SDK=$d/MacOSX26.5.sdk
  else
    c=$(ls -d $d/MacOSX26.*.sdk 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$c" ] || c=$(ls -d $d/MacOSX26.sdk 2>/dev/null || true)
    if [ -n "$c" ]; then SDK=$c; warn "SDK 26.5 not installed: using $(basename "$c") (the libraries were verified byte-identical with 26.5 and 26.0)"
    else SDK=$(xcrun --show-sdk-path); warn "no 26.x SDK under $d: using $SDK; an SDK >= 27 breaks ld with 'malformed file libSystem.B.tbd'"; fi
  fi
  export SDK; ok "SDK $SDK"
}
usb_dock_present() {   # passive IORegistry read; opens nothing. ASM2464 dock ids 3801:0001 / ADD1:0001 = vendor 14337 / 44497
  ioreg -p IOUSB -l -w0 2>/dev/null | grep -E '"idVendor" = (14337|44497)$' >/dev/null
}
preflight() {
  step "preflight"
  [ "$(uname -s)" = Darwin ] || die "needs macOS (the serving runtime is native macOS: libusb to the dock + Metal; it cannot run in a container)"
  [ "$(uname -m)" = arm64 ] || die "needs Apple silicon (arm64), this is $(uname -m)"
  ok "macOS $(sw_vers -productVersion) on $(uname -m)"
  xcode-select -p >/dev/null 2>&1 || die "Xcode Command Line Tools missing: xcode-select --install"
  command -v clang >/dev/null && command -v clang++ >/dev/null || die "clang not found (Command Line Tools)"
  ok "Command Line Tools: $(clang --version | head -1)"
  pick_sdk
  local cv; cv=$(clang --version | sed -n 's/^Apple clang version \([0-9]*\).*/\1/p')
  if [ -n "${CLANG_BIN:-}" ]; then ok "CUDA host compiles use $CLANG_BIN/clang: $("$CLANG_BIN/clang" --version | head -1)"
  elif [ -n "$cv" ] && [ "$cv" -lt 21 ]; then warn "Apple clang $cv: the engine's CUDA host compile was built with Apple clang 21 (the mini's) and fails with older ones (__ldcs overloads); set CLANG_BIN=<dir of an LLVM >= 21, e.g. /opt/homebrew/opt/llvm@22/bin>"
  else ok "clang: $(clang --version | head -1)"; fi
  if xcrun -f metal >/dev/null 2>&1; then HAVE_METAL=1; ok "Metal toolchain: $(xcrun -f metal)"
  else HAVE_METAL=0; warn "no Metal compiler (xcrun -f metal fails): libexl3metal and strata-vision-metal come from $BUNDLE/metal (./run.sh --build-metal on a Mac that has it)"; fi
  command -v uv >/dev/null || die "uv not found: https://docs.astral.sh/uv/getting-started/installation/ (pinned toolchain: uv 0.12.24, Python $PYPIN)"
  ok "uv $(uv --version | cut -d' ' -f2)"
  command -v git >/dev/null || die "git not found"
  command -v curl >/dev/null || die "curl not found"
  if usb_dock_present; then ok "USB dock present (ASM2464 3801:0001 / ADD1:0001)"
  elif [ $CHECK -eq 1 ] || [ "$MODE" != up ]; then warn "no USB dock (3801:0001 / ADD1:0001) on this machine: fine for --check / --build-metal, the engine cannot start without it"
  else die "the RTX 3090 dock (USB 3801:0001 / ADD1:0001) is not on the bus; plug it in (and power the card) first"; fi
  local free; free=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
  info "free disk at $ROOT: ${free} GiB (installed pack 46 GB + dense GGUF 84 GB + builds)"
  if pgrep -x strata >/dev/null 2>&1; then
    if [ $CHECK -eq 1 ] || [ "$MODE" != up ]; then info "an engine ('strata') is running on this machine: left alone"
    elif [ $RESTART -eq 0 ]; then die "an engine is already running (pgrep -x strata); ./run.sh --restart stops it first"; fi
  fi
}

# ------------------------------------------------------------------ bundle
verify_bundle() {
  step "bundle $BUNDLE"
  [ -f "$BUNDLE/SHA256SUMS" ] || die "no build bundle at $BUNDLE: run  docker compose run --rm build  (or make) first"
  (cd "$BUNDLE" && shasum -a 256 -c SHA256SUMS --quiet) || die "bundle checksum mismatch (rebuild it: make clean && make)"
  ok "bundle files match their SHA256SUMS ($(wc -l < "$BUNDLE/SHA256SUMS" | tr -d ' ') files)"
  head -8 "$BUNDLE/BUILD_INFO.txt" | sed 's/^/       /'
  BUNDLE_ID=$(shasum -a 256 "$BUNDLE/SHA256SUMS" | cut -c1-16)
}

# ------------------------------------------------------------------ toolchain pieces
ensure_venv() {   # ensure_venv <dir>
  [ -x "$1/bin/python" ] && return 0
  mkdir -p "$(dirname "$1")"
  if uv venv -q --python $PYPIN "$1" 2>/dev/null; then :
  elif uv venv -q --python $PYFALLBACK "$1"; then warn "Python $PYPIN not available to uv here: $1 uses $("$1/bin/python" -V)"
  else die "cannot create a Python $PYFALLBACK venv at $1"; fi
}
ensure_cuda_headers() {   # CUDA 12.8 headers for clang (host side of the CUDA files); the same four redist tarballs as docs/reproduce.md
  step "CUDA 12.8 headers"
  if [ -f "$HOME/cuda-12.8/include/cuda_runtime.h" ]; then ok "$HOME/cuda-12.8 present"; stub_curand; return; fi
  local tmp; tmp=$(mktemp -d); local sha p f
  mkdir -p "$HOME/cuda-12.8"
  while read -r sha p; do
    f=$CACHE/$(basename "$p")
    [ -f "$f" ] || curl -fsSL --retry 3 -o "$f" "https://developer.download.nvidia.com/compute/cuda/redist/$p"
    [ "$(sha_of "$f")" = "$sha" ] || die "sha256 mismatch for $f (expected $sha)"
    tar -xJf "$f" -C "$tmp"
  done < "$REPO/shim/cuda-redist-12.8.sha256"
  for p in "$tmp"/*/; do cp -R "$p". "$HOME/cuda-12.8/"; done; rm -rf "$tmp"
  [ -f "$HOME/cuda-12.8/include/cuda_runtime.h" ] || die "CUDA 12.8 merge produced no include/cuda_runtime.h"
  ok "installed $HOME/cuda-12.8 (4 tarballs, sha256 verified)"
  stub_curand
}
stub_curand() {   # clang <= 17's CUDA wrapper force-includes curand_mtgp32_kernel.h (Apple clang 21, which the mini has, does not): an empty stand-in is enough for host-only compiles
  local w; w=$(clang -print-resource-dir)/include/__clang_cuda_runtime_wrapper.h
  if [ -f "$w" ] && grep -q '#include "curand_mtgp32_kernel.h"' "$w" && [ ! -f "$HOME/cuda-12.8/include/curand_mtgp32_kernel.h" ]; then
    echo '// empty stand-in written by run.sh: this clang force-includes the header, nothing here uses curand' > "$HOME/cuda-12.8/include/curand_mtgp32_kernel.h"
    warn "this clang ($(clang --version | head -1)) force-includes curand_mtgp32_kernel.h: wrote an empty stand-in into $HOME/cuda-12.8/include"
  fi
}
ensure_simde() {
  step "SIMDe (x86 intrinsics on arm64 for Strata's CPU kernels)"
  local pin=0ff5341927e15df0c3da8bb13bb32ec7f95aef84
  if [ -d "$HOME/simde/.git" ] && [ "$(git -C "$HOME/simde" rev-parse HEAD 2>/dev/null)" = $pin ]; then ok "$HOME/simde at $pin"; return; fi
  [ -e "$HOME/simde" ] && { warn "$HOME/simde exists but is not at $pin: kept"; return; }
  git init -q "$HOME/simde" && git -C "$HOME/simde" fetch -q --depth 1 https://github.com/simd-everywhere/simde $pin && git -C "$HOME/simde" checkout -q FETCH_HEAD
  [ "$(git -C "$HOME/simde" rev-parse HEAD)" = $pin ] || die "simde checkout is not $pin"
  ok "simde at $pin"
}
ensure_tinygrad() {
  step "tinygrad (PR #17369 head e765a2abf) + patch, venv"
  local pin=e765a2abf75c1f3a37e810f8856550397f2bcc23 tg=$HOME/tg-strata
  if [ -d "$tg/.git" ] && [ "$(git -C "$tg" rev-parse HEAD)" = $pin ]; then ok "$tg at $pin"
  elif [ -e "$tg" ]; then warn "$tg exists and is not at $pin: kept"
  else
    git init -q "$tg"; git -C "$tg" remote add origin https://github.com/tinygrad/tinygrad
    git -C "$tg" fetch -q --depth 100 origin pull/17369/head
    git -C "$tg" cat-file -e $pin^{commit} 2>/dev/null || git -C "$tg" fetch -q --unshallow origin pull/17369/head
    git -C "$tg" checkout -q $pin; ok "tinygrad at $pin"
  fi
  if git -C "$tg" apply --check -R "$REPO/patches/tinygrad/0001-tg-strata-nv-usb-fixes.patch" 2>/dev/null; then ok "tinygrad patch 0001 already applied"
  elif git -C "$tg" apply --check "$REPO/patches/tinygrad/0001-tg-strata-nv-usb-fixes.patch" 2>/dev/null; then git -C "$tg" apply "$REPO/patches/tinygrad/0001-tg-strata-nv-usb-fixes.patch"; ok "applied tinygrad patch 0001"
  else warn "tinygrad patch 0001 neither applied nor applicable in $tg"; fi
  ensure_venv "$HOME/tinygrad/.venv"
  if "$HOME/tinygrad/.venv/bin/python" -c "import numpy, usb1, libusb_package, pytest" 2>/dev/null; then ok "tinygrad venv has numpy, libusb1, libusb-package, pytest"
  else
    uv pip install -q --python "$HOME/tinygrad/.venv/bin/python" numpy==2.5.3 libusb1==3.4.0 libusb-package==1.0.30.0 ninja==1.13.2 pytest==9.1.1
    ok "tinygrad venv: numpy 2.5.3, libusb1 3.4.0, libusb-package 1.0.30.0, ninja 1.13.2, pytest 9.1.1"
  fi
  LIBUSB=$("$HOME/tinygrad/.venv/bin/python" -c "import libusb_package, os; print(os.path.join(os.path.dirname(libusb_package.__file__), 'libusb-1.0.dylib'))")
  [ -f "$LIBUSB" ] || die "libusb-1.0.dylib not found at $LIBUSB"
  if [ -f "$HOME/.local/lib/libamd_comgr.dylib" ]; then ok "libamd_comgr.dylib present (COMGR_PATH)"
  else warn "$HOME/.local/lib/libamd_comgr.dylib is missing: its source was never recorded (docs/reproduce.md); whether the NV backend needs it was not established"; fi
  if [ ! -f "$HOME/.chestnut_env" ]; then
    { echo '# written by run.sh (paths only, no secrets)'
      echo 'export PATH="$HOME/.local/bin:$PATH"'
      echo "export LIBUSB_PATH=\"$LIBUSB\""
      [ -f "$HOME/.local/lib/libamd_comgr.dylib" ] && echo 'export COMGR_PATH="$HOME/.local/lib/libamd_comgr.dylib"'
      true; } > "$HOME/.chestnut_env"; ok "wrote $HOME/.chestnut_env"
  else ok "$HOME/.chestnut_env exists: kept"; fi
  if "$HOME/tinygrad/.venv/bin/python" -c "import sys; sys.path.insert(0, '$tg'); import tinygrad.runtime.ops_nv" 2>/dev/null; then ok "tinygrad NV backend imports (no device opened)"
  else bad "tinygrad's NV backend does not import in $HOME/tinygrad/.venv"; fi
}
strata_python() { echo "$HOME/strata/.venv/bin/python"; }
ensure_strata_venv() {
  ensure_venv "$HOME/strata/.venv"
  if "$(strata_python)" -c "import numpy, jinja2, regex, yaml, tqdm, requests, PIL, psutil" 2>/dev/null; then ok "Strata venv has its requirements"
  else uv pip install -q --python "$(strata_python)" -r "$HOME/strata/requirements.txt"; ok "Strata venv: requirements.txt (pinned) installed"; fi
}

# ------------------------------------------------------------------ install
install_bundle() {
  step "install the bundle under $HOME"
  install_tree "$BUNDLE/engine/strata" "$HOME/strata" "Strata v0.1.41 + patches" .venv .git __pycache__
  install_tree "$BUNDLE/engine/strata-linux" "$HOME/strata-linux" "Strata configure tree (llama.cpp sources, compile commands)"
  # strata_cmds.txt carries the container's build root; point it at this install root
  if [ "$ROOT" != "$BUILD_ROOT" ] && grep -q "$BUILD_ROOT" "$HOME/strata-linux/strata_cmds.txt"; then
    sed "s#$BUILD_ROOT#$ROOT#g" "$HOME/strata-linux/strata_cmds.txt" > "$HOME/strata-linux/strata_cmds.txt.new" && mv "$HOME/strata-linux/strata_cmds.txt.new" "$HOME/strata-linux/strata_cmds.txt"
    ok "strata_cmds.txt re-rooted $BUILD_ROOT -> $ROOT"
  fi
  install_tree "$REPO/exl3" "$HOME/exl3" "EXL3 sources, packers, profile" td pack out __pycache__ '*.dylib' '*.fatbin' '*.air' '*.metallib' exl3_metallib.h
  local m; for m in exl3_strata exl3_strata_recon exl3_strata_pf; do install_file "$BUNDLE/fatbins/$m.fatbin" "$HOME/exl3/$m.fatbin"; done
  # the shim, exactly as shim/install.sh lays it out
  local f
  for f in tgcu.py tgcu_api.py tgcu_prof.py usb_drain.py pcie_cycle.py tgcudart.cpp tgcublas.cpp tgcuda_launch.h tgcu_builtin.cu tgcu_libc.ptx \
           run-strata.sh srv.sh ls_srv.sh mkcfg.py macbuild.py build-libcudart.sh; do install_file "$REPO/shim/$f" "$HOME/shim/$f"; done
  install_tree "$REPO/shim/include" "$HOME/shim/include" "shim headers"
  install_tree "$REPO/shim/x86compat" "$HOME/shim/x86compat" "shim x86 compat headers"
  install_file "$BUNDLE/fatbins/tgcu_builtin.fatbin" "$HOME/shim/tgcu_builtin.fatbin"
  install_file "$BUNDLE/fatbins/tgcu_libc.cubin" "$HOME/shim/tgcu_libc.cubin"
  chmod +x "$HOME"/shim/*.sh 2>/dev/null || true
}

# ------------------------------------------------------------------ models
link_file() {   # link_file <src> <dst>: APFS clone when on the same volume (instant, no extra space), else a symlink
  [ -e "$2" ] && return 0
  mkdir -p "$(dirname "$2")"; cp -c "$1" "$2" 2>/dev/null || ln -s "$1" "$2"
}
install_models() {
  step "models"
  local M=$BUNDLE/models
  if [ -d "$M" ]; then
    [ -f "$M/exl3/pack/experts.bin" ] && { link_file "$M/exl3/pack/experts.bin" "$HOME/exl3/pack/experts.bin"
      for f in exl3_pack.json exl3_experts.txt; do [ -f "$M/exl3/pack/$f" ] && install_file "$M/exl3/pack/$f" "$HOME/exl3/pack/$f"; done; }
    [ -d "$M/exl3/td" ] && install_tree "$M/exl3/td" "$HOME/exl3/td" "turboderp shard index"
    local g; for g in "$M"/qwen38fn/*.gguf; do [ -f "$g" ] && link_file "$g" "$HOME/models/qwen38fn/$(basename "$g")"; done
  fi
  local pk=$HOME/exl3/pack
  if [ -f "$pk/experts.bin" ]; then
    local sz; sz=$(stat -f %z "$pk/experts.bin")
    [ "$sz" = 45770342400 ] && ok "experts.bin: 45,770,342,400 B (48 layers x 953,548,800)" || bad "experts.bin is $sz B, expected 45,770,342,400"
    [ -f "$pk/exl3_experts.txt" ] && ok "exl3_experts.txt marker present" || bad "no $pk/exl3_experts.txt (python exl3/pack_exl3.py marker, or the fetch container writes it)"
  else MISSING=1; warn "no experts pack at $pk/experts.bin: docker compose --profile fetch run --rm fetch experts  (then re-run; it is read from $BUNDLE/models)"; fi
  local q=$HOME/models/qwen38fn
  for f in Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf mmproj-Qwen3.8-Flash-Next-BF16.gguf; do
    if [ -e "$q/$f" ]; then ok "$f present"; else MISSING=1; warn "missing $q/$f: docker compose --profile fetch run --rm fetch dense"; fi
  done
  if [ $VERIFY_PACK -eq 1 ] && [ -f "$pk/experts.bin" ]; then
    info "sha256 of all 48 layers against results/pack/exl3_pack.json (reads 45.8 GB) ..."
    cp "$REPO/results/pack/exl3_pack.json" "$pk/exl3_pack.json.reference"
    ( cd "$HOME/exl3" && "$(strata_python)" pack_td.py verify --sha-only ) > "$CACHE/verify-pack.log" 2>&1 && ok "pack sha256 == manifest ($(tail -1 "$CACHE/verify-pack.log"))" || { tail -5 "$CACHE/verify-pack.log"; bad "pack sha256 does not match the manifest"; }
  fi
}
build_packs() {   # dense pack + tokenizer + MTP draft (Strata's tools, docs/reproduce.md section 4); not exercised in this repo's verification
  step "dense pack, tokenizer, MTP draft"
  local q=$HOME/models/qwen38fn py; py=$(strata_python)
  if [ -f "$q/packs/iq3_s/dense.bin" ] && [ -f "$q/mtp/rt/dense.bin" ] && [ -f "$q/mtp/rt/experts.bin" ]; then ok "packs present"; return; fi
  local g1=$q/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  if [ ! -f "$g1" ]; then warn "no GGUF yet: skipping iq_pack / MTP (fetch the models first)"; return; fi
  if [ $CHECK -eq 1 ]; then info "--check: would run tools/iq_pack.py, mtp_fetch.py, mtp_pack.py, mtp_rt.py (5.2 GB range fetch)"; return; fi
  [ -f "$q/packs/iq3_s/dense.bin" ] || STRATA_GGUF_PY="$HOME/strata-linux/_deps/strata_llamacpp-src/gguf-py" "$py" "$HOME/strata/tools/iq_pack.py" --gguf "$g1" --out "$q/packs/iq3_s"
  if [ ! -f "$q/mtp/rt/dense.bin" ]; then
    (cd "$HOME/strata" && "$py" tools/mtp_fetch.py fetch --out "$q/mtp" && "$py" tools/mtp_pack.py --src "$q/mtp" --experts q2_0 --out "$q/mtp/mtp-q2_0.gguf" \
       && "$py" tools/mtp_rt.py --gguf "$q/mtp/mtp-q2_0.gguf" --out "$q/mtp/rt" && cp data/draft_vocab.bin "$q/mtp/rt/draft_vocab.bin")
  fi
  ok "dense pack + MTP draft built"
}
assemble_pack_dir() {
  local d=$HOME/exl3/pack_full512 q=$HOME/models/qwen38fn
  mkdir -p "$d"
  if [ -d "$q/packs/iq3_s" ]; then for f in conversions.json dense.bin index.txt tokenizer; do [ -e "$d/$f" ] || ln -s "$q/packs/iq3_s/$f" "$d/$f"; done; fi
  [ -e "$d/experts.bin" ] || { [ -f "$HOME/exl3/pack/experts.bin" ] && ln -s "$HOME/exl3/pack/experts.bin" "$d/experts.bin"; } || true
  [ -f "$d/exl3_experts.txt" ] || { [ -f "$HOME/exl3/pack/exl3_experts.txt" ] && cp "$HOME/exl3/pack/exl3_experts.txt" "$d/"; } || true
  [ -f "$HOME/exl3/placement/f512c/profile.bin" ] && ok "expert profile present" || bad "missing $HOME/exl3/placement/f512c/profile.bin"
}

# ------------------------------------------------------------------ native builds
cuda_clang() { if [ -n "${CLANG_BIN:-}" ]; then PATH="$CLANG_BIN:$PATH" "$@"; else "$@"; fi; }
expected_sha() {   # expected_sha <label> <file> <sha256>: info only (SDK / toolchain dependent)
  local got; got=$(sha_of "$2")
  if [ "$got" = "$3" ]; then ok "$1 sha256 ${got:0:16}... = the deployed one"; else info "$1 sha256 ${got:0:16}... (deployed ${3:0:16}...)"; fi
}
build_libs() {
  step "native libraries"
  local py_prefix h
  # libexl3cpu (NEON tier)
  h=$(hash_files "$REPO/exl3/cpu/exl3_cpu.c" "$REPO/exl3/cpu/build.sh")
  if need_build libexl3cpu "$HOME/exl3/cpu/libexl3cpu.dylib" "$h"; then zsh "$HOME/exl3/cpu/build.sh" "$HOME/exl3/cpu" > "$CACHE/libexl3cpu.log" 2>&1 || { tail -20 "$CACHE/libexl3cpu.log"; die "libexl3cpu build failed"; }; stamp libexl3cpu x "$h"; ok "built libexl3cpu.dylib"
  else ok "libexl3cpu.dylib present"; fi
  expected_sha libexl3cpu "$HOME/exl3/cpu/libexl3cpu.dylib" 9fe9cab5491e32e39fcbedfc93043f017f7f1ed60cd65d5211e96a436e42ef3c
  # libexl3metal (Metal)
  h=$(hash_files "$REPO/exl3/metal/exl3_metal.metal" "$REPO/exl3/metal/exl3_metal.mm" "$REPO/exl3/metal/exl3_metal.h" "$REPO/exl3/metal/build.sh")
  if need_build libexl3metal "$HOME/exl3/metal/libexl3metal.dylib" "$h"; then
    if [ $HAVE_METAL -eq 1 ]; then bash "$HOME/exl3/metal/build.sh" "$HOME/exl3/metal" > "$CACHE/libexl3metal.log" 2>&1 || { tail -20 "$CACHE/libexl3metal.log"; die "libexl3metal build failed"; }; stamp libexl3metal x "$h"; ok "built libexl3metal.dylib"
    elif [ -f "$BUNDLE/metal/libexl3metal.dylib" ]; then mkdir -p "$HOME/exl3/metal"; cp "$BUNDLE/metal/libexl3metal.dylib" "$HOME/exl3/metal/libexl3metal.dylib"; ok "installed libexl3metal.dylib from $BUNDLE/metal"
    else bad "no Metal toolchain and no $BUNDLE/metal/libexl3metal.dylib: run ./run.sh --build-metal on a Mac with the Metal toolchain"; fi
  else ok "libexl3metal.dylib present"; fi
  [ -f "$HOME/exl3/metal/libexl3metal.dylib" ] && expected_sha libexl3metal "$HOME/exl3/metal/libexl3metal.dylib" e38a011c79933a8efdc31e7b15789a7ff63aa5e62597f5b4f68995070c4e6ea9
  # libcudart (the CUDA runtime shim) embeds CPython: use the interpreter the tinygrad venv runs on
  py_prefix=$("$HOME/tinygrad/.venv/bin/python" -c "import sys; print(sys.base_prefix)")
  h=$(hash_files "$REPO/shim/tgcudart.cpp" "$REPO/shim/tgcublas.cpp" "$REPO/shim/tgcuda_launch.h" "$REPO/shim/build-libcudart.sh"); h="$h-$(basename "$py_prefix")"
  if need_build libcudart "$HOME/shim/libcudart.dylib" "$h"; then
    PY=$py_prefix CUDA_HOST=$HOME/cuda-12.8 zsh "$HOME/shim/build-libcudart.sh" "$HOME/shim" > "$CACHE/libcudart.log" 2>&1 || { tail -20 "$CACHE/libcudart.log"; die "libcudart build failed"; }
    stamp libcudart x "$h"; ok "built libcudart.dylib ($(nm -gU "$HOME/shim/libcudart.dylib" | wc -l | tr -d ' ') exported symbols; the deployed one has 136 + the tgcuHs* ones of its newer source)"
  else ok "libcudart.dylib present"; fi
  # libstrata_exl3 (host library around the EXL3 fatbins)
  h=$(hash_files "$REPO/exl3/strata_m4/exl3_strata_host.cu" "$REPO/exl3/strata_m4/exl3_strata_recon_host.cu" "$REPO/exl3/strata_m4/exl3_strata_pf_host.cu" "$BUNDLE/fatbins/exl3_strata.fatbin" "$BUNDLE/fatbins/exl3_strata_recon.fatbin" "$BUNDLE/fatbins/exl3_strata_pf.fatbin" "$HOME/shim/libcudart.dylib")
  if need_build libstrata_exl3 "$HOME/exl3/strata_m4/libstrata_exl3.dylib" "$h"; then
    cuda_clang zsh "$HOME/exl3/strata_m4/build.sh" > "$CACHE/libstrata_exl3.log" 2>&1 || { tail -20 "$CACHE/libstrata_exl3.log"; die "libstrata_exl3 build failed"; }
    stamp libstrata_exl3 x "$h"; ok "built libstrata_exl3.dylib"
  else ok "libstrata_exl3.dylib present"; fi
}
build_engine() {
  step "Strata engine (host half; the device code is in the bundle)"
  local h; h=$(hash_files "$HOME/strata-linux/strata_cmds.txt" "$REPO/shim/macbuild.py" "$BUNDLE/engine/strata-mac-fatbins.sha256")
  if need_build strata "$HOME/strata-mac/strata" "$h"; then
    SDK=$SDK python3 "$HOME/shim/macbuild.py" --lin "$HOME/strata-linux" > "$CACHE/macbuild.log" 2>&1 || { tail -20 "$CACHE/macbuild.log"; die "macbuild.py failed"; }
    tail -1 "$CACHE/macbuild.log" | sed 's/^/       /'
    ( cd "$BUNDLE/engine/strata-mac-fatbins" && tar cf - . ) | tar -C "$HOME/strata-mac" -xf -
    ok "fatbins placed under $HOME/strata-mac ($(find "$HOME/strata-mac" -name '*.fatbin' | wc -l | tr -d ' '))"
    info "host build: $JOBS jobs (135 clang rules + link) ..."
    cuda_clang nice -n 10 "$HOME/strata/.venv/bin/ninja" -C "$HOME/strata-mac" -f host.ninja -j "$JOBS" > "$CACHE/host-ninja.log" 2>&1 || { tail -30 "$CACHE/host-ninja.log"; die "host.ninja failed (log $CACHE/host-ninja.log)"; }
    stamp strata x "$h"; ok "built $HOME/strata-mac/strata"
  else ok "$HOME/strata-mac/strata present"; fi
  # the binary must resolve the shim's libcudart through its rpath
  if otool -L "$HOME/strata-mac/strata" | grep -q '@rpath/libcudart.dylib' && otool -l "$HOME/strata-mac/strata" | grep -A2 LC_RPATH | grep -q "$HOME/shim"; then ok "engine links @rpath/libcudart.dylib with rpath $HOME/shim"
  else bad "engine binary does not link libcudart through rpath $HOME/shim"; fi
  [ -f "$HOME/shim/libcudart.dylib" ] || bad "$HOME/shim/libcudart.dylib missing"
}
build_vision() {
  step "vision encoder (Metal)"
  local dest=$HOME/strata-mac-vision/strata-vision-metal
  local h; h=$(hash_files "$HOME/strata/tools/vision/CMakeLists.txt" "$REPO/vision/build_vision_mac.sh")
  if need_build vision "$dest" "$h"; then
    if [ $HAVE_METAL -eq 1 ]; then
      ensure_strata_venv
      SRC=$HOME/strata/tools/vision LLAMA=$HOME/strata-linux/_deps/strata_llamacpp-src OUT=$HOME/strata-mac-vision/build-vision DEST=$dest \
        bash "$REPO/vision/build_vision_mac.sh" > "$CACHE/vision.log" 2>&1 || { tail -20 "$CACHE/vision.log"; die "vision build failed"; }
      stamp vision x "$h"; ok "built strata-vision-metal ($(stat -f %z "$dest") B)"
    elif [ -f "$BUNDLE/metal/strata-vision-metal" ]; then mkdir -p "$(dirname "$dest")"; cp "$BUNDLE/metal/strata-vision-metal" "$dest"; codesign -s - -f "$dest" 2>/dev/null || true; ok "installed strata-vision-metal from $BUNDLE/metal"
    else bad "no Metal toolchain and no $BUNDLE/metal/strata-vision-metal: run ./run.sh --build-metal on a Mac with the Metal toolchain"; fi
  else ok "strata-vision-metal present"; fi
}
build_metal_only() {   # ./run.sh --build-metal: products into $BUNDLE/metal, nothing installed
  step "Metal binaries -> $BUNDLE/metal"
  [ $HAVE_METAL -eq 1 ] || die "this Mac has no Metal toolchain either (xcrun -f metal): install Xcode, or 'xcodebuild -downloadComponent MetalToolchain'"
  verify_bundle
  mkdir -p "$BUNDLE/metal"
  install_tree "$BUNDLE/engine/strata" "$HOME/strata" "Strata v0.1.41 + patches" .venv .git __pycache__
  install_tree "$BUNDLE/engine/strata-linux" "$HOME/strata-linux" "Strata configure tree"
  ensure_strata_venv
  bash "$REPO/exl3/metal/build.sh" "$CACHE/metal-build" > "$CACHE/libexl3metal.log" 2>&1 || { tail -20 "$CACHE/libexl3metal.log"; die "libexl3metal build failed"; }
  cp "$CACHE/metal-build/libexl3metal.dylib" "$BUNDLE/metal/"
  SRC=$HOME/strata/tools/vision LLAMA=$HOME/strata-linux/_deps/strata_llamacpp-src OUT=$CACHE/vision-build DEST=$BUNDLE/metal/strata-vision-metal \
    bash "$REPO/vision/build_vision_mac.sh" > "$CACHE/vision.log" 2>&1 || { tail -20 "$CACHE/vision.log"; die "vision build failed"; }
  (cd "$BUNDLE/metal" && shasum -a 256 libexl3metal.dylib strata-vision-metal > SHA256SUMS)
  ok "built $BUNDLE/metal/{libexl3metal.dylib,strata-vision-metal}"; cat "$BUNDLE/metal/SHA256SUMS"
  echo "copy ./out/metal to the mini's ./out/metal, then run ./run.sh there"
}

# ------------------------------------------------------------------ config, key, start
ensure_key() {
  if [ -s "$KEY_FILE" ]; then ok "API key file present ($KEY_FILE, not printed)"; return; fi
  mkdir -p "$(dirname "$KEY_FILE")"; chmod 700 "$(dirname "$KEY_FILE")" 2>/dev/null || true
  ( umask 077; head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$KEY_FILE"; echo >> "$KEY_FILE" )
  [ -s "$KEY_FILE" ] || die "could not write $KEY_FILE"
  ok "generated an API key in $KEY_FILE (mode 600, not printed)"
}
render_config() {
  step "config"
  mkdir -p "$HOME/shim"
  sh "$REPO/configs/render.sh" "$CONFIG_TMPL" "$HOME" > "$CONFIG.new" && mv "$CONFIG.new" "$CONFIG"
  python3 -c "import json,sys; json.load(open('$CONFIG'))" && ok "rendered $CONFIG (host $HOST, port $PORT; key from $KEY_FILE at start)"
  local missing=0 f
  for f in $(python3 - "$CONFIG" <<'PY'
import json, sys, os
c = json.load(open(sys.argv[1]))
paths = [c["exe"], c["cwd"], c["tokenizer"], c["vision"]["exe"], c["vision"]["mmproj"], c["vision"]["model"]]
a = c["args"]
for i, x in enumerate(a):
    if x.startswith("/") and a[i - 1].startswith("--"): paths.append(x)
print(" ".join(paths))
PY
  ); do [ -e "$f" ] || { missing=$((missing + 1)); echo "       missing: $f"; }; done
  if [ $missing -eq 0 ]; then ok "every path the config names exists"
  elif [ $CHECK -eq 1 ]; then warn "$missing path(s) named by the config do not exist here (models/packs: expected on a machine that has not fetched them)"
  else bad "$missing path(s) named by the config do not exist"; fi
}
stop_server() {
  local pf=$HOME/shim/server.pid
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    kill -TERM "$(cat "$pf")"; local i; for i in $(seq 1 90); do pgrep -x strata >/dev/null 2>&1 || break; sleep 1; done
    pgrep -x strata >/dev/null 2>&1 && die "engine still alive 90 s after SIGTERM (two engines fight over the USB device: not starting another)"
    ok "server stopped"
  else info "no server of ours running (no live pid in $pf)"; fi
}
start_server() {
  step "start"
  local pf=$HOME/shim/server.pid log=$HOME/shim/serve_run.log
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    [ $RESTART -eq 1 ] && stop_server || die "a server is already running (pid $(cat "$pf")): ./run.sh --restart"
  fi
  [ $RESTART -eq 1 ] && pgrep -x strata >/dev/null 2>&1 && stop_server
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && die "port $PORT is in use"
  local key; key=$(tr -d '\n' < "$KEY_FILE")
  ( cd "$HOME/strata"; STRATA_PREFILL_CPU_SHARE=1 STRATA_API_KEY="$key" nohup "$(strata_python)" serve/server.py --engine strata --config "$CONFIG" --host "$HOST" --port "$PORT" > "$log" 2>&1 < /dev/null & echo $! > "$pf" )
  unset key
  info "server pid $(cat "$pf"), log $log; waiting for ready (startup took ~200 s on the mini) ..."
  local i; for i in $(seq 1 180); do
    grep -q '^ready' "$log" 2>/dev/null && break
    kill -0 "$(cat "$pf")" 2>/dev/null || { tail -20 "$log"; die "server died during startup"; }
    sleep 5
  done
  grep -q '^ready' "$log" || { tail -20 "$log"; die "no 'ready' within 900 s"; }
  ok "READY: http://$HOST:$PORT/v1 (Authorization: Bearer \$(cat $KEY_FILE))"
}

summary() {
  echo; echo "== summary: $NOK ok, $NWARN warnings, $NBAD failures$([ $CHECK -eq 1 ] && echo ' (--check: engine not started, USB not touched)')"
  [ $NBAD -eq 0 ]
}

# ------------------------------------------------------------------ main
MISSING=0; HAVE_METAL=0
case $MODE in
  stop) stop_server; exit 0 ;;
  build-metal) preflight; build_metal_only; exit 0 ;;
esac
preflight
verify_bundle
ensure_cuda_headers
ensure_simde
ensure_tinygrad
install_bundle
ensure_strata_venv
install_models
build_packs
assemble_pack_dir
build_libs
build_engine
build_vision
ensure_key
render_config
if [ $CHECK -eq 1 ]; then summary; exit $?; fi
[ $NBAD -eq 0 ] || { summary || true; die "not starting: fix the failures above"; }
start_server
summary
