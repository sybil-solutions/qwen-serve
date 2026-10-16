# Reproduce

From a bare Mac mini to the server whose numbers are in [results.md](results.md). Every version below was read from the machine that produced the
results (2026-10-12). Anything marked **not pinned** could not be recovered from the machine or its logs; **[not re-run]** marks steps that were
reconstructed from build files and logs but not executed again for this repo. Steps that were re-run card-free are listed in
[results.md](results.md) section 8 and are runnable with `scripts/verify-card-free.sh`.

**Automated path:** `docker compose run --rm build` (the Linux/CUDA half, pinned) followed by `./run.sh` on the Mac does sections 2-6 below in one go; what was run and what was not is in [section 10](#10-one-click-what-was-run).
The numbered steps remain the manual reference.

Paths: `$HOME` is the mini's home. The shim must live in `$HOME/shim`, the EXL3 libraries in `$HOME/exl3/{cpu,metal,strata_m4}` (the engine
`dlopen`s them from there), Strata in `$HOME/strata`, the engine build in `$HOME/strata-mac`, models in `$HOME/models/qwen38fn`.

## 0. Hardware

| part | what was used | pinned? |
|---|---|---|
| host | Mac mini M4 (`Mac16,10`), 10 cores (4 P + 6 E), 16 GiB, internal APFS NVMe (228 GiB volume, 96 % full at the time: budget ~190 GB, see section 4) | yes |
| GPU | RTX 3090 24 GB (sm_86) | yes |
| link | "chestnut" ASM2464 dock, USB 3.2 10 Gb/s to the mini, PCIe Gen3 x2 to the card (`LNKSTA 0x1023` measured 2026-10-10), external PSU for the card; USB ids `3801:0001` / `ADD1:0001` (used by `shim/pcie_cycle.py`) | dock model, bridge firmware (`ed4e39b` as recorded for the dock on 2026-10-10), PSU: **not pinned** |
| second Mac | any Apple-silicon Mac with Xcode's Metal toolchain (`xcrun -f metal`) to build `libexl3metal` and `strata-vision-metal`: the mini has only the Command Line Tools (no Metal compiler, cannot link ObjC) | the driver Mac here was an Apple-silicon MacBook Pro |

## 1. Versions

| component | version | source of the number |
|---|---|---|
| macOS | 26.6.2 (25G83) | `sw_vers` |
| Command Line Tools | 26.6.0.0.1781586589; Apple clang 21.0.0 (clang-2100.1.1.101) | `pkgutil`, `clang --version` |
| SDK used for every native build | `/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` (`xcrun --show-sdk-version` says 27.0 and **the 27.0 SDK breaks `ld`**: `tapi error: malformed file ... libSystem.B.tbd`) | `macbuild.py`, `exl3/cpu/build.sh`, tested |
| Python | 3.12.15, uv-managed `cpython-3.12.15-macos-aarch64-none`; uv 0.12.24 | `python -V`, `uv --version` |
| tinygrad | PR #17369 head `e765a2abf75c1f3a37e810f8856550397f2bcc23` (`bounty/nv-usb3-wip`, merge of master `7d2420e6488fed24afb5b9b45f4d983dec8c3fb8`), + `patches/tinygrad/0001-...patch`; `tinygrad-0.14.0` | `git rev-parse` in `~/tg-strata`; whether that commit is still fetchable from the PR **not pinned** |
| tinygrad venv | numpy 2.5.3, libusb1 3.4.0, libusb-package 1.0.30.0, ninja 1.13.2, pytest 9.1.1 (+ iniconfig, packaging, pluggy, pygments, importlib_resources) | `site-packages/*.dist-info` |
| `libamd_comgr.dylib` | 111,259,480 B in `~/.local/lib`, loaded by tinygrad through `COMGR_PATH` | **not pinned** (where it came from was not recorded) |
| Strata | v0.1.41, `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, `https://github.com/Niko1221/Strata` | `git rev-parse`, public tag checked |
| llama.cpp (ggml, mtmd) | `3cf03257f219afbe7334045ff7c6a06ac68c627d` (Strata's `FetchContent` pin) | Strata `CMakeLists.txt`, `git rev-parse` of the fetched tree |
| Strata Python env | `~/strata/.venv`, Python 3.12, Strata's own `requirements.txt` (numpy 2.5.3, jinja2 3.1.6, regex 2026.9.10, pyyaml 6.0.3, tqdm 4.70.1, requests 2.34.2, cmake 4.4.3, ninja 1.13.2, pillow 12.3.0, psutil 7.2.2, ...) | `.venv` dist-info matches `requirements.txt` |
| SIMDe (x86 intrinsics on arm64 for Strata's CPU kernels) | git `0ff5341927e1` at `~/simde`, no tag | `git rev-parse` (the commit's upstream URL **not pinned**; it is `simd-everywhere/simde`) |
| CUDA headers for clang (host side) | 12.8.93: `shim/cuda-redist-12.8.txt` (cuda_cudart 12.8.90, cuda_nvcc 12.8.93, cuda_cccl 12.8.90, cuda_profiler_api 12.8.90, `linux-sbsa` redist tarballs) merged into `~/cuda-12.8` | `~/cuda128.txt`, `version.json` |
| nvcc for the device side | 13.4.92: `shim/cuda-redist-13.4.txt` (cuda_nvcc, cuda_cudart, cuda_crt, libnvvm, cuda_cuobjdump, cuda_nvdisasm, cuda_profiler_api 13.4.92, cccl 13.3.4.3.1) merged into `~/cuda-13.4`; runs in the container (they are Linux arm64 binaries); the tarballs' sha256 are in `shim/cuda-redist-13.4.sha256` (same for 12.8: `shim/cuda-redist-12.8.sha256`, release 12.8.1) | `~/cuda13_paths.txt`, NVIDIA `redistrib_13.4.2.json` / `redistrib_12.8.1.json` |
| container (one-click) | root `Dockerfile`: `ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55` (linux/arm64), apt tools pinned by version (g++ 4:13.2.0-7ubuntu1, cmake 3.28.3-1build7, ninja-build 1.11.1-2, python3 3.12.3-0ubuntu2.1, git 1:2.43.0-1ubuntu7.3, ...), CUDA 13.4.92 from NVIDIA's redist tarballs pinned by sha256 (`shim/cuda-redist-13.4.sha256`, from `redistrib_13.4.2.json`), Strata by commit; built with Docker Desktop 29.1.3 on an Apple-silicon Mac | `Dockerfile`, `docker compose build` |
| container (mini, manual flow) | colima v0.10.3 (git `00f6c297e92a`), Docker 29.9.0 (build f415da8); image `strata-cuda:latest` = `sha256:431fed7e83df...`, built from `shim/docker/Dockerfile.strata-cuda` (ubuntu:24.04 + g++, make, cmake, ninja-build, python3, xz-utils, file, git, ca-certificates) | `docker images`; apt package versions inside the image and the VM's CPU/RAM settings **not pinned** |
| model | turboderp/Qwen3.8-Flash-Next-exl3, branch `3.05bpw_h5_ng5`, `69e33439ae950f17bcbe95c98f117d80f759ab6d` | `pack_td.py`, HF refs API 2026-10-12 |
| dense part | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, `IQ3_S/` two shards + `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | see section 4; **revision not pinned** |
| MTP draft | Qwen/Qwen3.8-Flash-Next `de4b8e4d43b917e7706784d8bb445c9af86a3540` | `results/provenance/mtp-inventory.json` |

## 2. Toolchain on the mini

1. `xcode-select --install` (Command Line Tools). The CLT bundle here has the SDKs `MacOSX26.5.sdk`, `26.sdk`, `27.0.sdk`, `27.sdk`, `15.4.sdk`, `15.sdk`; the build files name `MacOSX26.5.sdk`.
2. `uv` (0.12.24), `colima` + `docker` into `~/.local/bin` (how they were installed was not recorded). `colima start` with the default `$HOME` mount: the build
   scripts run `docker run -v $HOME:$HOME -w $PWD strata-cuda:latest ...` so inside and outside paths are identical.
3. CUDA toolkits (no CUDA driver or GPU library is installed on macOS): download the redist tarballs listed in `shim/cuda-redist-12.8.txt` and
   `shim/cuda-redist-13.4.txt` from `https://developer.download.nvidia.com/compute/cuda/redist/<path>` (the URL pattern was checked on 2026-10-12: HTTP 200),
   extract each and merge its `bin/ include/ lib/ nvvm/ ...` into `~/cuda-12.8` and `~/cuda-13.4`. Then
   `cp shim/include/cublas_v2.h ~/cuda-13.4/include/` (the shim's minimal cuBLAS header, byte-identical to the file on the mini; `exl3/build.sh` compiles against it).
   The merge procedure itself was not recorded; the resulting directory layouts are `bin dl include lib lib64 nvvm pkg-config` (13.4) and `bin dl include lib nvvm pkg-config` (12.8) [not re-run].
4. `docker build -t strata-cuda:latest --build-arg CUDA_BIN=$HOME/cuda-13.4/bin -f shim/docker/Dockerfile.strata-cuda shim/docker` (re-run with a throw-away tag: builds).
5. SIMDe: `git clone https://github.com/simd-everywhere/simde ~/simde && git -C ~/simde checkout 0ff5341927e15df0c3da8bb13bb32ec7f95aef84` [not re-run].

## 3. tinygrad, the shim, Strata

```sh
# tinygrad (PR #17369 head, see versions) and its venv
git clone https://github.com/tinygrad/tinygrad ~/tg-strata && cd ~/tg-strata
git fetch origin pull/17369/head && git checkout e765a2abf75c1f3a37e810f8856550397f2bcc23   # if the PR head moved, merge master 7d2420e into it [not re-run]
git apply <repo>/patches/tinygrad/0001-tg-strata-nv-usb-fixes.patch
uv venv --python 3.12.15 ~/tinygrad/.venv
uv pip install --python ~/tinygrad/.venv numpy==2.5.3 libusb1==3.4.0 libusb-package==1.0.30.0 ninja==1.13.2 pytest==9.1.1
```

`~/.chestnut_env` is sourced by `shim/run-strata.sh`; its three lines (paths only) are in `shim/env.example`.

```sh
# the shim: sources, prebuilt blobs, libcudart.dylib  (copies into ~/shim: do not run on a mini that is serving)
zsh <repo>/shim/install.sh
```

`install.sh` copies `tgcu.py tgcu_api.py usb_drain.py pcie_cycle.py tgcudart.cpp tgcublas.cpp tgcuda_launch.h run-strata.sh srv.sh ls_srv.sh macbuild.py ...` and
`shim/prebuilt/{tgcu_builtin.fatbin,tgcu_libc.cubin}` to `~/shim`, then builds `libcudart.dylib` (`shim/build-libcudart.sh`: clang++ -O2, CUDA 12.8 headers, CPython 3.12.15
from uv, `-install_name @rpath/libcudart.dylib`; compiles and exports the same 136 symbols as the deployed one, flags of the deployed build were not recorded).
Every engine start runs `pcie_cycle.py` and `usb_drain.py` (power-cycles the card over the bridge and drains stale USB state) before tinygrad opens the device;
`TGCU_NO_USB_RESET=1` skips that.

```sh
# Strata
git clone https://github.com/Niko1221/Strata ~/strata && cd ~/strata && git checkout v0.1.41      # fb58e0dbc8399662c0e47c76578c6e878b14f6cf
for p in <repo>/patches/strata/0*.patch; do git apply --check $p && git apply $p; done              # in order; scripts/verify-card-free.sh patches
uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
```

## 4. Models and packs (about 190 GB on disk)

```sh
mkdir -p ~/models/qwen38fn && cd ~/models/qwen38fn
H=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main
for f in IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf mmproj-Qwen3.8-Flash-Next-BF16.gguf; do curl -L -C - -o $(basename $f) $H/$f; done
```

Sizes on disk (matched against the Hugging Face file listing on 2026-10-12; `main` was at `ed59f92082b1` last modified 2026-09-29, before the files were downloaded
2026-10-09/10): shard 1 54,817,524,224 B (LFS sha256 `4c1eb2ce...`), shard 2 28,800,138,432 B (`316b46f3...`), mmproj 907,543,008 B (`b1a82259...`). The local files' sha256
were **not** compared (hashing 83 GB would have evicted the live server's page cache), so the revision is size-matched, not hash-proven. The GGUFs are `qwen4exp`, 48 blocks, 512 experts, general.quantized_by ISTA DASLab.

```sh
# dense part -> a 1.5 GB pack + tokenizer (Strata's tools; gguf-py from the pinned llama.cpp)
STRATA_GGUF_PY=<llama.cpp checkout>/gguf-py ~/strata/.venv/bin/python ~/strata/tools/iq_pack.py \
  --gguf ~/models/qwen38fn/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --out ~/models/qwen38fn/packs/iq3_s
# MTP draft layer, from the BF16 checkpoint (only the 31 MTP tensors, 5.2 GB, range-fetched)
cd ~/strata && .venv/bin/python tools/mtp_fetch.py fetch --out ~/models/qwen38fn/mtp
.venv/bin/python tools/mtp_pack.py --src ~/models/qwen38fn/mtp --experts q2_0 --out ~/models/qwen38fn/mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf ~/models/qwen38fn/mtp/mtp-q2_0.gguf --out ~/models/qwen38fn/mtp/rt
cp data/draft_vocab.bin ~/models/qwen38fn/mtp/rt/draft_vocab.bin
```

These are the commands Strata's `setup.py` runs (and `docs/ORCA.md` lists); the logs of the original run are `results/provenance/{pack,mtp}.log`
(`pack.log`: 1,079 tensors, 302 served natively, arena 1.43 GiB, vocab 248,320). The MTP tensors fetched are listed with sizes and sha256 in
`results/provenance/mtp-manifest.json`. Expected outputs on the mini: `packs/iq3_s/dense.bin` 1,538,035,200 B, `mtp/rt/dense.bin` 116,099,072 B, `experts.bin` 707,788,800 B,
`draft_vocab.bin` 425,196 B (identical to Strata's `data/draft_vocab.bin`, sha256 `b1e1d3a7...`). These Strata tool runs were **not re-run** for this repo.

**The EXL3 experts** (45.77 GB), from the pinned Hugging Face revision by HTTP Range requests (no shard is stored), with the sha256 manifest verified:

```sh
mkdir -p ~/exl3 && cp -R <repo>/exl3/. ~/exl3/ && cd ~/exl3
zsh cpu/build.sh                                  # libexl3cpu.dylib: pack_td.py sync checks the NEON forward against the reference
python pack_td.py index      # 7 shard headers -> td/experts_index.json (26.8 MB; regenerates byte-identically)
python pack_td.py sync       # all 48 layers: fetch, check fields, write pack/experts.bin, NEON-vs-reference forward, section sha256 -> pack/exl3_pack.json
python pack_td.py verify     # every layer: 6 experts re-fetched and byte-compared, section sha256 vs the manifest -> "VERIFY OK"
python pack_exl3.py marker   # pack/exl3_experts.txt: field offsets + per-layer sha256 (Strata refuses the pack unless all 48 layers are listed)
diff pack/exl3_pack.json <repo>/results/pack/exl3_pack.json    # the `written` timestamps differ; the sha256 values must not
```

`sync` took ~22 s per layer (0.95 GB each) on the mini on 2026-10-11 (`results/pack/td_sync.log`); a single layer fetched into a temporary pack on 2026-10-12 took
76-120 s and its sha256 matched the manifest (layer 24, `80eb9002da51...`). The marker regenerated from `results/pack/exl3_pack.json` is byte-identical to `exl3/pack_full512/exl3_experts.txt`.
`python pack_td.py verify L [L...]` checks selected layers only. Use `~/strata/.venv/bin/python` (numpy 2.5.3) for these scripts.

Then assemble the pack directory the engine's `--pack` expects, and the profile:

```sh
mkdir -p ~/exl3/pack_full512 && cd ~/exl3/pack_full512
ln -sf ~/models/qwen38fn/packs/iq3_s/{conversions.json,dense.bin,index.txt,tokenizer} .
ln -sf ~/exl3/pack/experts.bin experts.bin && cp ~/exl3/pack/exl3_experts.txt .
mkdir -p ~/exl3/placement/f512c && cp <repo>/exl3/placement/f512c/profile.bin ~/exl3/placement/f512c/   # sha256 6e3653c1ea9a...
```

`profile.bin` (all 24,576 `(layer, expert)` pairs ranked hottest first) is shipped; `exl3/make_profile512_freq.py` regenerates it byte for byte from
`exl3/placement/inputs/{expert-profile-p256.bin,pack_p256_exl3_experts.txt}` plus Strata's `tools/make_profile.py` (the P256 inputs come from a routing
calibration of the earlier pruned Q2_0 model of this campaign and cannot be regenerated from public data).

## 5. Native libraries and the engine

```sh
# NEON tier + EXL3 CUDA kernels (fatbins in the container) + their host library
zsh ~/exl3/cpu/build.sh ~/exl3/cpu                        # libexl3cpu.dylib; byte-identical to the deployed one
cd ~/exl3 && for m in exl3_strata exl3_strata_recon exl3_strata_pf; do zsh build.sh $m; done     # nvcc 13.4, sm_86 fatbins
(cd strata_m4 && zsh build.sh)                            # libstrata_exl3.dylib (links ~/shim/libcudart.dylib)
# on the Mac with the Metal toolchain; then copy the dylib to the mini
bash exl3/metal/build.sh && scp exl3/metal/libexl3metal.dylib mini:exl3/metal/            # byte-identical to the deployed one
# vision encoder (Metal), from the patched Strata checkout
git -C ~/strata apply <repo>/patches/vision/0001-strata-vision-metal-macos.patch
git -C ~ clone https://github.com/ggml-org/llama.cpp llama.cpp && git -C llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
LLAMA=$HOME/llama.cpp bash <repo>/vision/build_vision_mac.sh            # -> ~/strata-mac-vision/strata-vision-metal
```

Tests that need no card (see `scripts/verify-card-free.sh`): `exl3/test_exl3_cpu.py` (106/106), `exl3/metal/test_exl3_metal.py` (all pass).
The tests that need the card are in `exl3/strata_m4/test_exl3s*.py`.

**The engine** [not re-run]. Strata's CUDA build is configured and its compile commands extracted in the Linux container, then rewritten into two ninja files:
`device.ninja` (nvcc -> sm_86 fatbins, run in the container) and `host.ninja` (clang `--cuda-host-only` objects that embed those fatbins, x86 intrinsics through
SIMDe, linked against the shim's `libcudart`), by `shim/macbuild.py`.

```sh
export PATH=~/.local/bin:$PATH
docker run --rm -v $HOME:$HOME -w $HOME -e H=$HOME strata-cuda:latest sh -c 'cmake -S $H/strata -B $H/strata-linux -G Ninja \
   -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=$H/cuda-13.4/bin/nvcc \
   && cd $H/strata-linux && ninja -t commands strata > strata_cmds.txt'      # the options are those of the deployed strata-linux/CMakeCache.txt; docker/build-engine.sh runs this configure (+ -DSTRATA_BUILD_TESTS=OFF -DCUDAToolkit_ROOT=$H/cuda-13.4, both read from that cache) and produced 63 nvcc rules, as the deployed one
python3 ~/shim/macbuild.py                                                       # writes ~/strata-mac/{device,host}.ninja (63 nvcc rules, 135 host rules)
docker run --rm -v $HOME:$HOME -w $HOME/strata-mac strata-cuda:latest ninja -f device.ninja
~/strata/.venv/bin/ninja -C ~/strata-mac -f host.ninja                           # -> ~/strata-mac/strata
```

`macbuild.py` hard-codes `SDK=.../MacOSX26.5.sdk`, `CUDA_HOST=~/cuda-12.8`, `CUDA_DEV=~/cuda-13.4`, `SHIM=~/shim` and reads `~/simde`. The Strata CMake configure fetches
llama.cpp at the pin above into `~/strata-linux/_deps/strata_llamacpp-src`, which the Metal vision build reuses. The shipped engine was built incrementally
(`cmd_*.sh` re-runs of single objects + relink in a copy of the build dir) from this same source tree; that is why patch `0003`/`0004` equal the tree and not a separate build recipe.

## 6. Run

```sh
zsh <repo>/configs/render.sh <repo>/configs/strata-exl3-full512-final-g4-lru.json.tmpl > ~/shim/strata-exl3-full512-final-g4-lru.json
zsh ~/shim/ls_srv.sh ~/shim/strata-exl3-full512-final-g4-lru.json      # stops any server + engine, starts this one, waits for "ready", prints cache/VRAM lines
# which is: cd ~/strata && STRATA_PREFILL_CPU_SHARE=1 .venv/bin/python serve/server.py --engine strata --config <cfg> --port 8080
```

Expected log lines (from `results/raw/lru/final_lru.log`): `expert cache auto: 15.36 GiB free, 400 MiB reserved (+218 MiB for the draft head) -> 7400 slots`,
`pre-filled 7400 of 7400 slots from the profile in 64.1 s`, `exl3-metal: lane ready (... 5600 slots of 1867776 B registered)`, `LRU preloaded with 5600 of 5600 slots by profile rank in 4.8 s`,
`716 MiB of VRAM free with everything loaded`, `READY`. Startup to ready took about 200 s in the earlier campaign notes (not timed in the final run). Stop it with `kill -TERM $(cat ~/shim/server.pid)` and wait until
`pgrep -x strata` is empty before starting another: two engines fight over the USB device. If the dock reports LTSSM other than `0x78` run `~/tinygrad/.venv/bin/python ~/shim/pcie_cycle.py` (with `PYTHONPATH=~/tg-strata`).

## 7. Measure

All scripts post to `http://127.0.0.1:8080/v1/chat/completions` and use the system `python3`; `final_vision.py` needs Pillow (use `~/strata/.venv/bin/python`). The shell drivers (`final_lru.sh`, `dd.sh`, `long_run.sh`, ...) call
`~/shim/ls_srv.sh`, `~/shim/split_probe.py`, `~/shim/ls_pf.py`, `~/shim/final_long.py` by those paths, so `cp bench/* ~/shim/` first (the shim installer does not copy the benchmarks); `final_long.py` imports Strata's `tools/needle_bench.py` (`bench/needle_bench.py` is a copy).

```sh
python3 bench/split_probe.py NAME 300             # the 3 decode prompts (run twice right after a restart; section 1.1 of results.md)
python3 bench/ls_pf.py 10000                      # prefill row: CHARS of docs text, random first line (3,783 tokens); 24000 -> 7,905; 25000 -> 8,218; 55000 -> 17,610
python3 bench/final_long.py needle 32000 50       # needle at TOKENS depth%; 100000 25; 190000 50 (8 minutes of prefill)
python3 bench/final_vision.py                     # NYT headline + bar chart (needs ~/strata-mac-vision/test-1.jpeg: the NYT 'MEN WALK ON MOON' photo, not in this repo)
bash bench/final_lru.sh                           # the whole sequence that produced results/raw/lru/final_lru.log (restart, decode x2, prefill rows, needle, vision)
```

`STRATA_DECODE_TIMING=1` in the config env adds the per-window breakdown to the engine log (`bench/dd.sh NAME CFG` runs restart + decode x2 + the timing lines). The test image
`test-1.jpeg` and the generated `chart.png` are not in this repo (the photo is not ours to redistribute); `bench/final_vision.py` builds the chart itself.

## 8. What cannot be rebuilt from this repo alone

- the dock/PSU model and firmware, the `libamd_comgr.dylib` source (and whether the NV backend needs it at all), colima/docker installation on the mini, SIMDe's upstream URL (pinned by commit);
- the exact flags of the deployed `libcudart.dylib` (the rebuilt one exports the same symbols; its bytes differ). Recovered since the first version of this file: `tgcu_libc.cubin` is `ptxas -arch=sm_86 -c tgcu_libc.ptx`
  and `tgcu_builtin.fatbin` is `nvcc -std=c++17 -O3 -Xptxas -v --fatbin -arch=sm_86 -no-compress` (both byte-identical, section 10); `exl3_strata.fatbin` is byte-identical when built at `/Users/sero/exl3` (the id is path-dependent);
  the Strata CMake options come from the deployed `CMakeCache.txt`;
- the full engine rebuild: the device half (63 fatbins) is now reproduced in the container (61 of 63 byte-identical to the deployed ones, section 10); the host half was built on a driver Mac, not on the mini, and the resulting `strata` binary was not run;
- the deployed GGUF files themselves were never hashed (size-matched only); `docker/models.sha256` pins Hugging Face revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` with the LFS sha256 the API reports for it (sizes equal the deployed files'). The routing calibration behind `profile.bin` (the file is shipped);
- the decode and prefill numbers need the card: nothing in the card-free checks measures them.

## 10. One-click: what was run

The flow is `docker compose run --rm build` (or `make`) -> `./out`, optionally `docker compose --profile fetch run --rm fetch all`, then `./run.sh` on the Mac. Files: `Dockerfile`, `docker-compose.yml`, `Makefile`, `docker/`
(`build-device.sh`, `build-engine.sh`, `fetch.sh`, `models.sha256`), `run.sh`. Why Docker cannot run the server: the engine drives the 3090 through libusb (tinygrad NV) and computes cold experts with Metal; Docker on macOS is a Linux VM
with neither Metal nor USB passthrough of this dock. Everything below was run on the driver Mac (MacBook Pro M1 Max (MacBookPro18,2), macOS 26.5, Docker Desktop 29.1.3 / linux/arm64, 10 CPUs, 8 GiB VM), 2026-10-12; no card, no dock, and the mini was only
read (`ssh` + `shasum` of deployed files, no load, nothing written).

| stage | status | evidence |
|---|---|---|
| `docker compose build --no-cache build` (apt pins, 8 CUDA 13.4.92 tarballs sha256-checked, Strata clone + 5 patches + vision patch, CMake configure, nvcc) | **run**, cold, 3 min 4 s | image `qwen38fn-build:local` |
| `docker compose run --rm build` / `make build` | **run**, 12 s / 6 s (cached) | `./out`: 6,191 files, `SHA256SUMS`, `BUILD_INFO.txt` |
| EXL3 fatbins `exl3_strata`, `_recon`, `_pf`; `tgcu_builtin.fatbin`; `tgcu_libc.cubin` | **run in the container**; all 5 **byte-identical to the deployed files** (the build fails on a mismatch); identical in two builds | `out/fatbins/SHA256SUMS`: `d19f0e17...` `bb7099ba...` `8baedf9c...` `d4139235...` `e36a0f87...` |
| Strata engine device code (CMake configure + 63 nvcc rules -> 63 sm_86 fatbins, 60 s with 6 jobs) | **run in the container**; configure options and `device.ninja` identical to the deployed ones (63 rules); **59 of 63 fatbins byte-identical to the deployed ones in every build (four builds compared)**; 4 (`fused_gr`, `sampler`, `moe_mmq`, `kernels`) differ from build to build, also between two `-j 1` builds in separate containers: nvcc puts a per-compilation id into the names of anonymous-namespace kernels (`_GLOBAL__N__<hash>_..._<hash>_<line>`); for `moe_mmq` the SASS differs only in those names, for `sampler` the instruction count (16,416) and register usage are equal but register allocation/scheduling differ (329 instruction lines) | `out/engine/strata-mac-fatbins.sha256`, deployed list read from `~/strata-mac` on the mini |
| `pack_td.py index/sync --no-neon --expect/verify/compare` in the fetch container, `EXPERT_LAYERS=24` | **run for one layer**: 0.95 GB fetched in 50 s, section sha256 `80eb9002da511ea3...` = manifest, `VERIFY OK`, `COMPARE OK` | `docker compose --profile fetch run --rm fetch experts` |
| the same for all 48 layers (45.8 GB) | **not run** | |
| `fetch dense`: size + sha256 against the Hugging Face LFS ids (revision `ed59f92082b1`) | **run for the mmproj only** (`DENSE_MATCH=mmproj`): sha256 `b1a82259...` OK, and a deliberately wrong hash makes it exit 1 (`.bad` file kept). The two GGUF shards (54.8 + 28.8 GB) were **not** hashed (a first run downloaded 28.7 GB of shard 1 before I stopped it; the resume with `curl -C -` worked) | |
| `./run.sh --check` from an empty install root (`RUN_ROOT=/tmp/...`), Metal toolchain hidden to exercise the `./out/metal` path | **run**: 75 s, **53 ok, 13 warnings, 0 failures**, exit 0; a second run is a no-op (everything "present"/"up to date") | warnings: SDK 26.2 instead of 26.5, Python 3.12.13 instead of 3.12.15 (uv), no Metal compiler (used `./out/metal`), no USB dock, `libamd_comgr.dylib` absent, `libexl3cpu` sha differs from the deployed one (see below), curand stand-in header, models/packs absent (6 config paths missing) |
| `./run.sh --build-metal` | **run** (here: 54 s): `libexl3metal.dylib` sha256 `e38a011c...` = the deployed one; `strata-vision-metal` 8,094,368 B | `out/metal/SHA256SUMS` |
| engine host build (`macbuild.py` + `host.ninja`, 135 rules, link) | **run** on this Mac with Homebrew LLVM 22.1.8 (`CLANG_BIN`); with this Mac's Apple clang 17 it **fails** (`__ldcs`). The binary links `@rpath/libcudart.dylib`, rpath = install root. **Never run** | |
| `./run.sh` without `--check`: starting the server | **not run** (no dock/card here, the mini is serving). The pieces it uses were smoke-tested with the mock engine: `--host`, `--port`, `STRATA_API_KEY` from the environment (401 without the key, 200 with it, the key is not in the log) | |
| `iq_pack.py`, `mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py` (`run.sh` `build_packs`) | **not run** (need the 55 GB shard and a 5.2 GB range fetch); they are the commands of section 4 | |
| `scripts/test-gates.sh` | **run**: ALL GATES OK (7 checks, including a wrong hash and a corrupted byte) | |

Findings while doing this: (1) `exl3_strata.fatbin` is path-dependent (built at `/Users/sero/exl3` it equals the deployed one; at `/build/exl3` it does not), hence `BUILD_ROOT=/Users/sero` in the Dockerfile; (2) `tgcu_builtin.fatbin` and
`tgcu_libc.cubin` are reproduced byte for byte (flags in section 8), the earlier "path effect" note was wrong: the shipped one was built with `-Xptxas -v`; (3) the deployed `libexl3cpu.dylib` (`9fe9cab5...`) was rebuilt here as `d874f168...`
with SDK 26.2 and Apple clang 17.0.0 (the mini: SDK 26.5, Apple clang 21.0.0); which of the two differences causes it was not isolated; (4) the mini's CLT has no Metal compiler, so `libexl3metal` and `strata-vision-metal` come from
`./run.sh --build-metal` on another Mac, as before; (5) the deployed `libamd_comgr.dylib` stays unexplained: the tinygrad NV backend imports without it here, but whether the engine needs it at run time is unknown;
(6) the repo had lagged behind the measured configuration: the host-tail step (`patches/strata/0005`, `shim/tgcu*.py`, `exl3/metal`, `configs/...-ht.json.tmpl`) was added from the campaign's diffs and verified byte-identical to the deployed files.
