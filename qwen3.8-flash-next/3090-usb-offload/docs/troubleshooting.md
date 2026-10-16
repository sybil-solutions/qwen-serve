# Troubleshooting

Every entry below was hit during the campaign; the source is the plan entry (`results/campaign/EXL3_PLAN-log.md`, date in UTC) or a file in this repo.

## Link and device

**Engine fails at the first GPU call, or shows stale/shifted USB reads.** A process killed mid-transfer leaves state in the ASM2464 (bulk IN overflows).
The shim runs `pcie_cycle.py` + `usb_drain.py` before every start (`shim/tgcu_api.py: init`); `TGCU_NO_USB_RESET=1` turns that off. If the dock reports
LTSSM other than `0x78` (for example after swapping the card): `PYTHONPATH=~/tg-strata ~/tinygrad/.venv/bin/python ~/shim/pcie_cycle.py` (GOAL.md, Link).

**Two engines running.** They fight over the USB device. `srv.sh` / `ls_srv.sh` send `SIGTERM` to the server, wait until no `strata` process is left, then start. Do it
by hand the same way (`kill -TERM $(cat ~/shim/server.pid)`; wait for `pgrep -x strata` to be empty).

**GPU boot reads garbage after a killed session** (`PROM` window read while the previous GSP is still winding down). Fixed by the tinygrad patch (64 KB pieces, accepted
once two consecutive reads agree), `patches/tinygrad/0001-tg-strata-nv-usb-fixes.patch`.

**`async copy timeout`, `GPU timeout waiting for semaphore`, 10 s hangs under G=4 or at ~100K tokens.** Two separate root causes (plan 2026-10-12T05:25Z):
(1) `tgcu.Context._transient` waited for a semaphore value that already counted the push buffer being submitted when the ring wrapped: a self-deadlock, fixed
in `shim/tgcu.py` (`shim/tests/test_wrap.py`: the old code fails, the new one runs 20,000 flushes); (2) with `STRATA_PF_GROUP=4`, at the periodic checkpoint
(deterministic at 100K tokens) a fresh 256 MiB tgcu chunk memset hangs when VRAM headroom is too small: use `--vram-reserve-mib 400` (7,400 slots).
`TGCU_HANG_DUMP=1 TGCU_WAIT_S=30` dumps every thread and a native `sample` 8 s into a GPU wait.

## Memory

**`Insufficient Memory` command-buffer failure at layer 4 of the first prompt.** Metal's `recommendedMaxWorkingSetSize` on this mini is 11.84 GiB
(`results/raw/lru/ws.m` prints it). 5,600 LRU slots (10.46 GB) plus separate per-prompt prefill sets (3 x 400 slots = 2.24 GB) exceeded it and the engine had to be restarted. The prefill
sets are now borrowed from the coldest LRU slots (`STRATA_PF_BORROW_LRU`, default on). If you raise `STRATA_EXL3_METAL_SLOTS` past ~5,600 expect this error; more needs `iogpu.wired_limit_mb`
(a system setting, not changed here).

**`no VRAM is left for the expert cache` at start.** The engine sizes the expert cache after the session; verify/draft/prompt buffers need the ~600 MiB that must stay free.
Lower `--max-context`, use `--kv q4_0`, or drop the vision key (GOAL.md, Vision + 200K plan).

**A 2.5K-token prefill kills the engine (`no progress for 60 s (reading the prompt (batched): layer 0 ...)`, then `GPU timeout waiting for semaphore`)** on configs that let the prompt path borrow VRAM expert
slots. Use `--no-prefill-borrow` (every config here does; plan 2026-10-11T18:35Z).

**Decode slower after the first image.** The Metal encoder holds ~1.1 GB wired RAM while loaded and the file cache / LRU need it: `idle_unload_s: 120` in the `vision` key
(server patch 0002) ends it after 120 idle seconds. Resident encoder: decode 8.2/12.0/9.0 vs 9.0/12.3/9.4 on the config of that day.

**Disk full.** The mini's 228 GiB volume was at 96 %. The experts (45.8 GB), the two GGUF shards (83.6 GB), the dense pack (1.5 GB), the MTP files (1.7 GB), the mmproj (0.9 GB),
and `td/experts_index.json` (27 MB) are what you need (plus the engine log, which reached 33 MB); `pack_td.py sync` into a temporary pack needs the same space again.

## Build

**`ld: tapi error: malformed file ... MacOSX27.0.sdk/usr/lib/libSystem.B.tbd`.** The default SDK is 27.0; pass `-isysroot .../MacOSX26.5.sdk` (all build scripts do).

**The mini cannot build `libexl3metal` or `strata-vision-metal`** (no `metal` compiler, cannot link ObjC): build on a Mac with Xcode's Metal toolchain and copy the dylib / binary.

**`libexl3cpu.dylib` differs from the deployed one.** clang takes the install name from the `-o` argument: build with a relative output name (`exl3/cpu/build.sh` does).

**`docker run ... nvcc` finds no files.** The VM mounts `$HOME` only; run builds from a directory under `$HOME`, not `/tmp`. `exl3/build.sh` removes the old fatbin first, so a failed nvcc run (for example colima stopped) cannot leave a stale one that looks fresh.

**Strata CUDA kernels fail with `cublas_v2.h` errors.** `~/cuda-13.4/include/cublas_v2.h` must be the shim's minimal header (`shim/include/cublas_v2.h`).

**Docker build stops at apt: `Version '...' for '...' was not found`.** The Dockerfile pins the apt tools by version and Ubuntu removes superseded versions from its archive. Read the current ones
(`docker run --rm ubuntu:24.04@sha256:534baea6... sh -c 'apt-get update -qq; apt-cache policy g++ make cmake ninja-build python3 xz-utils file git ca-certificates curl'`), update the pins in the Dockerfile (`toolchain` and
`fetcher` stages), rebuild, and compare the fatbin hashes the build prints against the deployed ones (it fails on a mismatch).

**`docker compose run --rm build` is very slow or `exec format error` on a PC.** The image is `linux/arm64` (NVIDIA's `linux-sbsa` redist tarballs): native on Apple silicon, emulated elsewhere (QEMU/binfmt needed, several times slower).

**`./run.sh`: `host.ninja failed` with `no matching function for call to '__ldcs'`.** The engine's CUDA host compile was done with Apple clang 21 (CLT 26.6, the mini's); older clangs (17 here) lack the overloads. Point `CLANG_BIN` at an LLVM >= 21
(`CLANG_BIN=/opt/homebrew/opt/llvm@22/bin ./run.sh`); only the engine and `libstrata_exl3` use it. With clang <= 17, `run.sh` also writes an empty `curand_mtgp32_kernel.h` into `~/cuda-12.8/include` (the old CUDA wrapper force-includes it).

**`./run.sh`: `no Metal compiler`.** The Command Line Tools have no `metal`: run `./run.sh --build-metal` on a Mac that does (Xcode, or `xcodebuild -downloadComponent MetalToolchain`) and copy `./out/metal` to the mini's `./out/metal`.

**`./run.sh`: `... differs from the bundle's copy: kept`.** `run.sh` never overwrites an installed file that differs from what it would install (the mini's live `~/shim` is such a case); `--force` replaces, otherwise the difference is only reported.

## Server and benchmarks

**`bench/final_vision.py`: `No module named 'PIL'`.** Run it with `~/strata/.venv/bin/python` (pillow is in Strata's requirements). The `final_lru` log contains exactly this error.

**Needle request returns HTTP 400/503.** `lr_g4_100k.log` (HTTP 400) and `pf500_final.log` (HTTP 503 on the 32K needle) are failed attempts kept as they ran: the 400 run was on a config (`g4b`, `--vram-reserve-mib 100`) that was then replaced; the 100K number comes from `lr_g4c_100k.log`. The causes of the 400 and the 503 were
not investigated.

**Numbers jump between restarts.** Decode right after a restart depends on the OS file cache and LRU contents: run each probe twice (the scripts do), compare p2 first (it is the least
SSD-bound), and keep the encoder unloaded. The first prompt after a restart reads its misses from NVMe (p1: 7,598 and 8,023 MiB of page-ins in the two `final_lru` runs).

**`STRATA_MAC_WARM=1` crashes the engine** (`bulk IN short read: 139784/147968 bytes` in `hs_wait`). Root-caused and fixed (the look-ahead thread's first touch of lazily downloaded router
pages ran a USB device-to-host copy concurrently with the armed pull) but it is a measured no-gain option and stays off.
