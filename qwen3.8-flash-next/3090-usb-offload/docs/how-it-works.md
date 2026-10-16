# How it works

Qwen3.8-Flash-Next (`qwen4exp`): 48 layers, 36 gated-delta-net (GDN) layers + 12 full-attention (QSA) layers, hidden 2560, 512 routed experts per
layer with 10 used per token, expert FFN 640, one shared expert, hyper-connections, an MTP draft layer. The routed experts are 45.77 GB at 3.05 bpw
(24,576 experts x 1,862,400 B). The rest (attention, GDN, shared experts, norms, hyper-connections: the "dense part") comes from the IQ3_S GSQ-RCO GGUF
(83.6 GB on disk, its own experts unused): the engine reads the dense tensors from a 1.5 GB pack made by Strata's `iq_pack.py` and the per-layer-embedding
table from the second shard (`--ple-gguf`). A 24 GB card cannot hold the experts and the 16 GB mini cannot either, so they are split three ways.

```text
  RTX 3090 (24 GB), behind the USB dock                    Mac mini M4 (16 GiB)
  +-------------------------------------------+            +-------------------------------------------------+
  | dense part, attention + GDN, MTP head      |            | Metal lane (libexl3metal): LRU of 5,600 mlocked, |
  | 200K k8v4 KV cache                         |  x, ids, w | registered expert slots (10.46 GB) + 0.2 GiB    |
  | 7,400 expert slots (12.84 GiB), hottest    | ---------> | static tier; computes every expert that is not   |
  | experts by profile rank; exl3_moe_coop     |   USB 3.2  | resident in VRAM                                 |
  | kernels compute them                       | <--------- | NEON pool (libexl3cpu) = fallback + exact path   |
  +-------------------------------------------+     y     +----------------------^--------------------------+
                                                                                 | F_NOCACHE preads, ~2.4 GB/s
                                                          +----------------------+--------------------------+
                                                          | NVMe: experts.bin, all 24,576 experts, one 1.86 MB |
                                                          | blob each (sha256 per layer in the manifest)       |
                                                          +-----------------------------------------------------+
```

Of the 24,576 experts, 7,400 (30 %) sit in VRAM, 5,600 (23 %) in the Mac's LRU, the other 11,576 only on NVMe. Nothing is pruned or masked: a router
pick that is in neither VRAM nor the LRU is read from NVMe and computed on the Mac. The static ranking that fills VRAM and primes the LRU is
`exl3/placement/f512c/profile.bin` (routing frequency from the earlier P256 calibration, mapped onto the 512-expert ids; `exl3/make_profile512_freq.py`).

## The link

The dock is an ASM2464 bridge: USB 3.2 10 Gb/s outside, PCIe Gen3 x2 to the card (`LNKSTA 0x1023`). Measured on this rig: ~190 MB/s with the stock copy
path; 353 MB/s host-to-device and 328 MB/s device-to-host with the pipelined transfers in `shim/tgcu.py` (one doorbell per message, two SRAM
regions); the link is half duplex (a download and an upload cannot overlap: 104.0 MB/s combined vs 104.2 sequential, `results/raw/duplex/`).
A single USB transaction round costs ~0.4 ms, which is why the design uses as few rings as possible per layer.

There is no CUDA driver for macOS. `shim/` is one: tinygrad's NV backend boots the GPU over the USB bridge (GSP firmware, VA space, channels);
`tgcu.py` loads nvcc-built cubins, builds QMDs and chains launches into batches on tinygrad's compute ring; `tgcu_api.py` is the Python half of
`libcudart.dylib` (`tgcudart.cpp`, a drop-in for the CUDA runtime/driver API that clang `--cuda-host-only` objects link against) and a cuBLAS subset
(`tgcublas.cpp`, WMMA GEMM kernels in `tgcu_builtin.cu`). Strata's CUDA sources compile on macOS in two halves: `nvcc` (13.4, in a Linux container) makes
sm_86 fatbins, `clang --cuda-host-only` makes the host objects that embed them (`shim/macbuild.py`). CUDA graphs are captured launch lists.

## Decode: one ring per layer

Decode runs in verify windows (MTP drafts up to 4 tokens, the target verifies them in one pass: 1.8-3.3 accepted tokens per window on the three probe
prompts). Per window and per layer, the GPU computes attention/GDN and the router, then runs the resident experts; the experts that are not in VRAM
are the "cold" ones:

1. **Pre-armed pull** (`STRATA_HS_PRE`, `STRATA_HS_DEFER`): a copy-engine job armed before the ring and gated on the ring word fetches the doorbell payload
   (x rows, expert ids, weights) to the host the moment the GPU writes it. One pull and one gated push per ring; the next ring's pull is queued behind the push.
2. **Early start** (`STRATA_EXL3_METAL_EARLY`, `STRATA_MAC_PULL_EARLY`): once ids are on the host, the lane (a) submits the cold experts that are already in
   place (static tier + LRU hits) as one async Metal job and (b) reads the misses from NVMe straight into free LRU slots with `F_NOCACHE` preads
   (`STRATA_MAC_PULL_NOCACHE`), overlapping both with the queueing of the push.
3. **Metal job for the misses** (`STRATA_EXL3_METAL_SPLIT`): a second job; `y = A + B` in fp32.
4. **Push and flag**: the fp32 rows and the "landed" flag go back in one gated copy; the GPU waits on the flag and adds the rows to the residual.

The Metal library computes an expert exactly like the CUDA kernels do (trellis decode, Hadamard, GEMV, SwiGLU, Hadamard, GEMV with the per-tensor
scales; fp32 accumulation) and is gated against the NEON code and a float64 reference (`exl3/metal/test_exl3_metal.py`).

**The LRU.** Metal reads a blob in place only if it is registered (`newBufferWithBytesNoCopy`), and registering file-backed pages wires them, so the
cold set lives in `STRATA_EXL3_METAL_SLOTS` mlocked, registered slots: an O(1) doubly-linked recency list over `(layer, expert)` -> slot, preloaded at startup in
profile order (4.8 s for 5,600 slots). On the probe prompts 89.2 % of cold lookups are in place (67.5 % with the previous 1,000-slot LRU + 5.1 GiB static tier), 1.23
misses per call. The static tier shrank to 0.2 GiB because at equal RAM a pure LRU beat tier + LRU in a simulation on the traced calls (84 % vs 79 %). The
cap is Metal's `recommendedMaxWorkingSetSize` on this mini (11.84 GiB), not RAM.

**Slim doorbell** (`STRATA_EXL3_SLIM`, default on for EXL3 configs): the host pulls only the rows of x that have a miss. **Pool spin off**
(`STRATA_POOL_SPIN_US=0`, default in `shim/run-strata.sh`): spinning NEON workers starved the USB host thread (13-15 ms per layer of stall).

## Prefill: the Mac as the cold-expert lane

Streaming every cold expert over the 190-350 MB/s link took 138 s per 2,048-token chunk (12 tok/s). With `STRATA_PF_MAC=1` the cold experts never
cross the link: per layer, the chunk's fp16 MoE input comes down (lazy device-to-host), the Mac runs the cold (token, expert) pairs with the
grouped Metal prefill entry (`exl3m_moe_prefill*`) while the 3090 runs the resident experts, and the partial sums go back as fp16
(`STRATA_PF_MAC_Y16`) to be combined on the GPU. The blobs come from the LRU in place (hits) or are read with `F_NOCACHE` into slot sets borrowed from the
LRU's coldest slots for the duration of the prompt (`STRATA_PF_SETS=3`; no extra wired memory).

**Lockstep groups** (`STRATA_PF_GROUP=4`): G = 4 chunks of 2,048 tokens run layer-major, so each layer's cold blobs are read from NVMe once per group
instead of once per chunk. One thread per chunk, a fixed turn order forming a software pipeline: front half (attention/GDN, routing, resident
experts, x16 download, async Metal submit), then the back half of the chunk `lag` turns earlier (Metal wait, y16 push, combine). Each extra chunk
keeps its own residual stream and QSA step table on the device (an 84 MB stream per chunk, which is where the VRAM headroom for G=4 goes:
`--vram-reserve-mib 400`). At equal slot counts the lockstep output is bitwise equal to the plain lane.

## KV cache and context

`--kv k8v4 --max-context 204800`: int8 keys with exact scores plus rotated 4-bit values, 816 B per cell per KV layer (12 QSA layers + the MTP draft
layer), 2.61 GB at 200K, 1,308 of the 9,108 expert slots that would otherwise fit. fp16 would cost 2,969 slots, int8 1,584, q4_0 914; `--kv-resident`
(host-resident KV) costs 2.75 GB of mini RAM and was measured negative. The 36 GDN layers hold constant-size state.

## Vision

`serve/server.py` runs `strata-vision` (llama.cpp mtmd + the BF16 mmproj) as a child process: it encodes the image on the **Mac's GPU** (Metal), writes an
embeddings file, the server expands `<|image_pad|>` and the engine's `GENI` path prefills the picture rows with M-RoPE positions through the batched
prefill path. The encoder holds ~1.1 GB wired RAM while loaded, so `idle_unload_s: 120` (patch 0002) ends it after 120 idle seconds.

## Settings in the final config

`configs/strata-exl3-full512-final-g4-lru.json.tmpl`:

| setting | value | what it does |
|---|---|---|
| `--pack` | `~/exl3/pack_full512` | dense pack (symlinks to the IQ3_S pack) + `experts.bin` + `exl3_experts.txt` marker |
| `--expert-profile` | `exl3/placement/f512c/profile.bin` | hottest-first ranking of all 24,576 (layer, expert) pairs |
| `--expert-cache auto` + `--vram-reserve-mib 400` | | 7,400 slots; 400 MiB headroom (needed for G=4: the 256 MiB tgcu chunk at the periodic checkpoint) |
| `--resident-budget-gib 0.2` | | static locked tier 0.2 GiB (115 blobs) |
| `--kv k8v4 --max-context 204800` | | 200K context |
| `--spec 4 --spec-min-p 0.5 --mtp ~/models/qwen38fn/mtp/rt` | | MTP drafting |
| `--prefill 2048` `--no-prefill-borrow` | | chunk size; the prompt path allocates its own buffers (borrowing VRAM slots for prompts crashed 2.5K-token prefills on this transport) |
| `--mmap-experts --pcie-frac 0 --adapt-every 0` | | `experts.bin` is mmapped; the other two switch off Strata's PCIe-streaming fraction and adaptive re-placement [INFERENCE from the flag names] |
| `STRATA_EXL3_METAL=1`, `_SLOTS=5600` | | Metal lane, LRU size |
| `STRATA_EXL3_METAL_SPLIT=1`, `_EARLY=1`, `STRATA_MAC_PULL_EARLY=1`, `STRATA_MAC_PULL_NOCACHE=1` | | decode lane as above |
| `STRATA_PF_MAC=1`, `_Y16=1`, `_NOCACHE=1`, `STRATA_PF_GROUP=4`, `STRATA_EXL3_METAL_JOBS=4`, `STRATA_PF_SETS=3` | | prefill lane as above |
| `STRATA_RESIDENT_HEADROOM_GIB=1.5` | | carried over from the first full-512 configs; with `--resident-budget-gib 0.2` it no longer sizes anything [INFERENCE] |
| defaults from `shim/run-strata.sh` | `STRATA_POOL_SPIN_US=0`, `STRATA_EXL3_SLIM=1`, `STRATA_HS_PRE=1`, `STRATA_HS_DEFER=1`, `STRATA_MAC_PULL=1` | |
| `vision` | Metal `strata-vision-metal`, mmproj BF16, `max_tokens 1024`, `idle_unload_s 120` | |

Older configs kept for rollback: `final-g4` (G=4, 1,000-slot LRU + 5.1 GiB tier) and `final` (G=3, no `--vram-reserve-mib 400`). Going back to `final-g4`
needs the engine binary from before the LRU change (`strata.bak-1012-prelru` on the mini); the new binary on the old config was not run.

## What is not here

A KL/top-1 comparison against stock exllamav3; a decode path that hides the 0.85 ms two-transaction floor (GPU-written landed flags are not
possible because layer l's router input only exists after the GPU finishes attention l); GPU-side expert prefetch from a layer-ahead router (measured
negative: 42-49 % coverage); more than ~5,600 LRU slots (needs `iogpu.wired_limit_mb`, a system setting that was not changed).
