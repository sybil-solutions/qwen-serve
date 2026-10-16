# Patches

Apply order matters. Base for every `strata/` patch: Strata **v0.1.41**, commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`
(<https://github.com/Niko1221/Strata>, MIT; the public tag points at exactly this commit, checked 2026-10-12).

| patch | what it is | files |
|---|---|---|
| `strata/0001-macos-tgcu-exl3-engine.patch` | the macOS port + EXL3 engine: build on macOS through the tgcu shim (`macos_compat.hpp`, `tgcu.hpp`), EXL3 expert layout, EXL3 GPU arms, NEON pool tier (`libexl3cpu`), pre-armed USB handshake (`STRATA_HS_PRE/DEFER`), whole-blob preads (`STRATA_MAC_PULL`), first Metal lane (`libexl3metal`), Mac cold-expert prefill lane (`STRATA_PF_MAC`), pipelined transport hooks. State of the engine tree on 2026-10-11 15:41 (PrefillPipeline). | 23 files, 5 of them new |
| `strata/0002-server-vision-idle-unload.patch` | `serve/server.py`: `idle_unload_s` for the vision encoder (the Metal encoder holds ~1.1-1.2 GB wired RAM; it is started by the first image and ended after N idle seconds) | `serve/server.py` |
| `strata/0003-prefill-lockstep-groups.patch` | prefill: layer-major chunk-group lockstep (`STRATA_PF_GROUP`), even chunk split, cross-group read-ahead, fp16-I/O Metal prefill entry, async Metal submit | `prefill.hpp`, `prefill.cpp`, `generate.cpp` |
| `strata/0004-exl3-metal-lane-lru-decode.patch` | decode + memory: Metal lane with an O(1) LRU of registered slots (`STRATA_EXL3_METAL_SLOTS`), in-place/miss split job (`STRATA_EXL3_METAL_SPLIT`), early start (`STRATA_EXL3_METAL_EARLY`, `STRATA_MAC_PULL_EARLY`), LRU preload by profile rank, prompt sets borrowed from the LRU, `STRATA_MAC_PULL_NOCACHE`, G=4 support | `expert_source.cpp`, `verify.cpp` |
| `strata/0005-prefill-splitf-min-groups.patch` | prefill: `STRATA_PF_SPLITF_MIN` gates the split (y push behind the next chunk's front) order to groups of at least that many chunks (default 2 = unchanged; the shipped config sets 4: +9 % on 4-chunk groups, -25 % on 2-chunk ones) | `prefill.cpp` |
| `tinygrad/0001-tg-strata-nv-usb-fixes.patch` | on top of tinygrad `e765a2abf` (PR #17369 head, `bounty/nv-usb3-wip`, merged with master `7d2420e`): compile-server short reads, NV reloc fields, PROM read stability after a killed session | 3 files |
| `vision/0001-strata-vision-metal-macos.patch` | Strata `tools/vision`: `STRATA_VISION_METAL` CMake option (ggml-metal, Accelerate, no OpenMP), no `posix_fadvise` on macOS | 2 files |

`strata/0001..0005` were produced by diffing the live source trees against the pristine tag (no hand edits) and verified to reproduce those
trees byte for byte: `0001`+`0002` == the deployed `~/strata` working tree; `0003`+`0004` == the `~/strata-lockstep` tree the shipped engine was built
from before the host-tail step (`expert_source.cpp` and `verify.cpp` taken from the backups made right before the next experiment started editing them; the tree was
unchanged at that point, checked with `cmp`); `0005` brings `prefill.cpp` to the host-tail step's tree (sha256 `adfdddc8fa1c...` = `src/prefill/prefill.cpp` of `~/strata-lockstep` on the mini, checked 2026-10-12).
`scripts/verify-card-free.sh patches` re-runs the apply check.

`history/` holds the per-step diffs recorded during the campaign (for reading the sequence of changes; they were made against intermediate trees
and are **not** meant to be applied): `hs2` (pre-armed handshake, whole-blob pread), `warmfix`, `metal` (libexl3metal), `pfmac`, `pfpipe`
(pipelined USB transport), `lockstep`, `decode`, `g4fix` (tgcu ring-wrap deadlock fix), `lru`, `hosttail` (HostTailAndPrefill: the `shim/tgcu.py`/`tgcu_api.py`/`tgcu_prof.py` and `exl3/metal` diffs of the last step are already applied to those files in this repo; `prefill.cpp.diff` is patch `0005`).

Files that are not patches because they are new in this repo: `shim/` (tgcu), `exl3/` (kernels, libs, packers), `vision/` (build script).
