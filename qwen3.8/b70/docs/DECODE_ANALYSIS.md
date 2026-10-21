# Decode Token Generation Analysis — Qwen3.8-27B on 2× Intel Arc Pro B70

## Known ceilings (from b70-optimization-lab, Aug 2026)

| Lane | Config | Decode tok/s | Notes |
|------|--------|-------------:|-------|
| **llama.cpp Q4_K_M TP2** (target-only) | mndodd fork + lab Q4K SwiGLU fusion | **49.72** | Our exact model/config. Lab's promoted record. |
| **vLLM AutoRound INT4 TP2 MTP5** | devan-carlin/Qwen3.8-27B-int4-AutoRound | **101.92** | INT4 weights (~half the bytes) + MTP. Different stack (vLLM not llama.cpp). |
| llama.cpp Q8_0 TP2 | mndodd fork, Q8_0 weights | 36.77 | Larger weights → slower (confirms BW-bound) |
| vLLM FP8 TP2 | vLLM-XPU graph c1 | 21.71 | Triton GDN fallback → slow |
| llama.cpp Q4_K_M TP2 + MTP (easy) | Cold Fusion MTP2, 94.4% accept | 84.3 (README) / 38.4 (PR#34 single-GPU) | MTP helps easy tasks only |

**Key insight: our 44.3 tok/s (JIT) vs lab's 49.72 tok/s (AOT+Q4K fusion) = 11% gap.
The 101.9 tok/s INT4 result proves the B70 hardware can go much faster — but requires
a different quantization (INT4, half the weight bytes) and stack (vLLM/XPU).**

## Revised bottleneck understanding

**Two views reconciled:**

1. **fusion.cpp comment** says "~1089 non-matmul kernels/token; each pays a fixed
   execution latency that dominates its byte cost" — but this was measured on **PVC**
   (which has much higher kernel launch overhead than BMG).

2. **Pipeline analysis** (from measured numbers) shows decode is **memory-bandwidth-bound**
   on the B70 (BMG G31, 608 GB/s peak GDDR6):
   - Our 44.3 tok/s = 22.6 ms/token; effective BW = 377 GB/s/card (62% of peak)
   - Lab's 49.7 tok/s = 20.1 ms/token; effective BW = 422 GB/s/card (69% of peak)
   - Weight read at peak BW: 14.0 ms = **62-70% of token time**
   - Non-weight overhead (launch latency + compute + comm): 6-9 ms = **30-38%**

**The truth: decode is ~65% bandwidth-bound, ~35% launch-overhead-bound.**

- The **bandwidth ceiling** is the hard limit: at 100% BW utilization, max = 608/8.5 ≈ 71.5 tok/s
- The **lab achieves 70%** of this ceiling (49.7 tok/s)
- Our **JIT build achieves 62%** (44.3 tok/s) — the 2.5ms gap is kernel launch overhead
  that the AOT build's MMVQ fusions eliminate

### Optimization implications:

| Lever | Mechanism | Expected gain | Status |
|-------|-----------|---------------|--------|
| **Reduce kernel launches** | Missing PRs (#26643, #26779, #26411) | ~11% (close JIT→AOT gap) | Available now |
| **Reduce weight bytes** | Q4_0 model (~14.5 GB vs ~17 GB) | ~12% (fewer bytes to read) | Available now |
| **Increase BW utilization** | AOT reordered weights (better cache/coalescing) | ~10% (69%→80% BW util) | Blocked (AOT runtime) |
| **Graph capture** | Eliminate all launch overhead | ~15% (if BW stays same) | Blocked (needs async_mem ext) |
| **INT4 quantization** | ~8.5 GB vs ~17 GB (half the bytes) | ~40% (if coherent) | Needs AutoRound/vLLM path |

### Per-layer breakdown (64 layers, TP2):

| Operation | Kernel count | Currently fused? | Notes |
|-----------|-------------:|-------------------|-------|
| RMS norm + weight mul | 1 | ✅ P1 (ADD+RMS+MUL) | Residual chain fused |
| Input projection (in_proj_qkvz, in_proj_ba) | 2 | ❌ MMVQ PAIR GDN disabled | Requires reordered Q8_0 (AOT-only) |
| **GDN layers (48 of 64):** | | | |
| — Conv1d update | 1 | ✅ FUSED_CONV_STATE_IO | State I/O fused |
| — Delta rule state update | 1 | ✅ FUSED_GDN_STATE_IO | State I/O fused |
| — Sigmoid gating (g + beta) | 1 | ✅ FUSED_GDN_BETA_SIGMOID | Beta in consumer |
| — L2 norm (q, k) | 1 | ✅ P3 (L2_NORM_PAIR) | Batched |
| — Output projection | 1 | ❌ Separate launch | |
| — GDN tail (RMS+SILU+MUL→Q8) | 1 | ✅ P5/P6 (FUSE_EXT bit4) | Fused to Q8 quantize |
| **GQA layers (16 of 64):** | | | |
| — QK norm + RoPE | 1 | ✅ FUSED_QK_NORM_ROPE | |
| — Flash attention (decode) | 1 | ✅ FATTN VEC, 256 threads | MMA disabled (JIT) |
| — Output gate (CONT+SIGMOID+MUL) | 1 | ✅ P4 | |
| — Output projection | 1 | ❌ Separate launch | |
| SwiGLU FFN (gate+up+down) | 2 | ✅ SWIGLU_Q8 (gate+up fused) | Down separate |
| Residual add + all-reduce | 1 | ✅ META_FUSE_ALLREDUCE_ADD | Comm fused |
| Q8 quantize for next layer | varies | ✅ Multiple fusions | Dedup enabled |

### What's DISABLED and why:

| Fusion | Requires | Status |
|--------|----------|--------|
| MMVQ PAIR (FFN gate+up) | Reordered Q8_0 weights | AOT-only layout; JIT can't produce |
| MMVQ PAIR GDN (QKV+Z) | Reordered Q8_0 weights | Same |
| MMVQ TRIPLE ATTN (Q/K/V) | Reordered Q8_0 weights | Same |
| MMVQ TRIPLE/QUAD GDN | Reordered Q8_0 weights | Same |
| MMVQ SWIGLU Q4K | Q4K reorder | AOT-only layout |
| FATTN MMA | joint_matrix | Unavailable under JIT oneAPI 2025.3 |
| Q4K REORDER | AOT reordered tensor layout | AOT-only |
| Graph capture | Command graph | GGML_SYCL_GRAPH=OFF at compile time |

**The structural blocker**: 7 MMVQ fusion knobs would each eliminate 1-3 kernel launches
per layer (up to ~200 fewer launches/token), but they require weight tensors in a reordered
layout that only the AOT `bmg_g31` build produces. On JIT, `opt_for_reorder` fails to set
`optimized_feature.reorder = true`, so the fusion guard returns false.

## New code hitting the repos (June-August 2026)

### Already in our build (4302fb5, sycl-sync-0812, Aug 12):

| PR | What | Impact |
|----|------|--------|
| #21527 | Q8_0 reorder (~3x tg on B70) | ✅ Foundational — enables Q8 reorder path |
| #21700 | Native subgroup 16 for K-quant DMMV | ✅ Decode DMMV faster |
| #21845 | Multi-column MMVQ port from CUDA | ✅ MTP +40-95% |
| #22147 | Battlemage AOT via spir64_gen | ✅ AOT build path fixed |
| #25025 | oneMKL GEMM flash attention (XMX) | ✅ Prefill FA |
| #25205 | fattn_vec_nthreads=256 for BMG | ✅ Decode FA faster |
| #25222 | oneDNN SDPA flash attention (Xe2) | ✅ Prefill FA |
| #25874 | oneDNN SDPA for quantized KV | ✅ Prefill FA with q8 KV |
| #26015 | Fuse RMS_NORM + MUL | ✅ Decode fusion infra |
| #26612 | SSM conv window coalescing | ✅ Prefill conv faster |

### MISSING — merged to upstream master, NOT in our build (109 commits behind):

| PR | Date | What | Decode impact | Priority |
|----|------|------|---------------|----------|
| **#26643** | Aug 14 | Fuse GDN state writeback cpy | -1 kernel/GDN layer = -48 kernels/token | ⭐⭐⭐ |
| **#26779** | Aug 14 | Fuse mul_mat(gate)+mul_mat(up)+GLU for Q4_K FFN | -2 kernels/layer (FFN) = -128 kernels/token | ⭐⭐⭐ |
| **#26411** | Aug 13 | Fuse UNARY(silu/sigmoid/softplus)+MUL | -1 kernel/activation | ⭐⭐ |
| **#27160** | Aug 17 | Fix thread/block in quantized cpy | q4_0→f32 cpy: 20→158 GB/s | ⭐⭐ |
| **#26623** | Aug 14 | Recurrent state rollback for ssm_scan | MTP correctness | ⭐ |
| **#26251** | Aug 13 | DMMV ESIMD Q3_K kernel | Faster Q3_K decode | ⭐ |
| **#26372** | Aug 13 | Remove fp32 promotion in gemm non-oneDNN | +4.3% prefill | ⭐ |
| **#26789** | Aug 13 | Host pinned mem for H2D | +13.5% prefill | ⭐ |
| **#26800** | Aug 13 | Concat supports Q4_0-Q8_0 | KV/SSM concat | ⭐ |
| **#27298** | Aug 18 | Honor GGML_HINT_SRC0_IS_HADAMARD | Minor | ⭐ |

### MISSING — still OPEN (not merged to master, but merge-ready):

| PR | What | Decode impact | Priority |
|----|------|---------------|----------|
| **#26689** | TILE for quantized KV decode on BMG | **+42% to +169% decode** (q4_0 KV @118K: 13→30 tok/s) | ⭐⭐⭐⭐ |
| **#27062** | Q4_K multicol MMVQ (2-row reuse) | **+30% spec decode** | ⭐⭐⭐ |

## Optimization plan (ordered by expected decode impact)

### Phase 1: Rebuild with upstream master (109 commits ahead) — closes JIT→AOT gap

Cherry-pick or merge the 7 missing merged PRs onto our patched fork. This adds:
- **#26643**: GDN state writeback fusion → -48 kernel launches/token
- **#26779**: Q4_K FFN gate+up+GLU fusion → -128 kernel launches/token (only Q4_K)
  - **The lab already adapted this** as the `q4k-increment.patch` we already have!
  - Lab result: 49.46→50.27 tok/s (+1.64%), 12-prompt cold suite 49.72 tok/s
  - Our patch file IS this fusion. Check if it's correctly applied + enabled.
- **#26411**: UNARY+MUL fusion → fewer activation kernels
- **#27160**: 8× faster quantized cpy (matters for KV cache operations)

**Critical check**: The lab enables `GGML_SYCL_MMQ_Q4K_REORDER=1` and
`GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=1` for the Q4K fusion. Our entrypoint has them
**set to 0** (disabled). This alone could be the 11% gap.

**Expected**: ~11% decode speedup (44→49 tok/s), closing the JIT→lab-AOT gap.

### Phase 2: Cherry-pick the two open decode PRs

- **#26689**: TILE for quantized KV decode on BMG. **NOTE: The fork already has this
  for q8_0 KV** (commit `4c2d55542`, +56.6% decode @32k). We use f16 KV which
  already routes to TILE. This PR is mainly for stock upstream; low incremental value
  for us unless we switch to q8_0 KV.
- **#27062**: Q4_K multicol MMVQ with 2-row activation reuse. +30% on spec decode.
  Helps MTP path only.

**Expected**: Minimal for our f16-KV config; helps if we enable MTP.

### Phase 3: Q4_0 model swap (reduce weight bytes)

Q4_0 is ~14.5 GB vs Q4_K_M's ~17 GB. At the same BW utilization, fewer bytes =
proportionally faster decode. Estimated: 44→~50 tok/s (12% gain).
**Risk**: Q4_0 is lower quality than Q4_K_M — need coherence gate.

### Phase 4: Enable graph capture (GGML_SYCL_GRAPH=ON)

Currently compiled with GRAPH=OFF. Requires `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC`
and `SYCL_EXT_ONEAPI_VIRTUAL_MEM` extensions. Our oneAPI 2025.3 does NOT define
these — requires oneAPI 2026.1.x with `/opt/intel/oneapi/umf/1.0/lib` on
`LD_LIBRARY_PATH` (so `libur_loader.so.0` and `libur_adapter_level_zero.so.0` find
`libumf.so.1` and enumerate B70 devices).

### Phase 5: AOT bmg_g31 build (unlocks MMVQ fusions + better BW utilization)

The AOT build produces reordered weight tensors that unlock all 7 disabled MMVQ fusion
knobs AND improve memory access coalescing (raising BW utilization from 62%→69%+).
The lab's 49.7 tok/s was achieved with the AOT build.

**Root cause of 2026.1.x enumeration failure & current AOT blocker**:
Under `intel/oneapi-basekit:2026.1.*`, the `0 platforms reported` error is caused by a
missing runtime library path (`libumf.so.1` in `/opt/intel/oneapi/umf/1.0/lib`), not by an
incompatible UR Level Zero adapter. Adding
`/opt/intel/oneapi/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/umf/1.0/lib`
to `LD_LIBRARY_PATH` (or `/etc/ld.so.conf.d/oneapi.conf` + `ldconfig`) resolves `libumf.so.1`
and enumerates both Arc Pro B70s (`level_zero:gpu:0`, `level_zero:gpu:1`). However, when built
with `-DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DEVICE_ARCH=bmg_g31` on the `dp4a2` snapshot,
`ggml_sycl_reordered` (`Reorder + MulMatVecQ`) triggers a GPU hang during `check_Prefill_Speed`
that resets both cards.

### Phase 6: INT4 / AutoRound quantization (bandwidth halving)

If weight bytes could be halved (~8.5 GB instead of ~17 GB), decode could approach
~90 tok/s (matching the vLLM AutoRound INT4 result of 91.9 tok/s). This is the single
biggest potential win but requires a different quantization path (AutoRound/vLLM, not
llama.cpp GGUF). Out of scope for the current llama.cpp SYCL stack.

## Summary

The decode bottleneck is **kernel launch overhead** (1345 launches/token at ~8µs each).
Our JIT build already has most fusions enabled but is missing 7 recently-merged PRs and
2 open merge-ready PRs that directly attack this. The fastest path to improvement:

1. **Rebuild** with upstream master + patches (gets #26643, #26779, #26411, #27160)
2. **Cherry-pick** #26689 (TILE KV decode, +42-169%) and #27062 (Q4_K multicol, +30%)
3. **Try** GGML_SYCL_GRAPH=ON to eliminate launch overhead entirely
4. **Long-term**: AOT build to unlock MMVQ fusions (blocked on driver/runtime)
