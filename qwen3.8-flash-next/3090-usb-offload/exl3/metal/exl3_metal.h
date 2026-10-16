/* exl3_metal.h - EXL3 K=3 MUL1 (2560x640) expert MoE compute on the Apple GPU (Metal). Plain C API, thread-safe, no ObjC.
 *
 * One expert = one 1,862,400 B pack blob (exl3_pack.json: gate/up trellis [160,40,48] u16, down trellis [40,160,48] u16, then fp16
 * suh/svh vectors). expert(x) = down( silu(gate(x)) * up(x) ) with the Hadamard rotation / suh / svh exactly as libexl3cpu +
 * Strata pool.cpp (x fp32 -> fp16 -> had_in -> trellis GEMV fp32 accumulate -> had_out*svh; SwiGLU in fp32 rounded to fp16).
 * y_out[r] = sum over pairs p with row_idx[p]==r of weight[p] * expert(blob_ptrs[expert_idx[p]], x[r]); y_out is OVERWRITTEN
 * (rows without a pair are zeroed). Pair order inside a row fixes the fp32 summation order: results are deterministic.
 *
 * Blobs: any host pointer works. Pointers inside a registered region (exl3m_register: page-aligned NoCopy MTLBuffers over
 * the mmap'd experts.bin / locked RAM-tier buffers) are read by the GPU in place; other blobs are memcpy'd into a per-slot
 * staging buffer (counted in exl3m_stats.blobs_staged). A job reads blob memory until exl3m_wait returns.
 *
 * Return codes: 0 ok, <0 error (exl3m_last_error() gives a thread-local message).
 */
#ifndef EXL3_METAL_H
#define EXL3_METAL_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EXL3M_H 2560          /* hidden size (x / y row length) */
#define EXL3M_FF 640          /* expert intermediate size */
#define EXL3M_BLOB_BYTES 1862400

typedef struct exl3m_ctx exl3m_ctx;
typedef struct exl3m_job exl3m_job;

typedef struct {
    uint64_t decode_calls, prefill_calls;
    uint64_t blobs_mapped, blobs_staged;       /* blob resolutions served in place vs copied */
    double gpu_ms_last;                         /* GPUEndTime - GPUStartTime of the last waited job */
    double wall_ms_last;                        /* submit -> complete (host clock) of the last waited job */
    double encode_ms_last;                      /* host time spent building tables + encoding of the last submit */
    double copy_ms_last;                        /* host time spent copying x in / y out + staging blobs, last job */
} exl3m_stats;

/* n_threads: hint for host-side table building / copies (currently unused, reserved). n_slots: concurrent in-flight jobs
 * (scratch sets), 1..8, 0 -> 2. Returns NULL on failure (no Metal device, metallib load error). */
exl3m_ctx *exl3m_create(int n_threads, int n_slots);
void exl3m_destroy(exl3m_ctx *ctx);
const char *exl3m_last_error(void);
const char *exl3m_device_name(exl3m_ctx *ctx);

/* Register a host region [base, base+len) holding blobs (mmap'd experts.bin range or locked RAM buffer). The page-aligned (16 KiB)
 * core is wrapped as NoCopy MTLBuffer(s) (pieces of <= 1 GiB overlapping by one blob so no blob straddles pieces). Returns the
 * number of bytes wrapped (>0), or <0. Registration of file-backed memory makes the GPU access pages through the kernel's wired mapping:
 * pages must be resident to be fast (see microbench). */
int64_t exl3m_register(exl3m_ctx *ctx, const void *base, size_t len);
int exl3m_unregister(exl3m_ctx *ctx, const void *base);   /* base as passed to exl3m_register */

/* Decode shape: m <= 8 rows (x [m x 2560] fp32), up to 4096 pairs, <= 96 distinct experts, <= 8 pairs per expert.
 * blob_ptrs[] is indexed by expert_idx[] (length max(expert_idx)+1). Synchronous = submit + wait. */
int exl3m_moe_rows(exl3m_ctx *ctx, const float *x, int m, int n_pairs, const int32_t *expert_idx, const int32_t *row_idx,
                   const float *weight, const void *const *blob_ptrs, float *y_out);
/* Asynchronous: x is copied during submit; y_out is written by exl3m_wait. Blob memory must stay valid until the wait. */
int exl3m_moe_rows_submit(exl3m_ctx *ctx, const float *x, int m, int n_pairs, const int32_t *expert_idx, const int32_t *row_idx,
                          const float *weight, const void *const *blob_ptrs, float *y_out, exl3m_job **job);
int exl3m_wait(exl3m_ctx *ctx, exl3m_job *job);             /* blocks, copies y_out, releases the job and its scratch slot */

/* Prefill shape: x [n_tok x 2560], many rows per expert (n_tok <= 16384, <= 16384 rows per expert, <= 1<<20 pairs). Decode of every
 * used expert to fp16 + tiled simdgroup GEMM (fp32 accumulate) + fused rotation / SwiGLU / weighted combine; deterministic. */
int exl3m_moe_prefill(exl3m_ctx *ctx, const float *x, int n_tok, int64_t n_pairs, const int32_t *expert_idx, const int32_t *row_idx,
                      const float *weight, const void *const *blob_ptrs, float *y_out);
int exl3m_moe_prefill_submit(exl3m_ctx *ctx, const float *x, int n_tok, int64_t n_pairs, const int32_t *expert_idx,
                             const int32_t *row_idx, const float *weight, const void *const *blob_ptrs, float *y_out, exl3m_job **job);
/* fp16 in / fp16 out variant: x16 [n_tok x 2560] fp16 bits, y16_out [n_tok x 2560] fp16 bits (fp32 accumulation, clamp +-65504, round to nearest even;
 * bit-identical to exl3m_moe_prefill + the host's clamp/cast).  Collect with exl3m_wait. */
int exl3m_moe_prefill_h_submit(exl3m_ctx *ctx, const uint16_t *x16, int n_tok, int64_t n_pairs, const int32_t *expert_idx, const int32_t *row_idx,
                               const float *weight, const void *const *blob_ptrs, uint16_t *y16_out, exl3m_job **job);

/* Tuning: "pf_chunk_rows" (default 4096), "pf_chunk_experts" (default 16). Returns 0 or -1 for an unknown key. */
int exl3m_set_param(exl3m_ctx *ctx, const char *key, int64_t value);
void exl3m_get_stats(exl3m_ctx *ctx, exl3m_stats *out);
int64_t exl3m_alloc_bytes(exl3m_ctx *ctx);          /* MTLDevice.currentAllocatedSize */
int64_t exl3m_recommended_bytes(exl3m_ctx *ctx);    /* MTLDevice.recommendedMaxWorkingSetSize */

#ifdef __cplusplus
}
#endif
#endif
