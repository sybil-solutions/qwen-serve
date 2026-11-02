// M4: the EXL3 device module Strata loads (libstrata_exl3.dylib embeds this fatbin; host side strata_m4/exl3_strata_host.cu).
// Decode (verify windows, graph-captured): exl3_plan_kernel (residency -> pointer tables), exl3s_f2h (MoE input to fp16),
// exllamav3 exl3_moe_coop rot / A / B (vendored unmodified, 3 bpw MUL1, wide tiles).  Prefill (FP16 fallback arm):
// exl3_recon_had_k3_cb2 (W_orig fp16 [k, n]) + exl3s_gu_interleave / exl3s_transpose into Strata's dq_gu / dq_d layout.
// Build: `build.sh exl3_strata`.
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>
#define EXL3_MOE_COOP_DEFINE_ROT
#include "vendor/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh"
#include "exl3_plan.cu"

namespace exl3_moe_coop_ns {
template __global__ void exl3_moe_coop_a_kernel<3, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_b_kernel<3, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_a_kernel<2, 2, true, false>(const MoeCoopParams);   // mixed precision: 2 bpw layers
template __global__ void exl3_moe_coop_b_kernel<2, 2, true, false>(const MoeCoopParams);
}

// x fp32 rows (stride x_stride) -> fp16 [rows, H]
extern "C" __global__ void exl3s_f2h(const float* __restrict__ x, int x_stride, half* __restrict__ y, int rows, int H) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) rows * H) return;
    const int64_t r = i / H, c = i % H;
    y[i] = __float2half_rn(x[r * x_stride + c]);
}

// gate / up W_orig [K_in = 2560][n = 640] (fp16) -> dq_gu [1280][2560]: row 2k = gate column k, row 2k+1 = up column k
// (Strata's interleaved gate/up rows, swiglu_interleaved).  32x32 tiles through shared memory.
extern "C" __global__ void exl3s_gu_interleave(const half* __restrict__ g, const half* __restrict__ u,
                                               half* __restrict__ dq, int k_in, int n) {
    __shared__ half tg[32][33], tu[32][33];
    const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;   // c: output column (0..n), r: input row (0..k_in)
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        tg[j][threadIdx.x] = g[(size_t) (r0 + j) * n + c0 + threadIdx.x];
        tu[j][threadIdx.x] = u[(size_t) (r0 + j) * n + c0 + threadIdx.x];
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int c = c0 + j;
        dq[(size_t) (2 * c) * k_in + r0 + threadIdx.x] = tg[threadIdx.x][j];
        dq[(size_t) (2 * c + 1) * k_in + r0 + threadIdx.x] = tu[threadIdx.x][j];
    }
}

// W [rows][cols] -> W^T [cols][rows] (down: W_orig [640][2560] -> dq_d [2560][640])
extern "C" __global__ void exl3s_transpose(const half* __restrict__ w, half* __restrict__ t, int rows, int cols) {
    __shared__ half tile[32][33];
    const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    for (int j = threadIdx.y; j < 32; j += blockDim.y) tile[j][threadIdx.x] = w[(size_t) (r0 + j) * cols + c0 + threadIdx.x];
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += blockDim.y) t[(size_t) (c0 + j) * rows + r0 + threadIdx.x] = tile[threadIdx.x][j];
}

// Resident-only routing (STRATA_EXL3_RESIDENT_ONLY=1): the router's logits of experts not in the VRAM cache -> -inf, so
// the model's own top-k (softmax over all, top 10, renormalised) selects among resident experts only = a prune to the
// resident set with the router renormalised.  One block per token row; a row whose layer has fewer than k resident
// experts is left as it is.
extern "C" __global__ void exl3s_mask_logits(float* __restrict__ logits, int NE, const int32_t* __restrict__ res, int k) {
    __shared__ int cnt;
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    int c = 0;
    for (int e = threadIdx.x; e < NE; e += blockDim.x) c += res[e] >= 0;
    atomicAdd(&cnt, c);
    __syncthreads();
    if (cnt < k) return;
    float* row = logits + (size_t) blockIdx.x * NE;
    for (int e = threadIdx.x; e < NE; e += blockDim.x)
        if (res[e] < 0) row[e] = -INFINITY;
}
