// M4 prefill: exllamav3's fused prompt-shaped MoE kernel (exl3_moe_kernel.cuh, vendored unmodified) for 3 bpw MUL1 on
// sm_86, the three row tiers upstream's BlockSparseMLP launches (16-row N = 128: the dims 2560 / 640 are not both
// multiples of 256; 32- and 64-row N = 128).  One launch per tier per layer runs every expert of the layer (a group of
// SMs per expert, dynamic expert tickets).  Host side + C API: strata_m4/exl3_strata_pf_host.cu.  Build:
// `build.sh exl3_strata_pf`.
#include <cuda_fp16.h>
#include <stdint.h>
#include "vendor/exllamav3_ext/quant/exl3_moe_kernel.cuh"

template __global__ void exl3_moe_kernel<3, 128, 2, 16, false, false>(EXL3_MOE_KERNEL_ARGS);
template __global__ void exl3_moe_kernel<3, 128, 2, 32, false, false>(EXL3_MOE_KERNEL_ARGS);
template __global__ void exl3_moe_kernel<3, 128, 2, 64, false, false>(EXL3_MOE_KERNEL_ARGS);
template __global__ void exl3_moe_kernel<2, 128, 2, 16, false, false>(EXL3_MOE_KERNEL_ARGS);   // mixed precision: 2 bpw
template __global__ void exl3_moe_kernel<2, 128, 2, 32, false, false>(EXL3_MOE_KERNEL_ARGS);
template __global__ void exl3_moe_kernel<2, 128, 2, 64, false, false>(EXL3_MOE_KERNEL_ARGS);

// Pointer tables for the kernel from the C API's packed entry table (built on the host, one upload per call):
// ent[i] = device base of entry i's EXL3 blob; tables[f][i] = ent[i] + field_off[f] (f in MoeCoopParams order:
// g/u/d x trellis/suh/svh).
extern "C" __global__ void exl3s_pf_tables(const unsigned long long* __restrict__ ent, int n, unsigned f0, unsigned f1,
                                           unsigned f2, unsigned f3, unsigned f4, unsigned f5, unsigned f6, unsigned f7,
                                           unsigned f8, unsigned long long* __restrict__ tables) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned off[9] = {f0, f1, f2, f3, f4, f5, f6, f7, f8};
    const unsigned long long b = ent[i];
#pragma unroll
    for (int f = 0; f < 9; ++f) tables[(size_t) f * n + i] = b + off[f];
}
