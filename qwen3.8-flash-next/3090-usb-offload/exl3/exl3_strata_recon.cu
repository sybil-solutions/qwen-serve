// M4 prefill: exllamav3 reconstruct_had_tile for 3 bpw MUL1 (vendored unmodified), a separate TU from exl3_strata.cu
// (the moe_coop and reconstruct headers both define codebook helpers).  Build: `build.sh exl3_strata_recon`.
#include <cuda_fp16.h>
#include <stdint.h>
#include "vendor/exllamav3_ext/quant/reconstruct_tiles.cuh"
extern "C" __global__ __launch_bounds__(RH_THREADS) void exl3_recon_had_k3_cb2
    (half* __restrict__ out, const uint16_t* __restrict__ packed, const half* __restrict__ suh,
     const half* __restrict__ svh, int packed_blocks_n)
{ reconstruct_had_tile<3, 2>(out, packed, suh, svh, packed_blocks_n, 0); }
extern "C" __global__ __launch_bounds__(RH_THREADS) void exl3_recon_had_k2_cb2
    (half* __restrict__ out, const uint16_t* __restrict__ packed, const half* __restrict__ suh,
     const half* __restrict__ svh, int packed_blocks_n)
{ reconstruct_had_tile<2, 2>(out, packed, suh, svh, packed_blocks_n, 0); }
