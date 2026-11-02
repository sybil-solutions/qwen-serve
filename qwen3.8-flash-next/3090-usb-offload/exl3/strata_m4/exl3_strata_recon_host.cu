// M4: host stub for ../exl3_strata_recon.fatbin (exllamav3 reconstruct_had_tile, 3 bpw MUL1).  See exl3_strata_host.cu.
#include <cuda_fp16.h>
#include <cstdint>
extern "C" __global__ void exl3_recon_had_k3_cb2(half*, const uint16_t*, const half*, const half*, int) {}
extern "C" __global__ void exl3_recon_had_k2_cb2(half*, const uint16_t*, const half*, const half*, int) {}
// W_orig fp16 [k, n] = diag(suh) H128 W_inner H128 diag(svh); grid (n/128, k/128), block 256 (test_exl3_m1.py)
extern "C" int exl3s_recon_had(half* out, const uint16_t* packed, const half* suh, const half* svh, int k, int n,
                               int K, cudaStream_t s) {
    if (K == 2) exl3_recon_had_k2_cb2<<<dim3(n / 128, k / 128), 256, 0, s>>>(out, packed, suh, svh, n / 16);
    else exl3_recon_had_k3_cb2<<<dim3(n / 128, k / 128), 256, 0, s>>>(out, packed, suh, svh, n / 16);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
