// Kernels the runtime itself needs: memset and device-to-device copies, so they can be queued and captured like launches.
#include <cstdint>
extern "C" __global__ void tgcu_memset8(uint8_t* p, uint32_t v, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x;
  uint32_t v4 = (v & 0xff) * 0x01010101u;
  if (((uintptr_t)p & 3) == 0) {
    for (uint64_t j = i; j < n / 4; j += stride) ((uint32_t*)p)[j] = v4;
    for (uint64_t j = (n / 4) * 4 + i; j < n; j += stride) p[j] = (uint8_t)v;
  } else for (uint64_t j = i; j < n; j += stride) p[j] = (uint8_t)v;
}
extern "C" __global__ void tgcu_memset32(uint32_t* p, uint32_t v, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) p[i] = v;
}
extern "C" __global__ void tgcu_memcpy(uint8_t* dst, const uint8_t* src, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x;
  if ((((uintptr_t)dst | (uintptr_t)src) & 15) == 0) {
    for (uint64_t j = i; j < n / 16; j += stride) ((uint4*)dst)[j] = ((const uint4*)src)[j];
    for (uint64_t j = (n / 16) * 16 + i; j < n; j += stride) dst[j] = src[j];
  } else for (uint64_t j = i; j < n; j += stride) dst[j] = src[j];
}
// 2D copy: rows of `width` bytes with independent pitches
extern "C" __global__ void tgcu_memcpy2d(uint8_t* dst, uint64_t dpitch, const uint8_t* src, uint64_t spitch, uint64_t width, uint64_t height) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x;
  for (; i < width * height; i += stride) { uint64_t r = i / width, c = i % width; dst[r * dpitch + c] = src[r * spitch + c]; }
}

// cuBLAS stand-in: C(MxN, col-major, ldc) = alpha * A^T B + beta * C with A (K x M, lda) and B (K x N, ldb) column-major,
// i.e. both operands have K contiguous (cublasGemmEx OP_T, OP_N). 64x64 tile per 128-thread block, WMMA 16x16x16, fp32 accum.
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using namespace nvcuda;
template <typename TI, typename TO>
__device__ void gemm_tn(int M, int N, int K, float alpha, const TI* __restrict__ A, int lda, const TI* __restrict__ B, int ldb,
                        float beta, TO* __restrict__ C, int ldc) {
  constexpr int BM = 64, BN = 64, BK = 32, PAD = 8;
  __shared__ __align__(32) TI As[BM][BK + PAD];
  __shared__ __align__(32) TI Bs[BN][BK + PAD];
  __shared__ __align__(32) float Cs[BN][BM + 4];   // column-major tile: Cs[n][m]
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, warp = threadIdx.x / 32;
  const int wm = (warp & 1) * 32, wn = (warp >> 1) * 32;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int e = threadIdx.x; e < BM * BK; e += blockDim.x) {
      const int r = e / BK, c = e % BK, gm = m0 + r, gn = n0 + r, gk = k0 + c;
      As[r][c] = (gm < M && gk < K) ? A[(size_t) gm * lda + gk] : TI(0.0f);
      Bs[r][c] = (gn < N && gk < K) ? B[(size_t) gn * ldb + gk] : TI(0.0f);
    }
    __syncthreads();
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, TI, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, TI, wmma::col_major> b[2];
      for (int i = 0; i < 2; i++) wmma::load_matrix_sync(a[i], &As[wm + i * 16][kk], BK + PAD);
      for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], &Bs[wn + j * 16][kk], BK + PAD);
      for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
    wmma::store_matrix_sync(&Cs[wn + j * 16][wm + i * 16], acc[i][j], BM + 4, wmma::mem_col_major);
  __syncthreads();
  for (int e = threadIdx.x; e < BM * BN; e += blockDim.x) {
    const int r = e % BM, c = e / BM, gm = m0 + r, gn = n0 + c;
    if (gm < M && gn < N) {
      float v = alpha * Cs[c][r];
      TO* p = C + (size_t) gn * ldc + gm;
      if (beta != 0.0f) v += beta * (float) *p;
      *p = (TO) v;
    }
  }
}
extern "C" __global__ void __launch_bounds__(128) tgcu_gemm_tn_bf16_f32(int M, int N, int K, float alpha, const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* B, int ldb, float beta, float* C, int ldc) { gemm_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
extern "C" __global__ void __launch_bounds__(128) tgcu_gemm_tn_f16_f32(int M, int N, int K, float alpha, const __half* A, int lda,
    const __half* B, int ldb, float beta, float* C, int ldc) { gemm_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
extern "C" __global__ void __launch_bounds__(128) tgcu_gemm_tn_f16_f16(int M, int N, int K, float alpha, const __half* A, int lda,
    const __half* B, int ldb, float beta, __half* C, int ldc) { gemm_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }

// ---- the fast GEMM: C (M x N, column-major, ldc) = alpha * A^T B + beta * C with A (M rows of K, lda) and B (N rows of K,
// ldb), both K-contiguous. 128 x 128 x 32 tiles, 8 warps of 64 x 32, 16-byte cp.async loads double-buffered in shared
// memory (rows past M / N and columns past K zero-filled). Needs K, lda, ldb multiples of 8 and 16-byte aligned A, B.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
template <typename TI, typename TO>
__device__ void gemm2_tn(int M, int N, int K, float alpha, const TI* __restrict__ A, int lda, const TI* __restrict__ B, int ldb,
                         float beta, TO* __restrict__ C, int ldc) {
  constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;
  __shared__ __align__(128) TI sm[2 * (BM + BN) * LD];       // 40 KB: [stage][A rows | B rows][LD]
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int wm = (warp & 1) * 64, wn = (warp >> 1) * 32;
  auto As = [&](int s) { return sm + s * (BM + BN) * LD; };
  auto Bs = [&](int s) { return sm + s * (BM + BN) * LD + BM * LD; };
  auto load = [&](int s, int k0) {
    #pragma unroll
    for (int i = 0; i < 2; i++) {                             // 512 chunks of 8 elements per operand, 2 per thread
      const int c = tid + i * 256, r = c / 4, kc = (c % 4) * 8, gk = k0 + kc;
      const int gm = m0 + r, gn = n0 + r;
      cp_async16(As(s) + r * LD + kc, (gm < M && gk < K) ? A + (size_t) gm * lda + gk : A, gm < M && gk < K);
      cp_async16(Bs(s) + r * LD + kc, (gn < N && gk < K) ? B + (size_t) gn * ldb + gk : B, gn < N && gk < K);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
  #pragma unroll
  for (int i = 0; i < 4; i++)
    #pragma unroll
    for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);
  const int KT = (K + BK - 1) / BK;
  load(0, 0);
  for (int kt = 0; kt < KT; kt++) {
    if (kt + 1 < KT) load((kt + 1) & 1, (kt + 1) * BK);
    else asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    const TI* as = As(kt & 1);
    const TI* bs = Bs(kt & 1);
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, TI, wmma::row_major> a[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, TI, wmma::col_major> b[2];
      #pragma unroll
      for (int i = 0; i < 4; i++) wmma::load_matrix_sync(a[i], as + (wm + i * 16) * LD + kk, LD);
      #pragma unroll
      for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], bs + (wn + j * 16) * LD + kk, LD);
      #pragma unroll
      for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  // epilogue: each warp stages a 16 x 16 fragment in shared memory (column-major) and writes it with alpha / beta
  float* scratch = reinterpret_cast<float*>(sm) + warp * 256;
  #pragma unroll
  for (int i = 0; i < 4; i++)
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      wmma::store_matrix_sync(scratch, acc[i][j], 16, wmma::mem_col_major);
      __syncwarp();
      #pragma unroll
      for (int e = lane; e < 256; e += 32) {
        const int r = e % 16, c = e / 16, gm = m0 + wm + i * 16 + r, gn = n0 + wn + j * 16 + c;
        if (gm < M && gn < N) {
          float v = alpha * scratch[c * 16 + r];
          TO* p = C + (size_t) gn * ldc + gm;
          if (beta != 0.0f) v += beta * (float) *p;
          *p = (TO) v;
        }
      }
      __syncwarp();
    }
}
extern "C" __global__ void __launch_bounds__(256) tgcu_gemm2_tn_bf16_f32(int M, int N, int K, float alpha, const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* B, int ldb, float beta, float* C, int ldc) { gemm2_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
extern "C" __global__ void __launch_bounds__(256) tgcu_gemm2_tn_f16_f32(int M, int N, int K, float alpha, const __half* A, int lda,
    const __half* B, int ldb, float beta, float* C, int ldc) { gemm2_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
extern "C" __global__ void __launch_bounds__(256) tgcu_gemm2_tn_f16_f16(int M, int N, int K, float alpha, const __half* A, int lda,
    const __half* B, int ldb, float beta, __half* C, int ldc) { gemm2_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
