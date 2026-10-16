// cuBLAS subset for Strata on tgcu: cublasGemmEx (OP_T x OP_N, bf16/f16 in, f32/f16 out) through the shim's WMMA kernels.
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
extern "C" int tgcu_builtin_launch(const char* name, unsigned gx, unsigned gy, unsigned bx, void** args, cudaStream_t st);

struct tgcublasContext { cudaStream_t stream = nullptr; };

extern "C" {
const char* cublasGetStatusString(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? "CUBLAS_STATUS_SUCCESS" : "CUBLAS_STATUS_ERROR"; }
cublasStatus_t cublasCreate_v2(cublasHandle_t* h) { *h = new tgcublasContext; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDestroy_v2(cublasHandle_t h) { delete h; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetStream_v2(cublasHandle_t h, cudaStream_t s) { h->stream = s; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t, void*, size_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) { return CUBLAS_STATUS_SUCCESS; }

cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void* alpha,
                            const void* A, cudaDataType at, int lda, const void* B, cudaDataType bt, int ldb, const void* beta, void* C,
                            cudaDataType ct, int ldc, cublasComputeType_t comp, cublasGemmAlgo_t) {
  if (ta != CUBLAS_OP_T || tb != CUBLAS_OP_N || at != bt || comp != CUBLAS_COMPUTE_32F) {
    fprintf(stderr, "[tgcublas] unsupported GemmEx ta=%d tb=%d at=%d bt=%d ct=%d comp=%d\n", ta, tb, at, bt, ct, comp);
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
  const char* name = nullptr;
  if (at == CUDA_R_16BF && ct == CUDA_R_32F) name = "tgcu_gemm_tn_bf16_f32";
  else if (at == CUDA_R_16F && ct == CUDA_R_32F) name = "tgcu_gemm_tn_f16_f32";
  else if (at == CUDA_R_16F && ct == CUDA_R_16F) name = "tgcu_gemm_tn_f16_f16";
  if (!name) { fprintf(stderr, "[tgcublas] unsupported types at=%d ct=%d\n", at, ct); return CUBLAS_STATUS_NOT_SUPPORTED; }
  if (m <= 0 || n <= 0) return CUBLAS_STATUS_SUCCESS;
  float al = *(const float*) alpha, be = *(const float*) beta;
  void* args[] = {&m, &n, &k, &al, (void*) &A, &lda, (void*) &B, &ldb, &be, &C, &ldc};
  // the cp.async kernel (128 x 128 tiles, ~50 TFLOP/s on the 3090) needs 16-byte rows; the plain one takes the rest
  static const bool fast_on = [] { const char* e = getenv("TGCU_GEMM2"); return !(e && e[0] == '0'); }();
  const bool fast = fast_on && k % 8 == 0 && lda % 8 == 0 && ldb % 8 == 0 && ((uintptr_t) A % 16) == 0 && ((uintptr_t) B % 16) == 0;
  if (fast) {
    char fast_name[64];
    snprintf(fast_name, sizeof fast_name, "tgcu_gemm2_%s", name + strlen("tgcu_gemm_"));
    return tgcu_builtin_launch(fast_name, (unsigned) ((m + 127) / 128), (unsigned) ((n + 127) / 128), 256, args, h->stream) == 0
               ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
  }
  return tgcu_builtin_launch(name, (unsigned) ((m + 63) / 64), (unsigned) ((n + 63) / 64), 128, args, h->stream) == 0 ? CUBLAS_STATUS_SUCCESS
                                                                                                         : CUBLAS_STATUS_EXECUTION_FAILED;
}
cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int,
                              const float*, int, const float*, float*, int) {
  fprintf(stderr, "[tgcublas] cublasSgemm is not implemented (Pascal-only path)\n");
  return CUBLAS_STATUS_NOT_SUPPORTED;
}
}
