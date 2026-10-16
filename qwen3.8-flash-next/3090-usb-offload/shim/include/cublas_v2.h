// Minimal cuBLAS API for Strata on tgcu: the GEMMs its prefill uses, served by tgcublas (WMMA kernels in the shim).
#pragma once
#include <cuda_runtime_api.h>
#include <library_types.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct tgcublasContext* cublasHandle_t;
typedef enum { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_INITIALIZED = 1, CUBLAS_STATUS_ALLOC_FAILED = 3, CUBLAS_STATUS_INVALID_VALUE = 7,
               CUBLAS_STATUS_ARCH_MISMATCH = 8, CUBLAS_STATUS_EXECUTION_FAILED = 13, CUBLAS_STATUS_INTERNAL_ERROR = 14,
               CUBLAS_STATUS_NOT_SUPPORTED = 15 } cublasStatus_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;
typedef enum { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TENSOR_OP_MATH = 1, CUBLAS_PEDANTIC_MATH = 2, CUBLAS_TF32_TENSOR_OP_MATH = 3 } cublasMath_t;
typedef enum { CUBLAS_COMPUTE_16F = 64, CUBLAS_COMPUTE_32F = 68, CUBLAS_COMPUTE_32F_FAST_16F = 74, CUBLAS_COMPUTE_32F_FAST_16BF = 75,
               CUBLAS_COMPUTE_32F_FAST_TF32 = 77 } cublasComputeType_t;
typedef enum { CUBLAS_GEMM_DEFAULT = -1, CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99 } cublasGemmAlgo_t;
const char* cublasGetStatusString(cublasStatus_t status);
cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace, size_t bytes);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb,
                            const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                              const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasSetStream cublasSetStream_v2
#define cublasSetWorkspace cublasSetWorkspace_v2
#define cublasSgemm cublasSgemm_v2
#ifdef __cplusplus
}
#endif
