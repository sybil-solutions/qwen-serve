// Force-included into host-only CUDA compiles (clang --cuda-host-only). Apple clang reports the macOS SDK version as the
// CUDA SDK version, so <<<>>> lowers to the legacy launch API, which CUDA 12 headers no longer declare.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
cudaError_t cudaLaunch(const void *func);
#ifdef __cplusplus
}
#endif
