// VRAM baseline through the tgcu shim: free/total right after context creation, and after a 1 GiB cudaMalloc
#include <cuda_runtime.h>
#include <cstdio>
int main() {
    size_t f = 0, t = 0;
    cudaFree(0);
    cudaMemGetInfo(&f, &t);
    printf("after init: free %.3f GiB of %.3f GiB\n", f / 1073741824.0, t / 1073741824.0);
    void* p = nullptr;
    cudaMalloc(&p, 1ull << 30);
    cudaMemGetInfo(&f, &t);
    printf("after 1 GiB malloc: free %.3f GiB\n", f / 1073741824.0);
    return 0;
}
