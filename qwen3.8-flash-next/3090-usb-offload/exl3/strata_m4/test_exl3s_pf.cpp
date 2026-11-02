// M4 prefill GPU test of exl3s_prefill_moe (libstrata_exl3): reads a case from test_exl3s_pf.py, runs the call, writes Dm
// back, times repeated calls.  Owns the card through the tgcu libcudart; run from test_exl3s_pf.py.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static std::vector<char> slurp(const std::string& p) {
    std::vector<char> v; FILE* f = fopen(p.c_str(), "rb"); if (!f) return v;
    fseek(f, 0, SEEK_END); v.resize(ftell(f)); fseek(f, 0, SEEK_SET);
    if (fread(v.data(), 1, v.size(), f) != v.size()) v.clear(); fclose(f); return v;
}
// bounce buffer: write() straight from a buffer the shim touched (D2H target) can come out empty (see test_exl3s.cpp)
static bool spit(const std::string& p, const void* d, size_t n) {
    FILE* f = fopen(p.c_str(), "wb"); if (!f) return false;
    static char bounce[1 << 16]; bool ok = true;
    for (size_t o = 0; o < n && ok; o += sizeof bounce) {
        const size_t k = n - o < sizeof bounce ? n - o : sizeof bounce;
        std::memcpy(bounce, (const char*) d + o, k); ok = fwrite(bounce, 1, k, f) == k;
    }
    return fclose(f) == 0 && ok;
}
static void* up(const std::vector<char>& v) { void* d = nullptr; cudaMalloc(&d, v.size()); cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice); return d; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_exl3s_pf LIB DIR\n"); return 2; }
    const std::string dir = argv[2];
    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto create = (void* (*)(int, int, const unsigned*, char*, int)) dlsym(h, "exl3s_pf_create");
    auto vram = (size_t (*)(void*)) dlsym(h, "exl3s_pf_vram_bytes");
    auto run = (int (*)(void*, const uint16_t*, float*, int, const int32_t*, const int32_t*, const unsigned long long*,
                        int, const unsigned*, cudaStream_t)) dlsym(h, "exl3s_prefill_moe");
    if (!create || !vram || !run) { fprintf(stderr, "missing symbols\n"); return 1; }
    int n_exp, rows, reps; unsigned long long blob; unsigned off9[9];
    FILE* m = fopen((dir + "/meta.txt").c_str(), "r");
    if (!m || fscanf(m, "%d %d %llu %d", &n_exp, &rows, &blob, &reps) != 4) return 3;
    for (int i = 0; i < 9; ++i) if (fscanf(m, "%u", &off9[i]) != 1) return 3;
    int bits = 3;
    if (fscanf(m, "%d", &bits) != 1) bits = 3;   // trellis bits (mixed precision: 2)
    fclose(m);
    char err[256] = {};
    void* ctx = create(2048 * 10, 256, off9, err, sizeof err);
    if (!ctx) { fprintf(stderr, "%s\n", err); return 4; }
    auto slots = slurp(dir + "/slots.bin"), xs = slurp(dir + "/xs.bin"), dm0 = slurp(dir + "/dm0.bin"),
         cnt = slurp(dir + "/cnt.bin"), off = slurp(dir + "/off.bin");
    if (slots.size() != (size_t) n_exp * blob || cnt.size() != (size_t) n_exp * 4) return 5;
    void *dS = up(slots), *dX = up(xs), *dD = up(dm0);
    std::vector<unsigned long long> bp((size_t) n_exp);
    for (int i = 0; i < n_exp; ++i) bp[(size_t) i] = (unsigned long long) dS + (unsigned long long) i * blob;
    cudaStream_t s; cudaStreamCreate(&s);
    int r = run(ctx, (const uint16_t*) dX, (float*) dD, n_exp, (const int32_t*) cnt.data(), (const int32_t*) off.data(), bp.data(), bits, nullptr, s);
    if (r || cudaStreamSynchronize(s)) { fprintf(stderr, "exl3s_prefill_moe %d\n", r); return 6; }
    std::vector<char> out(dm0.size());
    cudaMemcpy(out.data(), dD, out.size(), cudaMemcpyDeviceToHost);
    spit(dir + "/dm.bin", out.data(), out.size());
    r = run(ctx, (const uint16_t*) dX, (float*) dD, n_exp, (const int32_t*) cnt.data(), (const int32_t*) off.data(), bp.data(), bits, nullptr, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(out.data(), dD, out.size(), cudaMemcpyDeviceToHost);
    spit(dir + "/dm_rep.bin", out.data(), out.size());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
        run(ctx, (const uint16_t*) dX, (float*) dD, n_exp, (const int32_t*) cnt.data(), (const int32_t*) off.data(), bp.data(), bits, nullptr, s);
    cudaStreamSynchronize(s);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
    printf("{\"rows\": %d, \"experts\": %d, \"us_per_call\": %.1f, \"vram_bytes\": %zu}\n", rows, n_exp, us, vram(ctx));
    return 0;
}
