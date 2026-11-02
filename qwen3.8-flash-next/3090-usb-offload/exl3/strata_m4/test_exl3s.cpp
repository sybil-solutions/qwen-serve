// M4 GPU test of libstrata_exl3 (the C API Strata calls): reads a case written by test_exl3s.py, runs exl3s_dec_pre /
// exl3s_dec_post (and exl3s_dequant on slot 0), writes the outputs back for the Python side to score.  Owns the card
// through the tgcu libcudart; run from test_exl3s.py.
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
// through a stack bounce buffer: a write() straight from a host buffer the tgcu shim has touched (D2H target) came out
// empty for 6.5 MB buffers (the syscall cannot take the shim's page-protection faults); a user-space memcpy can
static bool spit(const std::string& p, const void* d, size_t n) {
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) return false;
    static char bounce[1 << 16];
    bool ok = true;
    for (size_t o = 0; o < n && ok; o += sizeof bounce) {
        const size_t k = n - o < sizeof bounce ? n - o : sizeof bounce;
        std::memcpy(bounce, (const char*) d + o, k);
        ok = fwrite(bounce, 1, k, f) == k;
    }
    return fclose(f) == 0 && ok;
}
static void* up(const std::vector<char>& v) { void* d = nullptr; cudaMalloc(&d, v.size()); cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice); return d; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_exl3s LIB DIR\n"); return 2; }
    const std::string dir = argv[2];
    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto create = (void* (*)(int, int, int, int, int, const unsigned*, unsigned long long, char*, int)) dlsym(h, "exl3s_create");
    auto pre = (int (*)(void*, int, const float*, int, const int32_t*, const float*, int, const int32_t*, const unsigned long long*,
                        unsigned long long, long long, int, const unsigned*, cudaStream_t)) dlsym(h, "exl3s_dec_pre");
    auto post = (int (*)(void*, int, float*, int, int, cudaStream_t)) dlsym(h, "exl3s_dec_post");
    auto dequant = (int (*)(void*, const uint8_t*, uint16_t*, uint16_t*, int, const unsigned*, cudaStream_t)) dlsym(h, "exl3s_dequant");
    int n_expert, rows, n_slots, reps; unsigned long long blob; unsigned off[9];
    FILE* m = fopen((dir + "/meta.txt").c_str(), "r");
    if (!m || fscanf(m, "%d %d %d %llu %d", &n_expert, &rows, &n_slots, &blob, &reps) != 5) return 3;
    for (int i = 0; i < 9; ++i) if (fscanf(m, "%u", &off[i]) != 1) return 3;
    int bits = 3;
    if (fscanf(m, "%d", &bits) != 1) bits = 3;   // trellis bits (mixed precision: 2)
    fclose(m);
    char err[256] = {};
    void* ctx = create(n_expert, 2560, 640, 10, 8, off, blob, err, sizeof err);
    if (!ctx) { fprintf(stderr, "%s\n", err); return 4; }
    auto blobs = slurp(dir + "/slots.bin"), res = slurp(dir + "/res.bin"), x = slurp(dir + "/x.bin"),
         ids = slurp(dir + "/ids.bin"), w = slurp(dir + "/w.bin"), bo = slurp(dir + "/bo.bin");
    if (blobs.size() != (size_t) n_slots * blob || res.size() != (size_t) n_expert * 4) return 5;
    void *dB = up(blobs), *dR = up(res), *dX = up(x), *dI = up(ids), *dW = up(w), *dO = up(bo);
    cudaStream_t s; cudaStreamCreate(&s);
    for (int grp = 0; grp < 2; ++grp) {   // both scratch sets; grp 1 recomputes from the same bo copy
        if (grp == 1) cudaMemcpy(dO, bo.data(), bo.size(), cudaMemcpyHostToDevice);
        if (pre(ctx, grp, (const float*) dX, 2560, (const int32_t*) dI, (const float*) dW, rows, (const int32_t*) dR, nullptr,
                (unsigned long long) dB, (long long) blob, bits, nullptr, s) || post(ctx, grp, (float*) dO, 2560, rows, s)) {
            fprintf(stderr, "pre/post failed\n"); return 6;
        }
        if (cudaStreamSynchronize(s)) { fprintf(stderr, "sync failed\n"); return 7; }
        std::vector<char> out(bo.size());
        cudaMemcpy(out.data(), dO, out.size(), cudaMemcpyDeviceToHost);
        spit(dir + "/bo_out" + std::to_string(grp) + ".bin", out.data(), out.size());
    }
    // repeat (bit-identical?) + timing of pre+post chains (no graph: the C-API launch floor applies)
    cudaMemcpy(dO, bo.data(), bo.size(), cudaMemcpyHostToDevice);
    pre(ctx, 0, (const float*) dX, 2560, (const int32_t*) dI, (const float*) dW, rows, (const int32_t*) dR, nullptr,
        (unsigned long long) dB, (long long) blob, bits, nullptr, s);
    post(ctx, 0, (float*) dO, 2560, rows, s);
    cudaStreamSynchronize(s);
    std::vector<char> out(bo.size());
    cudaMemcpy(out.data(), dO, out.size(), cudaMemcpyDeviceToHost);
    spit(dir + "/bo_out_rep.bin", out.data(), out.size());
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        pre(ctx, 0, (const float*) dX, 2560, (const int32_t*) dI, (const float*) dW, rows, (const int32_t*) dR, nullptr,
            (unsigned long long) dB, (long long) blob, bits, nullptr, s);
        post(ctx, 0, (float*) dO, 2560, rows, s);
    }
    cudaStreamSynchronize(s);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
    // prefill dequant of slot 0
    void *dG, *dD;
    cudaMalloc(&dG, 1280 * 2560 * 2); cudaMalloc(&dD, 2560 * 640 * 2);
    if (dequant(ctx, (const uint8_t*) dB, (uint16_t*) dG, (uint16_t*) dD, bits, nullptr, s) || cudaStreamSynchronize(s)) return 8;
    std::vector<char> g(1280 * 2560 * 2), d(2560 * 640 * 2);
    cudaMemcpy(g.data(), dG, g.size(), cudaMemcpyDeviceToHost); cudaMemcpy(d.data(), dD, d.size(), cudaMemcpyDeviceToHost);
    spit(dir + "/dq_gu.bin", g.data(), g.size()); spit(dir + "/dq_d.bin", d.data(), d.size());
    printf("{\"rows\": %d, \"us_per_layer\": %.1f}\n", rows, us);
    return 0;
}
