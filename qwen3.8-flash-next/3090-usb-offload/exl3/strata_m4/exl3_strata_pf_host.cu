// M4 prefill: host side of ../exl3_strata_pf.fatbin (exllamav3 exl3_moe_kernel, 3 bpw MUL1, sm_86) and its C API for
// Strata's prompt path.  Same scheme as exl3_strata_host.cu: empty kernels = launch stubs registered under the fatbin's
// names.  The launcher is exllamav3 exl3_moe() (151539c, quant/exl3_moe.cu, CUDA branch) without torch, plus
// BlockSparseMLP's row tiers (MTILE_T1 16, MTILE_T2 32, fused rows 256) and its deterministic slot mode.
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include "../vendor/exllamav3_ext/quant/exl3_moe_common.cuh"
#include "../vendor/exllamav3_ext/quant/exl3_devctx.cuh"   // MAX_TILES_C, MOE_SCHED_OFFSET, MOE_MAX_GROUPS

// EXL3_MOE_KERNEL_ARGS with `long` for int64_t: the fatbin was built on Linux (int64_t = long, mangled `l`), and on macOS
// int64_t is `long long` (`x`), so the stubs must spell the device names' type to register under them (same 64 bits).
#define EXL3_MOE_STUB_ARGS                                                                                             \
    const half*, half*, half*, half*, half*, float*, const uint16_t**, const half**, const half**, const uint16_t**,  \
    const half**, const half**, const uint16_t**, const half**, const half**, const long*, const long*, const half*,  \
    const int, const int, const int, const int, const int, const int, const float, const int, const int, const int,   \
    const int, int*, float*, const long*, const int, const int
static_assert(sizeof(long) == sizeof(int64_t), "long must be 64-bit");
template <int t_bits, int MOE_TILESIZE_N, int cb, int M_TILE, bool t_half, bool PIPE>
__global__ void exl3_moe_kernel(EXL3_MOE_STUB_ARGS) {}
template __global__ void exl3_moe_kernel<3, 128, 2, 16, false, false>(EXL3_MOE_STUB_ARGS);
template __global__ void exl3_moe_kernel<3, 128, 2, 32, false, false>(EXL3_MOE_STUB_ARGS);
template __global__ void exl3_moe_kernel<3, 128, 2, 64, false, false>(EXL3_MOE_STUB_ARGS);
template __global__ void exl3_moe_kernel<2, 128, 2, 16, false, false>(EXL3_MOE_STUB_ARGS);
template __global__ void exl3_moe_kernel<2, 128, 2, 32, false, false>(EXL3_MOE_STUB_ARGS);
template __global__ void exl3_moe_kernel<2, 128, 2, 64, false, false>(EXL3_MOE_STUB_ARGS);
extern "C" __global__ void exl3s_pf_tables(const unsigned long long*, int, unsigned, unsigned, unsigned, unsigned, unsigned,
                                           unsigned, unsigned, unsigned, unsigned, unsigned long long*) {}

namespace {
typedef void (*MoeKernel)(EXL3_MOE_STUB_ARGS);
constexpr int kH = 2560, kI = 640, kRowsCap = 256;   // fused rows per entry (upstream FUSED_ROWS_WIDE); bigger experts split
constexpr int kTier1 = 16, kTier2 = 32;              // upstream MTILE_T1 / MTILE_T2
struct Pf {
    int sms = 82, groups_max = 10, smem = 0, max_entries = 0, max_rows = 0;
    unsigned field_off[9] = {};
    half *ts_g = nullptr, *ts_u = nullptr, *ti_g = nullptr, *ti_u = nullptr;   // [groups_max][kRowsCap][H or I]
    int* locks = nullptr;                    // MOE_SCHED_OFFSET + MOE_SCHED_INTS ints, zeroed (self-resetting)
    unsigned char* dev_tab = nullptr;        // ent u64[E] | count i64[E+1] | fused_base i64[E] | token_sorted i64[R] | ptrs u64[9][E] | weights f16[R]
    std::vector<unsigned char> host_tab;
    bool attr_set = false;
};
size_t tab_bytes(int E, int R) { return (size_t) E * 8 + (size_t) (E + 1) * 8 + (size_t) E * 8 + (size_t) R * 8 + (size_t) 9 * E * 8 + (size_t) R * 2 + 64; }
}  // namespace

// Context for the prompt arm: max_rows = the most expert rows one call carries (chunk x top-k), max_experts = experts per
// call before splitting.  VRAM: 2 x groups x 256 x (2560 + 640) x 2 B temp (~33 MB at 10 groups) + 4.2 MB locks + tables.
extern "C" void* exl3s_pf_create(int max_rows, int max_experts, const unsigned* field_off, char* err, int errlen) {
    auto fail = [&](const char* m) { std::snprintf(err, (size_t) errlen, "exl3s_pf_create: %s", m); return (void*) nullptr; };
    Pf* p = new Pf;
    std::memcpy(p->field_off, field_off, sizeof p->field_off);
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&p->sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || p->sms <= 0) p->sms = 82;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess || optin <= 0) optin = 99 * 1024;
    p->smem = std::min(optin, 90 * 1024);                                  // DevCtx::get_smem_request
    p->groups_max = std::min(p->sms / MOE_SMS_PER_EXPERT, MOE_MAX_GROUPS); // exl3_moe_max_concurrency (CUDA)
    p->max_rows = max_rows;
    p->max_entries = max_experts + max_rows / kRowsCap + 1;
    const size_t g = (size_t) p->groups_max * kRowsCap;
    const size_t lock_ints = (size_t) MOE_SCHED_OFFSET + MOE_SCHED_INTS;
    if (cudaMalloc((void**) &p->ts_g, g * kH * 2) || cudaMalloc((void**) &p->ts_u, g * kH * 2) ||
        cudaMalloc((void**) &p->ti_g, g * kI * 2) || cudaMalloc((void**) &p->ti_u, g * kI * 2) ||
        cudaMalloc((void**) &p->locks, lock_ints * 4) || cudaMemset(p->locks, 0, lock_ints * 4) ||
        cudaMalloc((void**) &p->dev_tab, tab_bytes(p->max_entries, max_rows)))
        return fail("VRAM");
    p->host_tab.resize(tab_bytes(p->max_entries, max_rows));
    return p;
}

extern "C" size_t exl3s_pf_vram_bytes(void* ctx) {
    Pf* p = (Pf*) ctx;
    const size_t g = (size_t) p->groups_max * kRowsCap;
    return g * (kH + kI) * 4 + ((size_t) MOE_SCHED_OFFSET + MOE_SCHED_INTS) * 4 + tab_bytes(p->max_entries, p->max_rows);
}

// One layer's routed experts in one call.  Xs: fp16 rows [*, 2560]; expert i's cnt[i] rows start at row off[i];
// blob[i]: device address of its EXL3 blob (VRAM slot or staged ring copy).  Writes Dm[off[i] + r] = expert_i(Xs[off[i]
// + r]) (fp32, unweighted, every listed row overwritten; others untouched).  cnt/off/blob are HOST arrays.  Launches:
// one table kernel + one exl3_moe_kernel per row tier present (<= 3), after one host -> device table upload.
extern "C" int exl3s_prefill_moe(void* ctx, const uint16_t* Xs, float* Dm, int n_experts, const int32_t* cnt,
                                 const int32_t* off, const unsigned long long* blob, int K, const unsigned* field_off,
                                 cudaStream_t s) {
    Pf* p = (Pf*) ctx;
    // entries: experts split into <= kRowsCap-row pieces, grouped by tier (largest tier first, as upstream launches)
    struct Ent { unsigned long long b; int off, cnt; };
    std::vector<Ent> ent;
    ent.reserve((size_t) p->max_entries);
    int rows = 0;
    for (int i = 0; i < n_experts; ++i)
        for (int r = 0; r < cnt[i]; r += kRowsCap) {
            ent.push_back({blob[i], off[i] + r, std::min(kRowsCap, cnt[i] - r)});
            rows += ent.back().cnt;
        }
    const int E = (int) ent.size();
    if (E == 0) return 0;
    if (E > p->max_entries || rows > p->max_rows) return -1;
    // host table (layout as dev_tab)
    unsigned char* h = p->host_tab.data();
    unsigned long long* t_ent = (unsigned long long*) h;
    int64_t* t_cnt = (int64_t*) (t_ent + E);
    int64_t* t_base = t_cnt + E + 1;
    int64_t* t_tok = t_base + E;
    unsigned long long* t_ptr = (unsigned long long*) (t_tok + rows);
    uint16_t* t_w = (uint16_t*) (t_ptr + 9 * (size_t) E);
    int at = 0, n_t[3] = {0, 0, 0};
    for (int i = 0; i < E; ++i) {
        t_ent[i] = ent[i].b; t_cnt[i] = ent[i].cnt; t_base[i] = ent[i].off;
        for (int r = 0; r < ent[i].cnt; ++r) t_tok[at + r] = ent[i].off + r;   // token_sorted: the row itself
        at += ent[i].cnt;
        ++n_t[ent[i].cnt > kTier2 ? 2 : ent[i].cnt > kTier1 ? 1 : 0];
    }
    t_cnt[E] = 0;
    for (int r = 0; r < rows; ++r) t_w[r] = 0x3C00;   // fp16 1.0: the kernel scales by the weight; Strata applies it later
    const size_t bytes = (size_t) ((unsigned char*) (t_w + rows) - h);
    if (cudaMemcpyAsync(p->dev_tab, h, bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return -2;
    unsigned char* d = p->dev_tab;
    const unsigned long long* d_ent = (const unsigned long long*) d;
    const int64_t* d_cnt = (const int64_t*) (d_ent + E);
    const int64_t* d_base = d_cnt + E + 1;
    const int64_t* d_tok = d_base + E;
    unsigned long long* d_ptr = (unsigned long long*) (d_tok + rows);
    const half* d_w = (const half*) (d_ptr + 9 * (size_t) E);
    const unsigned* f = field_off ? field_off : p->field_off;   // per layer (mixed K)
    exl3s_pf_tables<<<(E + 255) / 256, 256, 0, s>>>(d_ent, E, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], d_ptr);
    MoeKernel kern3[3] = {exl3_moe_kernel<3, 128, 2, 16, false, false>, exl3_moe_kernel<3, 128, 2, 32, false, false>,
                          exl3_moe_kernel<3, 128, 2, 64, false, false>};
    MoeKernel kern2[3] = {exl3_moe_kernel<2, 128, 2, 16, false, false>, exl3_moe_kernel<2, 128, 2, 32, false, false>,
                          exl3_moe_kernel<2, 128, 2, 64, false, false>};
    if (K != 2 && K != 3) return -1;
    MoeKernel* kern = K == 2 ? kern2 : kern3;
    if (!p->attr_set) {
        for (MoeKernel k : kern3)
            if (cudaFuncSetAttribute((const void*) k, cudaFuncAttributeMaxDynamicSharedMemorySize, p->smem) != cudaSuccess) return -3;
        for (MoeKernel k : kern2)
            if (cudaFuncSetAttribute((const void*) k, cudaFuncAttributeMaxDynamicSharedMemorySize, p->smem) != cudaSuccess) return -3;
        p->attr_set = true;
    }
    const int lo[3] = {1, kTier1 + 1, kTier2 + 1}, hi[3] = {kTier1, kTier2, kRowsCap};
    const uint16_t** gt = (const uint16_t**) (d_ptr + 0 * (size_t) E); const half** gs = (const half**) (d_ptr + 1 * (size_t) E);
    const half** gv = (const half**) (d_ptr + 2 * (size_t) E);       const uint16_t** ut = (const uint16_t**) (d_ptr + 3 * (size_t) E);
    const half** us = (const half**) (d_ptr + 4 * (size_t) E);       const half** uv = (const half**) (d_ptr + 5 * (size_t) E);
    const uint16_t** dt = (const uint16_t**) (d_ptr + 6 * (size_t) E); const half** ds = (const half**) (d_ptr + 7 * (size_t) E);
    const half** dv = (const half**) (d_ptr + 8 * (size_t) E);
    for (int tier = 2; tier >= 0; --tier) {
        if (!n_t[tier]) continue;
        // exl3_moe(): groups = min(concurrency, num_active), width = min(sms / groups, 32); all co-resident
        const int groups = std::min(p->groups_max, n_t[tier]);
        const int width = std::min(p->sms / groups, MOE_MAX_SMS_PER_EXPERT);
        const half* hs = (const half*) Xs;
        int hidden = kH, inter = kI, ne = E, nept = 1, mtpe = kRowsCap, conc = groups, act = MOE_ACT_SILU, K3 = 2 * K;   // half-bit units (bits_k.cuh)
        float act_limit = 0.0f;
        int c_lo = lo[tier], c_hi = hi[tier];
        float* out_state = Dm;   // unused in slot mode (output_scratch set)
        void* args[] = {&hs, &p->ts_g, &p->ts_u, &p->ti_g, &p->ti_u, &out_state, &gt, &gs, &gv, &ut, &us, &uv, &dt, &ds, &dv,
                        &d_cnt, &d_tok, &d_w, &hidden, &inter, &ne, &nept, &mtpe, &conc, &act_limit, &act, &K3, &K3, &K3,
                        &p->locks, &Dm, &d_base, &c_lo, &c_hi};
        if (cudaLaunchKernel((const void*) kern[tier], dim3(width, 1, groups), dim3(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16),
                             args, (size_t) p->smem, s) != cudaSuccess)
            return -4;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -5;
}
