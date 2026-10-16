// M4: host side of ../exl3_strata.fatbin (decode MoE + prefill layout kernels), compiled host-only with
// -fcuda-include-gpubinary (the empty kernels below are launch stubs registered under the fatbin's names; the code
// that runs is the fatbin's).  Linked into libstrata_exl3.dylib with exl3_strata_recon_host.cu; Strata dlopens it
// (STRATA_EXL3_GPU_LIB, default ~/exl3/strata_m4/libstrata_exl3.dylib) and calls the C API at the bottom.
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include "../vendor/exllamav3_ext/quant/exl3_moe_coop.cuh"

struct Exl3PlanParams {   // == exl3_plan.cu (static_asserts there pin the layout)
    const int32_t* res; const unsigned long long* slot_off; unsigned long long cache_base; long long blob;
    unsigned long long zero_blob; unsigned int field_off[9]; int n_expert; const int32_t* ids; const float* w; int n;
    long long* tables; long long* sel; __half* rw; int* n_miss;
};
static_assert(sizeof(Exl3PlanParams) == 136, "Exl3PlanParams size");
static_assert(sizeof(MoeCoopParams) == 344, "MoeCoopParams size");

namespace exl3_moe_coop_ns {
template <int bits, int cb, bool WIDE, bool HALF> __global__ void exl3_moe_coop_a_kernel(const MoeCoopParams) {}
template <int bits, int cb, bool WIDE, bool HALF> __global__ void exl3_moe_coop_b_kernel(const MoeCoopParams) {}
template __global__ void exl3_moe_coop_a_kernel<3, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_b_kernel<3, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_a_kernel<2, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_b_kernel<2, 2, true, false>(const MoeCoopParams);
__global__ void exl3_moe_coop_rot_kernel(const MoeCoopParams) {}
}
extern "C" __global__ void exl3_plan_kernel(const Exl3PlanParams) {}
extern "C" __global__ void exl3s_f2h(const float*, int, half*, int, int) {}
extern "C" __global__ void exl3s_gu_interleave(const half*, const half*, half*, int, int) {}
extern "C" __global__ void exl3s_transpose(const half*, half*, int, int) {}
extern "C" __global__ void exl3s_mask_logits(float*, int, const int32_t*, int) {}
extern "C" int exl3s_recon_had(half* out, const uint16_t* packed, const half* suh, const half* svh, int k, int n,
                               int K, cudaStream_t s);   // exl3_strata_recon_host.cu

namespace {
constexpr int SMEM_A_FIXED = 16 * 8 * 32 * 4 + 16 * 128 * 4;   // + Hi * 2 (sm_86, 3 bpw)
constexpr int SMEM_B = 16 * 8 * 32 * 4 + 16 * 128 * 4 + 16 * 128 * 4;
struct Grp {
    long long* tables = nullptr; long long* sel = nullptr; __half* rw = nullptr; int* n_miss = nullptr;
    half *had_g = nullptr, *had_u = nullptr, *act_out = nullptr, *xh = nullptr;
    float *gu_g = nullptr, *gu_u = nullptr, *d_out = nullptr, *out = nullptr;
    int* ctr = nullptr;
    MoeCoopParams p{};
    int rows = 0, ks_a = 1, ks_b = 1, K = 3;
};
struct Ctx {
    int n_expert, H, I, topk, max_rows, slots_max, sms;
    unsigned field_off[9];
    unsigned long long blob;
    void* zero_blob = nullptr;
    Grp g[2];
    half *wg = nullptr, *wu = nullptr, *wd = nullptr;   // prefill dequant scratch
    int32_t* mask_res = nullptr;                        // a host residency row, uploaded for exl3s_mask_logits_host
};
int pick_ksplit(int sms, int base, int kslices, int max_split) {
    const int want = (sms + std::max(base, 1) - 1) / std::max(base, 1);
    return std::max(1, std::min(std::min(want, std::max(kslices / 64, 1)), max_split));
}
template <class T> bool alloc(T*& p, size_t n) { return cudaMalloc((void**) &p, n * sizeof(T)) == cudaSuccess; }
}  // namespace

extern "C" void* exl3s_create(int n_expert, int H, int I, int topk, int max_rows, const unsigned* field_off,
                              unsigned long long blob, char* err, int errlen) {
    auto fail = [&](const char* m) { std::snprintf(err, (size_t) errlen, "exl3s_create: %s", m); return (void*) nullptr; };
    if (H % 128 || I % 128 || topk < 1 || max_rows < 1 || max_rows * topk > 256) return fail("unsupported shape");
    Ctx* c = new Ctx{n_expert, H, I, topk, max_rows, max_rows * topk, 0, {}, blob};
    std::memcpy(c->field_off, field_off, sizeof c->field_off);
    int dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&c->sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || c->sms <= 0) c->sms = 82;
    if (cudaMalloc(&c->zero_blob, blob) || cudaMemset(c->zero_blob, 0, blob)) return fail("zero blob");
    const int S = c->slots_max;
    const int64_t ctr_len = exl3_moe_coop_ctr_len(S, max_rows, I, H);
    for (Grp& g : c->g) {
        if (!alloc(g.tables, (size_t) 9 * n_expert) || !alloc(g.sel, S) || !alloc(g.rw, S) || !alloc(g.n_miss, 1) ||
            !alloc(g.had_g, (size_t) S * H) || !alloc(g.had_u, (size_t) S * H) || !alloc(g.act_out, (size_t) S * I) ||
            !alloc(g.xh, (size_t) max_rows * H) || !alloc(g.gu_g, (size_t) S * I) || !alloc(g.gu_u, (size_t) S * I) ||
            !alloc(g.d_out, (size_t) S * H) || !alloc(g.out, (size_t) max_rows * H) || !alloc(g.ctr, (size_t) ctr_len))
            return fail("scratch");
        if (cudaMemset(g.ctr, 0, (size_t) ctr_len * 4) || cudaMemset(g.tables, 0, (size_t) 9 * n_expert * 8))
            return fail("memset");
    }
    if (!alloc(c->wg, (size_t) H * I) || !alloc(c->wu, (size_t) H * I) || !alloc(c->wd, (size_t) H * I) ||
        !alloc(c->mask_res, (size_t) n_expert))
        return fail("dequant scratch");
    return c;
}

// Before the router's consumers: x (fp32 rows) -> fp16, the device plan, then rot (rows > 1) and kernel A.
extern "C" int exl3s_dec_pre(void* ctx, int grp, const float* x, int x_stride, const int32_t* ids, const float* w, int rows,
                             const int32_t* res, const unsigned long long* slot_off, unsigned long long cache_base,
                             long long blob, int bits, const unsigned* field_off, cudaStream_t s) {
    Ctx* c = (Ctx*) ctx;
    if (grp < 0 || grp > 1 || rows < 1 || rows > c->max_rows) return -1;
    Grp& g = c->g[grp];
    const int H = c->H, I = c->I, K = c->topk, slots = rows * K;
    exl3s_f2h<<<(rows * H + 255) / 256, 256, 0, s>>>(x, x_stride, g.xh, rows, H);
    Exl3PlanParams pp{res, slot_off, cache_base, blob, (unsigned long long) c->zero_blob, {}, c->n_expert, ids, w, slots,
                      g.tables, g.sel, g.rw, g.n_miss};
    std::memcpy(pp.field_off, field_off ? field_off : c->field_off, sizeof pp.field_off);   // per layer (mixed K)
    if (bits != 2 && bits != 3) return -1;
    g.K = bits;
    exl3_plan_kernel<<<1, 256, 0, s>>>(pp);
    MoeCoopParams& p = g.p;
    std::memset(&p, 0, sizeof p);
    const int64_t* T = g.tables;
    const int64_t ctr_a_len = (int64_t) c->slots_max * (I / 128), ctr_b_len = (int64_t) c->max_rows * (H / 128);
    p.x = g.xh; p.x_stride = H; p.sel = g.sel; p.rw = g.rw; p.bsz = rows; p.topk = K; p.H = H; p.Hi = H; p.I = I; p.Ho = H;
    p.H_out = H; p.min_expert = -1; p.max_expert = -1;
    p.g_trellis = T + 0 * (int64_t) c->n_expert; p.g_suh = T + 1 * (int64_t) c->n_expert; p.g_svh = T + 2 * (int64_t) c->n_expert;
    p.u_trellis = T + 3 * (int64_t) c->n_expert; p.u_suh = T + 4 * (int64_t) c->n_expert; p.u_svh = T + 5 * (int64_t) c->n_expert;
    p.d_trellis = T + 6 * (int64_t) c->n_expert; p.d_suh = T + 7 * (int64_t) c->n_expert; p.d_svh = T + 8 * (int64_t) c->n_expert;
    p.act = MOE_COOP_ACT_SILU; p.gated = true;
    p.had_g = g.had_g; p.had_u = g.had_u; p.a_global = rows > 1;
    p.gu_g = g.gu_g; p.gu_u = g.gu_u; p.gu_f32 = true; p.act_out = g.act_out; p.d_out = g.d_out;
    p.ctr_a = g.ctr; p.ctr_b = g.ctr + ctr_a_len; p.ctr_a_len = (int) ctr_a_len; p.ctr_b_len = (int) ctr_b_len;
    const int base_a = slots * 2 * (I / 128), base_b = slots * (H / 128), max_split = std::max(1, c->slots_max / slots);
    g.ks_a = pick_ksplit(c->sms, base_a, H / 16, max_split);
    g.ks_b = pick_ksplit(c->sms, base_b, I / 16, max_split);
    p.ksplit_a = g.ks_a; p.ksplit_b = g.ks_b;
    p.runs = g.ctr + ctr_a_len + ctr_b_len; p.slots_max = c->slots_max; p.rows_max = c->max_rows; p.n_local = c->n_expert;
    p.out = g.out; p.out_stride = H;
    g.rows = rows;
    if (rows > 1)
        exl3_moe_coop_ns::exl3_moe_coop_rot_kernel<<<(slots * (H / 128) * 2 + 15) / 16, 512, 0, s>>>(p);
    if (bits == 2) exl3_moe_coop_ns::exl3_moe_coop_a_kernel<2, 2, true, false><<<base_a * g.ks_a, 512, H * 2 + SMEM_A_FIXED, s>>>(p);
    else exl3_moe_coop_ns::exl3_moe_coop_a_kernel<3, 2, true, false><<<base_a * g.ks_a, 512, H * 2 + SMEM_A_FIXED, s>>>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

// After Strata's combine: kernel B with bo as both the shared term and the output, i.e. bo += sum over the resident
// routed experts of rw * expert(x) (misses have rw = 0 and the zero blob).
extern "C" int exl3s_dec_post(void* ctx, int grp, float* bo, int bo_stride, int rows, cudaStream_t s) {
    Ctx* c = (Ctx*) ctx;
    if (grp < 0 || grp > 1) return -1;
    Grp& g = c->g[grp];
    if (rows != g.rows || bo_stride != c->H) return -1;
    MoeCoopParams p = g.p;
    p.out = bo; p.out_stride = bo_stride; p.sh_out = bo; p.sh_gate_w = nullptr; p.sh_gate_n = 0;
    const int base_b = rows * c->topk * (c->H / 128);
    if (g.K == 2) exl3_moe_coop_ns::exl3_moe_coop_b_kernel<2, 2, true, false><<<base_b * g.ks_b, 512, SMEM_B, s>>>(p);
    else exl3_moe_coop_ns::exl3_moe_coop_b_kernel<3, 2, true, false><<<base_b * g.ks_b, 512, SMEM_B, s>>>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

// Prefill FP16 arm: one EXL3 blob (device) -> Strata's dq_gu [2 I][H] (interleaved gate/up rows) + dq_d [H][I].
extern "C" int exl3s_dequant(void* ctx, const uint8_t* blob, uint16_t* dq_gu, uint16_t* dq_d, int K,
                             const unsigned* field_off, cudaStream_t s) {
    Ctx* c = (Ctx*) ctx;
    const unsigned* f = field_off ? field_off : c->field_off;
    const int H = c->H, I = c->I;
    auto P = [&](int i) { return (const uint16_t*) (blob + f[i]); };
    auto V = [&](int i) { return (const half*) (blob + f[i]); };
    if (exl3s_recon_had(c->wg, P(0), V(1), V(2), H, I, K, s) || exl3s_recon_had(c->wu, P(3), V(4), V(5), H, I, K, s) ||
        exl3s_recon_had(c->wd, P(6), V(7), V(8), I, H, K, s))
        return -1;
    exl3s_gu_interleave<<<dim3(I / 32, H / 32), dim3(32, 8), 0, s>>>(c->wg, c->wu, (half*) dq_gu, H, I);
    exl3s_transpose<<<dim3(H / 32, I / 32), dim3(32, 8), 0, s>>>(c->wd, (half*) dq_d, I, H);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

// Resident-only routing: logits [T, NE] of experts with res[e] < 0 -> -inf (res: device residency row of the layer).
extern "C" int exl3s_mask(void* ctx, float* logits, int T, int NE, const int32_t* res, int k, cudaStream_t s) {
    (void) ctx;
    if (T <= 0) return 0;
    exl3s_mask_logits<<<T, 256, 0, s>>>(logits, NE, res, k);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
// exl3s_mask with a HOST residency row (the prompt path's host_res): uploaded into the context first (not graph-safe).
extern "C" int exl3s_mask_host(void* ctx, float* logits, int T, int NE, const int32_t* host_res, int k, cudaStream_t s) {
    Ctx* c = (Ctx*) ctx;
    if (NE > c->n_expert) return -1;
    if (cudaMemcpyAsync(c->mask_res, host_res, (size_t) NE * 4, cudaMemcpyHostToDevice, s) != cudaSuccess) return -2;
    return exl3s_mask(ctx, logits, T, NE, c->mask_res, k, s);
}
