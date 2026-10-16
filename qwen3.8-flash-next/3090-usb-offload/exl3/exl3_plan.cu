// M4 step 2a: Strata residency -> exl3_moe_coop inputs, built on the device (graph-capture safe).
//
// exllamav3's exl3_moe_coop takes per-expert pointer tables (int64 device arrays indexed by expert id) inside
// MoeCoopParams, a by-value kernel parameter.  Strata's decode MoE runs inside CUDA graphs whose residency changes
// without recapture (verify.cpp: d_res / slot_off_d_ / resident_plan read at run time), so the tables cannot be
// host-built per launch.  This kernel is one extra captured launch per MoE layer, before rot / A / B, that reads the
// same inputs as Strata's resident_plan (verify_kernels.hpp) and writes the device arrays MoeCoopParams points at:
//
//   tables[f][e] = cache_base + (slot_off ? slot_off[s] : s * blob) + field_off[f]   e = ids[i] resident in slot s
//                = zero_blob + field_off[f]                                         e not resident (res[e] < 0)
//   sel[i] = ids[i] (int32 -> int64);  rw[i] = half(w[i]) for hits, 0 for misses;  *n_miss = misses among the n
//
// zero_blob is an all-zero blob in VRAM: suh = svh = 0 make every miss contribute exactly 0 even before rw = 0, so
// the fused output is the routed sum over the resident experts only and the CPU pool adds the misses (Strata's
// hit/miss split).  Only the n selected entries are written (tables of unselected experts are never read).
// Fields f: 0..8 = gate/up/down x trellis/suh/svh, the order of MoeCoopParams g_trellis .. d_svh; offsets come from
// exl3_pack.json.  One block of 256 threads; n <= 256 entries per launch (bsz * topk; Strata's window is <= 8 x 10).
// Build: `build.sh exl3_plan`; test: test_exl3_plan_gpu.py.
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>

struct Exl3PlanParams {
    const int32_t* res;                  // [n_expert] slot or -1 (VerifyHits::d_res + layer * n_expert)
    const unsigned long long* slot_off;  // [n_slots] per-slot byte offsets, or null: slot * blob
    unsigned long long cache_base;       // VRAM expert arena, slot 0
    long long blob;                      // bytes per blob (slot stride when slot_off is null)
    unsigned long long zero_blob;        // device address of an all-zero blob
    unsigned int field_off[9];           // g/u/d x trellis/suh/svh byte offsets inside a blob
    int n_expert;
    const int32_t* ids;                  // [n] routed expert ids (native_router_top10_multi)
    const float* w;                      // [n] routing weights
    int n;                               // bsz * topk, <= blockDim.x
    long long* tables;                   // [9][n_expert] -> MoeCoopParams g_trellis .. d_svh = tables + f * n_expert
    long long* sel;                      // [n] -> MoeCoopParams sel
    __half* rw;                          // [n] -> MoeCoopParams rw
    int* n_miss;                         // [1]
};
#define OFS(f, v) static_assert(offsetof(Exl3PlanParams, f) == v, "Exl3PlanParams layout: " #f);
OFS(res, 0) OFS(slot_off, 8) OFS(cache_base, 16) OFS(blob, 24) OFS(zero_blob, 32) OFS(field_off, 40) OFS(n_expert, 76)
OFS(ids, 80) OFS(w, 88) OFS(n, 96) OFS(tables, 104) OFS(sel, 112) OFS(rw, 120) OFS(n_miss, 128)
static_assert(sizeof(Exl3PlanParams) == 136, "Exl3PlanParams size");

extern "C" __global__ void __launch_bounds__(256) exl3_plan_kernel(const Exl3PlanParams p) {
    const int i = threadIdx.x;
    bool miss = false;
    if (i < p.n) {
        const int e = p.ids[i];   // contract: 0 <= e < n_expert (the device router's top-k)
        const int s = p.res[e];
        miss = s < 0;
        const unsigned long long base =
            miss ? p.zero_blob
                 : p.cache_base + (p.slot_off ? p.slot_off[s] : (unsigned long long) s * (unsigned long long) p.blob);
#pragma unroll
        for (int f = 0; f < 9; ++f) p.tables[(size_t) f * p.n_expert + e] = (long long) (base + p.field_off[f]);
        p.sel[i] = e;
        p.rw[i] = miss ? __float2half(0.0f) : __float2half_rn(p.w[i]);
    }
    const int misses = __syncthreads_count(miss);
    if (i == 0) *p.n_miss = misses;
}
