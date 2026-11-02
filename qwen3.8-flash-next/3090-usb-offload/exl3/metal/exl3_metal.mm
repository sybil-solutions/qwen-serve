// exl3_metal.mm - host side of libexl3metal (see exl3_metal.h). Objective-C++ internally, plain C API.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>
#include "exl3_metal.h"
#include "exl3_metallib.h"   // exl3_metallib[] / exl3_metallib_len (xxd -i of the offline-compiled metallib)

namespace {
constexpr size_t PAGE = 16384;
constexpr int NU_MAX = 96, PAIR_MAX = 4096;
constexpr size_t PIECE_MAX = (size_t)1 << 30;
constexpr size_t BLOB = EXL3M_BLOB_BYTES;
constexpr uint32_t H = EXL3M_H, FF = EXL3M_FF;
// decode table buffer layout
constexpr size_t T_UNITS = 0, T_PAIRS = 4096, T_ROWPTR = 4096 + (size_t)PAIR_MAX * 16, T_END = T_ROWPTR + 64;

thread_local std::string g_err;
int fail(const std::string &m, int code = -1) { g_err = m; return code; }
double now_ms() {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

struct UnitDH { uint64_t blob; int32_t rows[8]; };            // = MSL UnitD (40 B)
struct PairDH { uint32_t u, slot; float w; uint32_t pad; };
struct DPH { uint32_t mrx, nu, p0, p1; };
struct GPH { uint32_t kind, p[3]; };
struct UnitPH { uint64_t blob; uint32_t row0, nrows; };       // = MSL UnitP (16 B)
struct PPH { uint32_t nrows, cap, p0, p1; };
struct GTH { uint32_t u, r0, nv, pad; };
struct CPH { uint32_t pr; float w; };

struct Piece { uintptr_t base, end; id<MTLBuffer> buf; };
struct Region { uintptr_t key; std::vector<Piece> pieces; };

struct Slot {
    bool busy = false;
    id<MTLBuffer> d_x, d_tbl, d_xq, d_yg, d_zq, d_yd, d_out;   // decode scratch
    id<MTLBuffer> stage; size_t stage_blobs = 0;                 // staging for unregistered blobs
    id<MTLBuffer> p_x, p_tbl, p_A, p_C, p_Z, p_Yd, p_Wg, p_Wu, p_Wd, p_y, p_yh;   // prefill scratch (grow only); p_yh = the fp16 result rows (half-I/O jobs)
    size_t p_x_n = 0, p_tbl_n = 0, p_cap = 0, p_wg_n = 0, p_y_n = 0, p_yh_n = 0;
};

struct Pipes { id<MTLComputePipelineState> rot_dec, mid_dec, comb_dec, gemv[9], rot_pf, rot_pf_h, y_f2h, decode_w, gemm_gu, gemm_d, mid_pf, comb_pf; };
}  // namespace

struct exl3m_ctx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    Pipes P;
    std::shared_mutex rmu;
    std::vector<Region> regions;
    std::mutex smu;
    std::condition_variable scv;
    std::vector<Slot> slots;
    std::mutex stmu;
    exl3m_stats st{};
    std::string dname;
    int pf_chunk_rows = 4096, pf_chunk_experts = 16;
    int wait_spin = 0, unretained = 0;   // decode latency knobs: poll the command buffer status instead of waitUntilCompleted; unretained references
    int64_t skip_mask = 0;   // profiling only: bit i skips prefill kernel i (rot, decode_w x3, gemm_gu, mid, gemm_d, comb)
};

struct exl3m_job {
    Slot *slot;
    id<MTLCommandBuffer> cb;
    void *y_out;
    size_t y_bytes;
    id<MTLBuffer> src;       // shared buffer holding the result
    double t_submit, t_encode, t_copy;
    bool dec = false;         // a decode-shape job (the only kind the spin wait applies to)
};

namespace {
id<MTLComputePipelineState> mkpipe(exl3m_ctx *c, id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:name];
    if (!f) { g_err = std::string("missing kernel ") + name.UTF8String; return nil; }
    id<MTLComputePipelineState> p = [c->dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) g_err = std::string("pipeline ") + name.UTF8String + ": " + (err ? err.localizedDescription.UTF8String : "?");
    return p;
}

id<MTLBuffer> mkbuf(exl3m_ctx *c, size_t n, MTLResourceOptions o = MTLResourceStorageModeShared) {
    return [c->dev newBufferWithLength:(n + 255) & ~(size_t)255 options:o];
}
const MTLResourceOptions PRIV = MTLResourceStorageModePrivate;

Slot *acquire(exl3m_ctx *c) {
    std::unique_lock<std::mutex> l(c->smu);
    for (;;) {
        for (auto &s : c->slots) if (!s.busy) { s.busy = true; return &s; }
        c->scv.wait(l);
    }
}
void release(exl3m_ctx *c, Slot *s) {
    { std::lock_guard<std::mutex> l(c->smu); s->busy = false; }
    c->scv.notify_one();
}

// Resolve a blob pointer to (GPU address, buffer). Returns false if not covered by a registered piece.
bool resolve(exl3m_ctx *c, const void *p, uint64_t &addr, id<MTLBuffer> __strong &buf) {
    uintptr_t a = (uintptr_t)p;
    std::shared_lock<std::shared_mutex> l(c->rmu);
    for (auto &r : c->regions)
        for (auto &pc : r.pieces)
            if (a >= pc.base && a + BLOB <= pc.end) { addr = pc.buf.gpuAddress + (a - pc.base); buf = pc.buf; return true; }
    return false;
}

void use_all(id<MTLComputeCommandEncoder> en, const std::vector<id<MTLBuffer>> &bufs) {
    std::vector<id<MTLBuffer>> u;
    for (auto &b : bufs) { bool dup = false; for (auto &x : u) if (x == b) { dup = true; break; } if (!dup) u.push_back(b); }
    for (auto &b : u) [en useResource:b usage:MTLResourceUsageRead];
}

void ensure_stage(exl3m_ctx *c, Slot *s, size_t nblobs) {
    if (s->stage_blobs >= nblobs) return;
    s->stage = mkbuf(c, nblobs * BLOB);
    s->stage_blobs = nblobs;
}
void grow(exl3m_ctx *c, id<MTLBuffer> __strong &b, size_t &cur, size_t need, MTLResourceOptions o) {
    if (cur >= need) return;
    size_t n = std::max(need, cur + cur / 2);
    b = mkbuf(c, n, o); cur = n;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
extern "C" {

const char *exl3m_last_error(void) { return g_err.c_str(); }
const char *exl3m_device_name(exl3m_ctx *c) { return c->dname.c_str(); }

exl3m_ctx *exl3m_create(int n_threads, int n_slots) {
    (void)n_threads;
    @autoreleasepool {
        exl3m_ctx *c = new exl3m_ctx();
        c->dev = MTLCreateSystemDefaultDevice();
        if (!c->dev) { g_err = "no Metal device"; delete c; return nullptr; }
        c->dname = c->dev.name.UTF8String;
        c->q = [c->dev newCommandQueue];
        NSError *err = nil;
        dispatch_data_t dd = dispatch_data_create(exl3_metallib, exl3_metallib_len, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> lib = [c->dev newLibraryWithData:dd error:&err];
        if (!lib) { g_err = std::string("metallib: ") + (err ? err.localizedDescription.UTF8String : "?"); delete c; return nullptr; }
        Pipes &P = c->P;
        P.rot_dec = mkpipe(c, lib, @"rot_in_dec"); P.mid_dec = mkpipe(c, lib, @"mid_dec"); P.comb_dec = mkpipe(c, lib, @"comb_dec");
        for (int m = 1; m <= 8; m *= 2) P.gemv[m] = mkpipe(c, lib, [NSString stringWithFormat:@"gemv_dec_%d", m]);
        P.rot_pf = mkpipe(c, lib, @"rot_in_pf"); P.decode_w = mkpipe(c, lib, @"decode_w"); P.gemm_gu = mkpipe(c, lib, @"gemm_gu");
        P.gemm_d = mkpipe(c, lib, @"gemm_d"); P.mid_pf = mkpipe(c, lib, @"mid_pf"); P.comb_pf = mkpipe(c, lib, @"comb_pf");
        P.rot_pf_h = mkpipe(c, lib, @"rot_in_pf_h"); P.y_f2h = mkpipe(c, lib, @"y_f2h");
        if (!P.rot_dec || !P.mid_dec || !P.comb_dec || !P.gemv[1] || !P.gemv[2] || !P.gemv[4] || !P.gemv[8] || !P.rot_pf || !P.rot_pf_h || !P.y_f2h ||
            !P.decode_w || !P.gemm_gu || !P.gemm_d || !P.mid_pf || !P.comb_pf) { delete c; return nullptr; }
        if (n_slots <= 0) n_slots = 2;
        if (n_slots > 8) n_slots = 8;
        if (const char *e = getenv("EXL3M_WAIT_SPIN")) c->wait_spin = atoi(e);        // decode jobs: poll the command buffer instead of waitUntilCompleted
        if (const char *e = getenv("EXL3M_UNRETAINED")) c->unretained = atoi(e);
        if (const char *e = getenv("EXL3M_PF_CHUNK_ROWS")) { if (atoi(e) >= 64) c->pf_chunk_rows = atoi(e); }          // prefill scratch size (rows per chunk) ...
        if (const char *e = getenv("EXL3M_PF_CHUNK_EXPERTS")) { if (atoi(e) >= 1) c->pf_chunk_experts = atoi(e); }     // ... and decoded-weight experts per chunk: smaller = less Metal memory
        c->slots.resize((size_t)n_slots);
        for (auto &s : c->slots) {
            s.d_x = mkbuf(c, 8 * (size_t)H * 4);
            s.d_tbl = mkbuf(c, T_END);
            s.d_xq = mkbuf(c, 2 * (size_t)NU_MAX * 8 * H * 2, PRIV);
            s.d_yg = mkbuf(c, 2 * (size_t)NU_MAX * 8 * FF * 4, PRIV);
            s.d_zq = mkbuf(c, (size_t)NU_MAX * 8 * FF * 2, PRIV);
            s.d_yd = mkbuf(c, (size_t)NU_MAX * 8 * H * 4, PRIV);
            s.d_out = mkbuf(c, 8 * (size_t)H * 4);
        }
        return c;
    }
}

void exl3m_destroy(exl3m_ctx *c) { if (c) { @autoreleasepool { delete c; } } }

int64_t exl3m_register(exl3m_ctx *c, const void *base, size_t len) {
    @autoreleasepool {
        uintptr_t b = ((uintptr_t)base + PAGE - 1) & ~(uintptr_t)(PAGE - 1), e = ((uintptr_t)base + len) & ~(uintptr_t)(PAGE - 1);
        if (e <= b) return fail("region smaller than one page");
        Region r; r.key = (uintptr_t)base;
        size_t overlap = (BLOB + PAGE - 1) & ~(PAGE - 1);
        uintptr_t pos = b;
        size_t maxlen = std::min<size_t>(PIECE_MAX, c->dev.maxBufferLength & ~(PAGE - 1));
        if (const char *pm = getenv("EXL3M_PIECE_MB")) maxlen = std::min<size_t>(maxlen, ((size_t)atoll(pm) << 20) & ~(PAGE - 1));   // test hook
        while (pos < e) {
            size_t n = std::min<size_t>(e - pos, maxlen);
            id<MTLBuffer> buf = [c->dev newBufferWithBytesNoCopy:(void *)pos length:n
                                  options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
            if (!buf) return fail("newBufferWithBytesNoCopy failed (len " + std::to_string(n) + ")");
            r.pieces.push_back({pos, pos + n, buf});
            if (pos + n >= e) break;
            pos += n - overlap;
        }
        std::unique_lock<std::shared_mutex> l(c->rmu);
        c->regions.push_back(std::move(r));
        if (getenv("EXL3M_LOG_ALLOC")) fprintf(stderr, "exl3m: registered %.2f GB; Metal allocated %.2f GB of recommended %.2f GB\n", (double)(e - b) / 1e9, (double)c->dev.currentAllocatedSize / 1e9, (double)c->dev.recommendedMaxWorkingSetSize / 1e9);
        return (int64_t)(e - b);
    }
}

int exl3m_unregister(exl3m_ctx *c, const void *base) {
    std::unique_lock<std::shared_mutex> l(c->rmu);
    for (size_t i = 0; i < c->regions.size(); i++)
        if (c->regions[i].key == (uintptr_t)base) { c->regions.erase(c->regions.begin() + (long)i); return 0; }
    return fail("region not registered");
}

int exl3m_set_param(exl3m_ctx *c, const char *key, int64_t v) {
    if (!strcmp(key, "pf_chunk_rows") && v >= 64) { c->pf_chunk_rows = (int)v; return 0; }
    if (!strcmp(key, "pf_chunk_experts") && v >= 1) { c->pf_chunk_experts = (int)v; return 0; }
    if (!strcmp(key, "wait_spin")) { c->wait_spin = (int)v; return 0; }
    if (!strcmp(key, "unretained")) { c->unretained = (int)v; return 0; }
    if (!strcmp(key, "debug_skip_mask")) { c->skip_mask = v; return 0; }
    return -1;
}
void exl3m_get_stats(exl3m_ctx *c, exl3m_stats *out) { std::lock_guard<std::mutex> l(c->stmu); *out = c->st; }
// Metal memory accounting: bytes this process holds as Metal resources vs the device's recommended working set (HostTailAndPrefill 2026-10-12)
int64_t exl3m_alloc_bytes(exl3m_ctx *c) { return (int64_t)c->dev.currentAllocatedSize; }
int64_t exl3m_recommended_bytes(exl3m_ctx *c) { return (int64_t)c->dev.recommendedMaxWorkingSetSize; }

// ---------------------------------------------------------------------------------------------------------------------
int exl3m_moe_rows_submit(exl3m_ctx *c, const float *x, int m, int n_pairs, const int32_t *eidx, const int32_t *ridx, const float *w,
                          const void *const *blobs, float *y_out, exl3m_job **jobp) {
    @autoreleasepool {
        if (m < 1 || m > 8) return fail("m must be 1..8");
        if (n_pairs < 1 || n_pairs > PAIR_MAX) return fail("n_pairs must be 1..4096");
        const double t0 = now_ms();
        int32_t emax = -1;
        for (int p = 0; p < n_pairs; p++) {
            if (eidx[p] < 0 || ridx[p] < 0 || ridx[p] >= m) return fail("bad expert/row index");
            emax = std::max(emax, eidx[p]);
        }
        std::vector<int> unit_of((size_t)emax + 1, -1);
        std::vector<UnitDH> units; std::vector<int> cnt; std::vector<int32_t> uexp;
        std::vector<PairDH> pairs((size_t)n_pairs);
        for (int p = 0; p < n_pairs; p++) {
            int e = eidx[p], u = unit_of[(size_t)e];
            if (u < 0) {
                if ((int)units.size() >= NU_MAX) return fail("more than 96 distinct experts");
                u = (int)units.size(); unit_of[(size_t)e] = u;
                UnitDH ud{}; for (int r = 0; r < 8; r++) ud.rows[r] = -1;
                units.push_back(ud); cnt.push_back(0); uexp.push_back(e);
            }
            int slot = cnt[(size_t)u]++;
            if (slot >= 8) return fail("more than 8 pairs for one expert");
            units[(size_t)u].rows[slot] = ridx[p];
            pairs[(size_t)p] = { (uint32_t)u, (uint32_t)slot, w[p], 0 };
        }
        const int nu = (int)units.size();
        int maxslot = *std::max_element(cnt.begin(), cnt.end()), mrx = maxslot <= 1 ? 1 : maxslot <= 2 ? 2 : maxslot <= 4 ? 4 : 8;

        Slot *s = acquire(c);
        double tc = 0;
        // resolve blobs
        std::vector<id<MTLBuffer>> used;
        size_t nstaged = 0;
        for (int u = 0; u < nu; u++) {
            const void *bp = blobs[uexp[(size_t)u]];
            if (!bp) { release(c, s); return fail("null blob pointer"); }
            uint64_t addr; id<MTLBuffer> buf;
            if (resolve(c, bp, addr, buf)) { units[(size_t)u].blob = addr; used.push_back(buf); }
            else {
                if (!nstaged) ensure_stage(c, s, NU_MAX);
                double a = now_ms();
                memcpy((char *)s->stage.contents + nstaged * BLOB, bp, BLOB);
                tc += now_ms() - a;
                units[(size_t)u].blob = s->stage.gpuAddress + nstaged * BLOB; nstaged++;
            }
        }
        if (nstaged) used.push_back(s->stage);
        // CSR per output row in pair order
        std::vector<uint32_t> rowptr((size_t)m + 1, 0);
        for (int p = 0; p < n_pairs; p++) rowptr[(size_t)ridx[p] + 1]++;
        for (int r = 0; r < m; r++) rowptr[(size_t)r + 1] += rowptr[(size_t)r];
        std::vector<PairDH> sorted((size_t)n_pairs);
        { std::vector<uint32_t> fill(rowptr.begin(), rowptr.end() - 1); for (int p = 0; p < n_pairs; p++) sorted[fill[(size_t)ridx[p]]++] = pairs[(size_t)p]; }
        {
            double a = now_ms();
            memcpy(s->d_x.contents, x, (size_t)m * H * 4);
            char *t = (char *)s->d_tbl.contents;
            memcpy(t + T_UNITS, units.data(), units.size() * sizeof(UnitDH));
            memcpy(t + T_PAIRS, sorted.data(), sorted.size() * sizeof(PairDH));
            memcpy(t + T_ROWPTR, rowptr.data(), rowptr.size() * 4);
            tc += now_ms() - a;
        }

        id<MTLCommandBuffer> cb = c->unretained ? [c->q commandBufferWithUnretainedReferences] : [c->q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        use_all(en, used);
        DPH dp = { (uint32_t)mrx, (uint32_t)nu, 0, 0 };
        GPH g0 = { 0, {0, 0, 0} }, g1 = { 1, {0, 0, 0} };
        const MTLSize tg128 = MTLSizeMake(128, 1, 1), tg32 = MTLSizeMake(32, 1, 1);
        [en setComputePipelineState:c->P.rot_dec];
        [en setBuffer:s->d_x offset:0 atIndex:0]; [en setBuffer:s->d_tbl offset:T_UNITS atIndex:1]; [en setBuffer:s->d_xq offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3];
        if (!(c->skip_mask & 0x100)) [en dispatchThreadgroups:MTLSizeMake(20, (NSUInteger)nu * mrx, 2) threadsPerThreadgroup:tg128];
        [en setComputePipelineState:c->P.gemv[mrx]];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_xq offset:0 atIndex:1]; [en setBuffer:s->d_yg offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3]; [en setBytes:&g0 length:sizeof g0 atIndex:4];
        if (!(c->skip_mask & 0x200)) [en dispatchThreadgroups:MTLSizeMake(40, (NSUInteger)nu, 2) threadsPerThreadgroup:tg32];
        [en setComputePipelineState:c->P.mid_dec];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_yg offset:0 atIndex:1]; [en setBuffer:s->d_zq offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3];
        if (!(c->skip_mask & 0x400)) [en dispatchThreadgroups:MTLSizeMake(5, (NSUInteger)nu * mrx, 1) threadsPerThreadgroup:tg128];
        [en setComputePipelineState:c->P.gemv[mrx]];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_zq offset:0 atIndex:1]; [en setBuffer:s->d_yd offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3]; [en setBytes:&g1 length:sizeof g1 atIndex:4];
        if (!(c->skip_mask & 0x800)) [en dispatchThreadgroups:MTLSizeMake(160, (NSUInteger)nu, 1) threadsPerThreadgroup:tg32];
        [en setComputePipelineState:c->P.comb_dec];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_yd offset:0 atIndex:1]; [en setBuffer:s->d_tbl offset:T_PAIRS atIndex:2];
        [en setBuffer:s->d_tbl offset:T_ROWPTR atIndex:3]; [en setBuffer:s->d_out offset:0 atIndex:4]; [en setBytes:&dp length:sizeof dp atIndex:5];
        if (!(c->skip_mask & 0x1000)) [en dispatchThreadgroups:MTLSizeMake(20, (NSUInteger)m, 1) threadsPerThreadgroup:tg128];
        [en endEncoding];
        exl3m_job *j = new exl3m_job{ s, cb, y_out, (size_t)m * H * 4, s->d_out, now_ms(), 0, tc };
        j->dec = true;
        j->t_encode = j->t_submit - t0;
        [cb commit];
        *jobp = j;
        { std::lock_guard<std::mutex> l(c->stmu); c->st.decode_calls++; c->st.blobs_staged += (uint64_t)nstaged; c->st.blobs_mapped += (uint64_t)(nu - (int)nstaged); }
        return 0;
    }
}

int exl3m_wait(exl3m_ctx *c, exl3m_job *j) {
    @autoreleasepool {
        if (c->wait_spin && j->dec) { while (j->cb.status < MTLCommandBufferStatusCompleted) {} } else [j->cb waitUntilCompleted];
        if (!j->dec && getenv("EXL3M_LOG_ALLOC")) {
            static int n = 0;
            if (++n % 50 == 1) fprintf(stderr, "exl3m: prefill job %d: Metal allocated %.2f GB of recommended %.2f GB\n", n, (double)c->dev.currentAllocatedSize / 1e9, (double)c->dev.recommendedMaxWorkingSetSize / 1e9);
        }
        const double t1 = now_ms();
        int rc = 0;
        if (j->cb.status != MTLCommandBufferStatusCompleted) {
            NSError *e = j->cb.error;
            rc = fail(std::string("command buffer failed: ") + (e ? e.localizedDescription.UTF8String : "?"));
        } else {
            memcpy(j->y_out, j->src.contents, j->y_bytes);
        }
        const double t2 = now_ms();
        {
            std::lock_guard<std::mutex> l(c->stmu);
            c->st.gpu_ms_last = (j->cb.GPUEndTime - j->cb.GPUStartTime) * 1e3;
            c->st.wall_ms_last = t1 - j->t_submit;
            c->st.encode_ms_last = j->t_encode;
            c->st.copy_ms_last = j->t_copy + (t2 - t1);
        }
        release(c, j->slot);
        delete j;
        return rc;
    }
}

int exl3m_moe_rows(exl3m_ctx *c, const float *x, int m, int n_pairs, const int32_t *eidx, const int32_t *ridx, const float *w,
                   const void *const *blobs, float *y_out) {
    exl3m_job *j = nullptr;
    int rc = exl3m_moe_rows_submit(c, x, m, n_pairs, eidx, ridx, w, blobs, y_out, &j);
    return rc ? rc : exl3m_wait(c, j);
}

// ---------------------------------------------------------------------------------------------------------------------
static int prefill_submit_impl(exl3m_ctx *c, const void *x, bool half_io, int n_tok, int64_t n_pairs, const int32_t *eidx, const int32_t *ridx,
                               const float *w, const void *const *blobs, void *y_out, exl3m_job **jobp) {
    @autoreleasepool {
        if (n_tok < 1 || n_tok > 16384) return fail("n_tok must be 1..16384");
        if (n_pairs < 1 || n_pairs > (1 << 20)) return fail("n_pairs must be 1..2^20");
        const double t0 = now_ms();
        int32_t emax = -1;
        for (int64_t p = 0; p < n_pairs; p++) {
            if (eidx[p] < 0 || ridx[p] < 0 || ridx[p] >= n_tok) return fail("bad expert/row index");
            emax = std::max(emax, eidx[p]);
        }
        // counting sort by expert (stable: pair order kept within an expert)
        std::vector<uint32_t> ecount((size_t)emax + 2, 0);
        for (int64_t p = 0; p < n_pairs; p++) ecount[(size_t)eidx[p] + 1]++;
        for (size_t e = 0; e <= (size_t)emax; e++) ecount[e + 1] += ecount[e];
        std::vector<uint32_t> order((size_t)n_pairs);
        { std::vector<uint32_t> fill(ecount.begin(), ecount.end() - 1); for (int64_t p = 0; p < n_pairs; p++) order[fill[(size_t)eidx[p]]++] = (uint32_t)p; }
        std::vector<int32_t> uexp;                       // used experts ascending
        for (int32_t e = 0; e <= emax; e++) if (ecount[(size_t)e + 1] > ecount[(size_t)e]) uexp.push_back(e);
        const int nu = (int)uexp.size();
        for (int u = 0; u < nu; u++)
            if (ecount[(size_t)uexp[(size_t)u] + 1] - ecount[(size_t)uexp[(size_t)u]] > 16384) return fail("more than 16384 rows for one expert");

        // chunk plan
        struct Chunk { int u0, u1; uint32_t rows; };
        std::vector<Chunk> chunks;
        {
            int u0 = 0; uint32_t rows = 0;
            for (int u = 0; u < nu; u++) {
                uint32_t nr = ecount[(size_t)uexp[(size_t)u] + 1] - ecount[(size_t)uexp[(size_t)u]];
                if (u > u0 && (u - u0 >= c->pf_chunk_experts || rows + nr > (uint32_t)c->pf_chunk_rows)) { chunks.push_back({u0, u, rows}); u0 = u; rows = 0; }
                rows += nr;
            }
            chunks.push_back({u0, nu, rows});
        }
        uint32_t cap_rows = 0; int maxg = 0; size_t tbl_bytes = 0;
        for (auto &ch : chunks) { cap_rows = std::max(cap_rows, ch.rows); maxg = std::max(maxg, ch.u1 - ch.u0);
            tbl_bytes += 256 + (size_t)(ch.u1 - ch.u0) * 16 + 256 + (size_t)ch.rows * 8 + 256 + ((size_t)ch.rows / 32 + (size_t)(ch.u1 - ch.u0) + 1) * 16 + 256 + (size_t)ch.rows * 16 + 1024; }
        cap_rows += 64;

        Slot *s = acquire(c);
        double tc = 0;
        std::vector<id<MTLBuffer>> used;
        std::vector<uint64_t> addr((size_t)nu);
        size_t nstaged = 0;
        for (int u = 0; u < nu; u++) {
            const void *bp = blobs[uexp[(size_t)u]];
            if (!bp) { release(c, s); return fail("null blob pointer"); }
            id<MTLBuffer> buf;
            if (resolve(c, bp, addr[(size_t)u], buf)) used.push_back(buf);
            else nstaged++;
        }
        if (nstaged) {
            if (s->stage_blobs < nstaged) { s->stage = mkbuf(c, nstaged * BLOB); s->stage_blobs = nstaged; }
            size_t k = 0;
            for (int u = 0; u < nu; u++) {
                id<MTLBuffer> buf; uint64_t a;
                if (resolve(c, blobs[uexp[(size_t)u]], a, buf)) continue;
                double a0 = now_ms();
                memcpy((char *)s->stage.contents + k * BLOB, blobs[uexp[(size_t)u]], BLOB);
                tc += now_ms() - a0;
                addr[(size_t)u] = s->stage.gpuAddress + k * BLOB; k++;
            }
            used.push_back(s->stage);
        }
        // scratch
        const size_t xbytes = (size_t)n_tok * H * 4;                    // the fp32 accumulation rows (always fp32: a row can be touched by several expert chunks)
        const size_t xbytes_in = half_io ? xbytes / 2 : xbytes;        // the input rows as the host holds them
        grow(c, s->p_x, s->p_x_n, xbytes_in, MTLResourceStorageModeShared);
        grow(c, s->p_y, s->p_y_n, xbytes, MTLResourceStorageModeShared);
        if (half_io) grow(c, s->p_yh, s->p_yh_n, xbytes / 2, MTLResourceStorageModeShared);
        grow(c, s->p_tbl, s->p_tbl_n, tbl_bytes + 4096, MTLResourceStorageModeShared);
        if (s->p_cap < cap_rows) {
            s->p_A = mkbuf(c, 2 * (size_t)cap_rows * H * 2, PRIV);
            s->p_C = mkbuf(c, 2 * (size_t)cap_rows * FF * 4, PRIV);
            s->p_Z = mkbuf(c, (size_t)cap_rows * FF * 2, PRIV);
            s->p_Yd = mkbuf(c, (size_t)cap_rows * H * 4, PRIV);
            s->p_cap = cap_rows;
        }
        const size_t wbytes = (size_t)H * FF * 2;
        grow(c, s->p_Wg, s->p_wg_n, (size_t)maxg * wbytes, PRIV);
        if (!s->p_Wu || s->p_Wu.length < s->p_Wg.length) { s->p_Wu = mkbuf(c, s->p_Wg.length, PRIV); s->p_Wd = mkbuf(c, s->p_Wg.length, PRIV); }
        {
            double a = now_ms();
            memcpy(s->p_x.contents, x, xbytes_in);
            tc += now_ms() - a;
        }

        id<MTLCommandBuffer> cb = [c->q commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl fillBuffer:s->p_y range:NSMakeRange(0, xbytes) value:0];
        [bl endEncoding];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        use_all(en, used);
        char *tbl = (char *)s->p_tbl.contents; size_t toff = 0;
        auto take = [&](size_t n) { toff = (toff + 255) & ~(size_t)255; size_t o = toff; toff += n; return o; };
        std::vector<int32_t> row_slot((size_t)n_tok, -1);
        for (auto &ch : chunks) {
            const int G = ch.u1 - ch.u0; const uint32_t R = ch.rows;
            size_t o_units = take((size_t)G * 16), o_uor = take((size_t)R * 4), o_src = take((size_t)R * 4);
            size_t ntiles_max = (size_t)R / 32 + (size_t)G + 1;
            size_t o_tiles = take(ntiles_max * 16);
            size_t o_touched = take((size_t)R * 4), o_tptr = take(((size_t)R + 1) * 4), o_cps = take((size_t)R * 8);
            UnitPH *U = (UnitPH *)(tbl + o_units); uint32_t *uor = (uint32_t *)(tbl + o_uor); int32_t *src = (int32_t *)(tbl + o_src);
            GTH *tiles = (GTH *)(tbl + o_tiles); uint32_t ntiles = 0;
            std::vector<float> pw(R); std::vector<uint32_t> prow(R);
            uint32_t pr = 0;
            for (int g = 0; g < G; g++) {
                int u = ch.u0 + g; int32_t e = uexp[(size_t)u];
                uint32_t nr = ecount[(size_t)e + 1] - ecount[(size_t)e];
                U[g] = { addr[(size_t)u], pr, nr };
                for (uint32_t i = 0; i < nr; i++, pr++) {
                    uint32_t p = order[ecount[(size_t)e] + i];
                    uor[pr] = (uint32_t)g; src[pr] = ridx[p]; pw[pr] = w[p]; prow[pr] = (uint32_t)ridx[p];
                }
                for (uint32_t r = 0; r < nr; r += 32) tiles[ntiles++] = { (uint32_t)g, U[g].row0 + r, std::min<uint32_t>(32, nr - r), 0 };
            }
            // combine CSR: touched rows, pairs in chunk order
            uint32_t *touched = (uint32_t *)(tbl + o_touched), *tptr = (uint32_t *)(tbl + o_tptr); CPH *cps = (CPH *)(tbl + o_cps);
            uint32_t nt = 0;
            for (uint32_t i = 0; i < R; i++) { int32_t &rs = row_slot[prow[i]]; if (rs < 0) { rs = (int32_t)nt; touched[nt++] = prow[i]; } }
            std::vector<uint32_t> cnt(nt + 1, 0);
            for (uint32_t i = 0; i < R; i++) cnt[(size_t)row_slot[prow[i]] + 1]++;
            for (uint32_t i = 0; i < nt; i++) cnt[i + 1] += cnt[i];
            memcpy(tptr, cnt.data(), ((size_t)nt + 1) * 4);
            { std::vector<uint32_t> fill(cnt.begin(), cnt.end() - 1); for (uint32_t i = 0; i < R; i++) cps[fill[(size_t)row_slot[prow[i]]]++] = { i, pw[i] }; }
            for (uint32_t i = 0; i < nt; i++) row_slot[touched[i]] = -1;
            // encode
            PPH pp = { R, (uint32_t)s->p_cap, 0, 0 };
            const MTLSize tg128 = MTLSizeMake(128, 1, 1);
            [en setComputePipelineState:half_io ? c->P.rot_pf_h : c->P.rot_pf];
            [en setBuffer:s->p_x offset:0 atIndex:0]; [en setBuffer:s->p_tbl offset:o_units atIndex:1]; [en setBuffer:s->p_tbl offset:o_uor atIndex:2];
            [en setBuffer:s->p_tbl offset:o_src atIndex:3]; [en setBuffer:s->p_A offset:0 atIndex:4]; [en setBytes:&pp length:sizeof pp atIndex:5];
            if (!(c->skip_mask & 1)) [en dispatchThreadgroups:MTLSizeMake(20, R, 2) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:c->P.decode_w];
            [en setBuffer:s->p_tbl offset:o_units atIndex:0];
            for (uint32_t which = 0; which < 3; which++) {
                id<MTLBuffer> wb = which == 0 ? s->p_Wg : which == 1 ? s->p_Wu : s->p_Wd;
                [en setBuffer:wb offset:0 atIndex:1]; [en setBytes:&which length:4 atIndex:2];
                if (!(c->skip_mask & 2)) [en dispatchThreads:MTLSizeMake(6400 * 32, (NSUInteger)G, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [en setComputePipelineState:c->P.gemm_gu];
            [en setBuffer:s->p_A offset:0 atIndex:0]; [en setBuffer:s->p_Wg offset:0 atIndex:1]; [en setBuffer:s->p_Wu offset:0 atIndex:2];
            [en setBuffer:s->p_C offset:0 atIndex:3]; [en setBuffer:s->p_tbl offset:o_tiles atIndex:4]; [en setBytes:&pp length:sizeof pp atIndex:5];
            if (!(c->skip_mask & 4)) [en dispatchThreadgroups:MTLSizeMake(10, ntiles, 2) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:c->P.mid_pf];
            [en setBuffer:s->p_tbl offset:o_units atIndex:0]; [en setBuffer:s->p_tbl offset:o_uor atIndex:1]; [en setBuffer:s->p_C offset:0 atIndex:2];
            [en setBuffer:s->p_Z offset:0 atIndex:3]; [en setBytes:&pp length:sizeof pp atIndex:4];
            if (!(c->skip_mask & 8)) [en dispatchThreadgroups:MTLSizeMake(5, R, 1) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:c->P.gemm_d];
            [en setBuffer:s->p_Z offset:0 atIndex:0]; [en setBuffer:s->p_Wd offset:0 atIndex:1]; [en setBuffer:s->p_Yd offset:0 atIndex:2];
            [en setBuffer:s->p_tbl offset:o_tiles atIndex:3];
            if (!(c->skip_mask & 16)) [en dispatchThreadgroups:MTLSizeMake(40, ntiles, 1) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:c->P.comb_pf];
            [en setBuffer:s->p_tbl offset:o_units atIndex:0]; [en setBuffer:s->p_tbl offset:o_uor atIndex:1]; [en setBuffer:s->p_Yd offset:0 atIndex:2];
            [en setBuffer:s->p_tbl offset:o_touched atIndex:3]; [en setBuffer:s->p_tbl offset:o_tptr atIndex:4]; [en setBuffer:s->p_tbl offset:o_cps atIndex:5];
            [en setBuffer:s->p_y offset:0 atIndex:6];
            if (!(c->skip_mask & 32)) [en dispatchThreadgroups:MTLSizeMake(20, nt, 1) threadsPerThreadgroup:tg128];
        }
        if (half_io) {   // fp32 rows -> fp16 on the GPU (the host's clamp + cast, bit for bit), behind the last chunk's combine
            const uint32_t nel = (uint32_t)n_tok * H;
            [en setComputePipelineState:c->P.y_f2h];
            [en setBuffer:s->p_y offset:0 atIndex:0]; [en setBuffer:s->p_yh offset:0 atIndex:1]; [en setBytes:&nel length:4 atIndex:2];
            [en dispatchThreads:MTLSizeMake(nel, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [en endEncoding];
        exl3m_job *j = new exl3m_job{ s, cb, y_out, half_io ? xbytes / 2 : xbytes, half_io ? s->p_yh : s->p_y, now_ms(), 0, tc };
        j->t_encode = j->t_submit - t0;
        [cb commit];
        *jobp = j;
        { std::lock_guard<std::mutex> l(c->stmu); c->st.prefill_calls++; c->st.blobs_staged += (uint64_t)nstaged; c->st.blobs_mapped += (uint64_t)((size_t)nu - nstaged); }
        return 0;
    }
}

int exl3m_moe_prefill_submit(exl3m_ctx *c, const float *x, int n_tok, int64_t n_pairs, const int32_t *eidx, const int32_t *ridx,
                             const float *w, const void *const *blobs, float *y_out, exl3m_job **jobp) {
    return prefill_submit_impl(c, x, false, n_tok, n_pairs, eidx, ridx, w, blobs, y_out, jobp);
}

// fp16 in / fp16 out: x rows as fp16 (uint16 bits), y_out rows as fp16 = the fp32 accumulated rows clamped to +-65504 and rounded to nearest even
int exl3m_moe_prefill_h_submit(exl3m_ctx *c, const uint16_t *x16, int n_tok, int64_t n_pairs, const int32_t *eidx, const int32_t *ridx,
                               const float *w, const void *const *blobs, uint16_t *y16_out, exl3m_job **jobp) {
    return prefill_submit_impl(c, x16, true, n_tok, n_pairs, eidx, ridx, w, blobs, y16_out, jobp);
}

int exl3m_moe_prefill(exl3m_ctx *c, const float *x, int n_tok, int64_t n_pairs, const int32_t *eidx, const int32_t *ridx, const float *w,
                      const void *const *blobs, float *y_out) {
    exl3m_job *j = nullptr;
    int rc = exl3m_moe_prefill_submit(c, x, n_tok, n_pairs, eidx, ridx, w, blobs, y_out, &j);
    return rc ? rc : exl3m_wait(c, j);
}

}  // extern "C"
