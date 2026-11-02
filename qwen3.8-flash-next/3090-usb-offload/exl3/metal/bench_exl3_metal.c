// bench_exl3_metal.c - microbench of libexl3metal on the mini: decode shape, prefill shape, Metal+NEON concurrency / split.
// Build (mini): clang -O2 -o bench_exl3_metal bench_exl3_metal.c -I. -L. -lexl3metal -Wl,-rpath,@loader_path
// Run: ./bench_exl3_metal [pool_blobs=256] [section: all|decode|prefill|conc|split|mmap]   (reads blobs of layer 24 from experts.bin)
#include <arm_neon.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "exl3_metal.h"

#define H 2560
#define FF 640
#define BLOB EXL3M_BLOB_BYTES
#define LAYER 953548800ull
enum { O_GT = 0, O_UT = 614400, O_DT = 1228800, O_GSUH = 1843200, O_GSVH = 1848320, O_USUH = 1849600, O_USVH = 1854720, O_DSUH = 1856000, O_DSVH = 1857280 };

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
typedef struct { double min, med, mean, p95; } St;
static St stat(double *v, int n) { qsort(v, (size_t)n, sizeof(double), cmpd); double s = 0; for (int i = 0; i < n; i++) s += v[i]; St r = { v[0], v[n / 2], s / n, v[(int)(n * 0.95)] }; return r; }

// ---- NEON reference (libexl3cpu): one expert, one row, whole pipeline on the calling thread ---------------------------------------
static void (*cpu_had_in)(const uint16_t *, const uint16_t *, int, int, uint16_t *);
static int (*cpu_gemv)(const uint16_t *, const uint16_t *, const uint16_t *, int, int, int, int, int, float *, float *, uint16_t *, int);
typedef struct { uint16_t xf[H], xh[H], zh[FF]; uint16_t xq[H * 2 + 64]; float g[FF], u[FF], y[H], scratch[H]; _Float16 z[FF]; } NeonBuf;
static void neon_expert(const uint8_t *b, const float *x, float *out, NeonBuf *nb) {
    for (int i = 0; i < H; i++) { _Float16 h = (_Float16)x[i]; memcpy(&nb->xf[i], &h, 2); }
    uint16_t *xq = (uint16_t *)(((uintptr_t)nb->xq + 31) & ~(uintptr_t)31);
    cpu_had_in(nb->xf, (const uint16_t *)(b + O_GSUH), 1, H, nb->xh);
    cpu_gemv(nb->xh, (const uint16_t *)(b + O_GT), (const uint16_t *)(b + O_GSVH), 1, H, FF, 3, 2, nb->g, nb->scratch, xq, 1);
    cpu_had_in(nb->xf, (const uint16_t *)(b + O_USUH), 1, H, nb->xh);
    cpu_gemv(nb->xh, (const uint16_t *)(b + O_UT), (const uint16_t *)(b + O_USVH), 1, H, FF, 3, 2, nb->u, nb->scratch, xq, 1);
    for (int i = 0; i < FF; i++) nb->z[i] = (_Float16)(nb->g[i] / (1.f + expf(-nb->g[i])) * nb->u[i]);
    cpu_had_in((const uint16_t *)nb->z, (const uint16_t *)(b + O_DSUH), 1, FF, nb->zh);
    cpu_gemv(nb->zh, (const uint16_t *)(b + O_DT), (const uint16_t *)(b + O_DSVH), 1, FF, H, 3, 2, out, nb->scratch, xq, 1);
}

// ---- tiny worker pool (condvar) ------------------------------------------------------------------------------------------------
typedef struct { pthread_mutex_t mu; pthread_cond_t cv, done; int nthreads, gen, pending, stop; void (*fn)(int, void *); void *arg; } Pool;
typedef struct { Pool *p; int id; } PW;
static void *pool_main(void *a) {
    PW *w = a; Pool *p = w->p; int seen = 0;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (p->gen == seen && !p->stop) pthread_cond_wait(&p->cv, &p->mu);
        if (p->stop) { pthread_mutex_unlock(&p->mu); return NULL; }
        seen = p->gen; void (*fn)(int, void *) = p->fn; void *arg = p->arg;
        pthread_mutex_unlock(&p->mu);
        fn(w->id, arg);
        pthread_mutex_lock(&p->mu); if (--p->pending == 0) pthread_cond_signal(&p->done); pthread_mutex_unlock(&p->mu);
    }
}
static Pool *pool_new(int n) {
    Pool *p = calloc(1, sizeof *p); pthread_mutex_init(&p->mu, 0); pthread_cond_init(&p->cv, 0); pthread_cond_init(&p->done, 0); p->nthreads = n;
    for (int i = 0; i < n; i++) { PW *w = malloc(sizeof *w); w->p = p; w->id = i; pthread_t t; pthread_create(&t, 0, pool_main, w); pthread_detach(t); }
    return p;
}
static void pool_start(Pool *p, void (*fn)(int, void *), void *arg) {
    pthread_mutex_lock(&p->mu); p->fn = fn; p->arg = arg; p->pending = p->nthreads; p->gen++; pthread_cond_broadcast(&p->cv); pthread_mutex_unlock(&p->mu);
}
static void pool_wait(Pool *p) { pthread_mutex_lock(&p->mu); while (p->pending) pthread_cond_wait(&p->done, &p->mu); pthread_mutex_unlock(&p->mu); }

// ---- data --------------------------------------------------------------------------------------------------------------------------
static uint8_t *region; static size_t region_len; static int NB;
static const void **bptr;
static float *xs;     // [16384][H]
static exl3m_ctx *ctx;
static NeonBuf *nbufs[16];

static void load_pool(int n, const char *packpath) {
    NB = n; region_len = ((size_t)n * BLOB + 65535) & ~(size_t)16383;
    region = mmap(0, region_len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    int fd = open(packpath, O_RDONLY);
    if (fd < 0) { perror("open pack"); exit(1); }
    double t = now_ms();
    for (int i = 0; i < n; i++) if (pread(fd, region + (size_t)i * BLOB, BLOB, (off_t)(24 * LAYER + (off_t)i * BLOB)) != (ssize_t)BLOB) { perror("pread"); exit(1); }
    close(fd);
    mlock(region, region_len);
    printf("loaded %d blobs of layer 24 (%.0f MB) in %.0f ms\n", n, (double)n * BLOB / 1e6, now_ms() - t);
    bptr = malloc(sizeof(void *) * (size_t)n);
    for (int i = 0; i < n; i++) bptr[i] = region + (size_t)i * BLOB;
    int64_t r = exl3m_register(ctx, region, region_len);
    printf("registered %lld bytes NoCopy\n", (long long)r);
}

// pairs for decode: E distinct experts starting at e0 (mod NB); expert i serves row i%m, and row (i+1)%m when m >= 2
static int mk_decode(int m, int E, int e0, int32_t *ei, int32_t *ri, float *w) {
    int n = 0;
    for (int i = 0; i < E; i++) {
        ei[n] = (e0 + i * 7) % NB; ri[n] = i % m; w[n++] = 0.1f + 0.01f * i;
        if (m >= 2) { ei[n] = (e0 + i * 7) % NB; ri[n] = (i + 1) % m; w[n++] = 0.05f; }
    }
    return n;
}

static void bench_decode(void) {
    printf("\n== DECODE: exl3m_moe_rows (sync: encode + submit + GPU + wait + y copy), cold-ish blobs rotating over %d experts ==\n", NB);
    printf("m  E  pairs |  min   med  mean   p95 ms (wall/call) | gpu_ms(med) encode_ms copy_ms | NEON 1-thread ms (same experts, 1 row each)\n");
    static int32_t ei[512], ri[512]; static float w[512]; static float y[8 * H];
    int Ms[] = { 1, 2, 4 }, Es[] = { 10, 13, 20 };
    for (int mi = 0; mi < 3; mi++) for (int ei_ = 0; ei_ < 3; ei_++) {
        int m = Ms[mi], E = Es[ei_]; enum { IT = 300 };
        static double wall[IT], gpu[IT], enc[IT], cp[IT];
        for (int it = -20; it < IT; it++) {
            int n = mk_decode(m, E, ((it + 20) * 37 + 100) % NB, ei, ri, w);
            double t = now_ms();
            if (exl3m_moe_rows(ctx, xs, m, n, ei, ri, w, bptr, y)) { printf("ERR %s\n", exl3m_last_error()); exit(1); }
            double d = now_ms() - t;
            exl3m_stats s; exl3m_get_stats(ctx, &s);
            if (it >= 0) { wall[it] = d; gpu[it] = s.gpu_ms_last; enc[it] = s.encode_ms_last; cp[it] = s.copy_ms_last; }
        }
        double nt = 1e9;
        for (int r = 0; r < 5; r++) { double t = now_ms(); for (int i = 0; i < E; i++) neon_expert(bptr[(100 + i * 7) % NB], xs, y, nbufs[0]); double d = now_ms() - t; if (d < nt) nt = d; }
        int n = mk_decode(m, E, 100, ei, ri, w);
        St sw = stat(wall, IT), sg = stat(gpu, IT), se = stat(enc, IT), sc = stat(cp, IT);
        printf("%d %2d %4d | %5.3f %5.3f %5.3f %5.3f | %5.3f %5.3f %5.3f | %.2f (E x 1 row)\n", m, E, n, sw.min, sw.med, sw.mean, sw.p95, sg.med, se.med, sc.med, nt);
    }
    if (getenv("PROFILE")) {
        const char *nm[] = { "baseline", "-rot", "-gemv_gate_up", "-mid", "-gemv_down", "-comb", "-all kernels (floor)" };
        int64_t mk[] = { 0, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x1f00 };
        printf("DECODE PROFILE m=1 E=13 (GPU ms median of 200, kernels skipped):\n");
        for (int i = 0; i < 7; i++) { exl3m_set_param(ctx, "debug_skip_mask", mk[i]); double v[200]; int n = mk_decode(1, 13, 100, ei, ri, w);
            for (int it = -10; it < 200; it++) { exl3m_moe_rows(ctx, xs, 1, n, ei, ri, w, bptr, y); exl3m_stats s; exl3m_get_stats(ctx, &s); if (it >= 0) v[it] = s.gpu_ms_last; }
            printf("  %-24s %.3f ms\n", nm[i], stat(v, 200).med); }
        exl3m_set_param(ctx, "debug_skip_mask", 0);
    }
    for (int mode = 1; mode <= 3; mode++) {
        exl3m_set_param(ctx, "wait_spin", mode & 1); exl3m_set_param(ctx, "unretained", (mode >> 1) & 1);
        static double wall[300]; int n = mk_decode(1, 10, 100, ei, ri, w);
        for (int it = -20; it < 300; it++) { double t = now_ms(); exl3m_moe_rows(ctx, xs, 1, n, ei, ri, w, bptr, y); if (it >= 0) wall[it] = now_ms() - t; }
        St s = stat(wall, 300); printf("m=1 E=10 wait_spin=%d unretained=%d: min %.3f med %.3f mean %.3f p95 %.3f ms\n", mode & 1, (mode >> 1) & 1, s.min, s.med, s.mean, s.p95);
    }
    exl3m_set_param(ctx, "wait_spin", 0); exl3m_set_param(ctx, "unretained", 0);
    // latency floor: one expert, m=1
    { static double wall[500]; int n = mk_decode(1, 1, 5, ei, ri, w);
      for (int it = -20; it < 500; it++) { double t = now_ms(); exl3m_moe_rows(ctx, xs, 1, n, ei, ri, w, bptr, y); if (it >= 0) wall[it] = now_ms() - t; }
      St s = stat(wall, 500); printf("floor (m=1, 1 expert): min %.3f med %.3f mean %.3f p95 %.3f ms\n", s.min, s.med, s.mean, s.p95); }
}

static void bench_prefill(int ntok_list_len, const int *ntoks) {
    printf("\n== PREFILL: exl3m_moe_prefill, top-10 over %d experts (one layer), warm blobs in registered NoCopy region ==\n", NB);
    printf("tokens pairs chunk(rows,experts) | ms/layer (min med) | gpu_ms(med) encode copy | eff TFLOPS (useful, med) | tok/s if 48 layers like this\n");
    int32_t *ei = malloc(sizeof(int32_t) * 1000000), *ri = malloc(sizeof(int32_t) * 1000000); float *w = malloc(sizeof(float) * 1000000);
    float *y = malloc((size_t)16384 * H * 4);
    srand(7);
    int cfg[][2] = { { 4096, 16 }, { 4096, 32 }, { 8192, 32 }, { 2048, 16 } };
    for (int ti = 0; ti < ntok_list_len; ti++) {
        int nt = ntoks[ti];
        long n = 0;
        for (int r = 0; r < nt; r++) {
            int pick[10];
            for (int k = 0; k < 10; k++) { int e, dup; do { e = rand() % NB; dup = 0; for (int q = 0; q < k; q++) dup |= pick[q] == e; } while (dup); pick[k] = e; ei[n] = e; ri[n] = r; w[n] = 0.1f; n++; }
        }
        for (int ci = 0; ci < 4; ci++) {
            exl3m_set_param(ctx, "pf_chunk_rows", cfg[ci][0]); exl3m_set_param(ctx, "pf_chunk_experts", cfg[ci][1]);
            enum { IT = 6 }; double wall[IT], gpu[IT], enc[IT], cp[IT];
            for (int it = -2; it < IT; it++) {
                double t = now_ms();
                if (exl3m_moe_prefill(ctx, xs, nt, n, ei, ri, w, bptr, y)) { printf("ERR %s\n", exl3m_last_error()); exit(1); }
                double d = now_ms() - t; exl3m_stats s; exl3m_get_stats(ctx, &s);
                if (it >= 0) { wall[it] = d; gpu[it] = s.gpu_ms_last; enc[it] = s.encode_ms_last; cp[it] = s.copy_ms_last; }
            }
            St sw = stat(wall, IT), sg = stat(gpu, IT), se = stat(enc, IT), sc = stat(cp, IT);
            double flop = 2.0 * (double)n * 3.0 * H * FF;
            printf("%5d %6ld (%d,%d) | %7.1f %7.1f | %7.1f %5.1f %5.1f | %.2f | %.0f\n", nt, n, cfg[ci][0], cfg[ci][1], sw.min, sw.med, sg.med, se.med, sc.med,
                   flop / (sw.med * 1e9), nt / (48 * sw.med / 1e3));
        }
    }
    exl3m_set_param(ctx, "pf_chunk_rows", 4096); exl3m_set_param(ctx, "pf_chunk_experts", 16);
    if (getenv("PROFILE")) {
        int nt = ntoks[ntok_list_len - 1]; long n2 = 0; srand(7);
        for (int r = 0; r < nt; r++) { int pick[10]; for (int k = 0; k < 10; k++) { int e, dup; do { e = rand() % NB; dup = 0; for (int q = 0; q < k; q++) dup |= pick[q] == e; } while (dup); pick[k] = e; ei[n2] = e; ri[n2] = r; w[n2] = 0.1f; n2++; } }
        const char *names[] = { "none", "rot", "decode_w(all 3)", "gemm_gu", "mid", "gemm_d", "comb" };
        int64_t masks[] = { 0, 1, 2 | 4 | 8 | 16, 32, 64 * 0 + 8, 1 << 4, 1 << 5 };
        // bits: 0 rot, 1 decode_w, 2 gemm_gu, 3 mid, 4 gemm_d, 5 comb
        int64_t m2[] = { 0, 1, 2, 4, 8, 16, 32, 2 | 4 | 16 };
        const char *n2s[] = { "baseline", "-rot", "-decode_w", "-gemm_gu", "-mid", "-gemm_d", "-comb", "-decode_w -gemm_gu -gemm_d (tables+rot+mid+comb only)" };
        (void)names; (void)masks;
        printf("PROFILE %d tokens (skip kernels, GPU ms med of 5):\n", nt);
        for (int i = 0; i < 8; i++) { exl3m_set_param(ctx, "debug_skip_mask", m2[i]); double v[5];
            for (int it = -1; it < 5; it++) { exl3m_moe_prefill(ctx, xs, nt, n2, ei, ri, w, bptr, y); exl3m_stats s; exl3m_get_stats(ctx, &s); if (it >= 0) v[it] = s.gpu_ms_last; }
            printf("  %-60s %.1f ms\n", n2s[i], stat(v, 5).med); }
        exl3m_set_param(ctx, "debug_skip_mask", 0);
    }
    free(ei); free(ri); free(w); free(y);
}

// ---- concurrency: NEON threads busy on experts while Metal runs decode calls ------------------------------------------------------
typedef struct { atomic_int stop; atomic_long done; int n; } Neon;
static void *neon_loop(void *a) {
    void **p = a; Neon *s = p[0]; int id = (int)(intptr_t)p[1]; NeonBuf *nb = nbufs[id];
    int i = id * 17;
    while (!atomic_load(&s->stop)) { neon_expert(bptr[i % NB], xs, nb->y, nb); i += 29; atomic_fetch_add(&s->done, 1); }
    return NULL;
}
static double metal_loop(int m, int E, double secs, double *meanout, double *p95out) {
    static int32_t ei[512], ri[512]; static float w[512], y[8 * H]; double t0 = now_ms(), n = 0; static double lat[20000]; int k = 0;
    while (now_ms() - t0 < secs * 1e3 && k < 20000) { int np = mk_decode(m, E, (int)(n * 37) % NB, ei, ri, w); double t = now_ms(); exl3m_moe_rows(ctx, xs, m, np, ei, ri, w, bptr, y); lat[k++] = now_ms() - t; n++; }
    double s = 0; for (int i = 0; i < k; i++) s += lat[i]; *meanout = s / k; qsort(lat, (size_t)k, sizeof(double), cmpd); *p95out = lat[(int)(k * 0.95)];
    return n / ((now_ms() - t0) / 1e3);
}
static void bench_conc(void) {
    printf("\n== CONCURRENCY: Metal decode calls (m=1, E=13) vs NEON expert threads (1 row, whole expert per task), 2.0 s windows ==\n");
    int tn[] = { 4, 10 };
    double mm0, mp0; double mrate0 = metal_loop(1, 13, 2.0, &mm0, &mp0);
    printf("metal alone: %.0f calls/s, mean %.3f ms p95 %.3f ms\n", mrate0, mm0, mp0);
    for (int ti = 0; ti < 2; ti++) {
        int T = tn[ti];
        Neon s = { 0, 0, T }; pthread_t th[16]; void *args[16][2];
        double t0 = now_ms();
        for (int i = 0; i < T; i++) { args[i][0] = &s; args[i][1] = (void *)(intptr_t)i; pthread_create(&th[i], 0, neon_loop, args[i]); }
        usleep(2000000);
        atomic_store(&s.stop, 1); for (int i = 0; i < T; i++) pthread_join(th[i], 0);
        double neon_alone = (double)s.done / ((now_ms() - t0) / 1e3);
        Neon s2 = { 0, 0, T };
        t0 = now_ms();
        for (int i = 0; i < T; i++) { args[i][0] = &s2; args[i][1] = (void *)(intptr_t)i; pthread_create(&th[i], 0, neon_loop, args[i]); }
        double mm, mp; double mrate = metal_loop(1, 13, 2.0, &mm, &mp);
        atomic_store(&s2.stop, 1); for (int i = 0; i < T; i++) pthread_join(th[i], 0);
        double neon_conc = (double)s2.done / ((now_ms() - t0) / 1e3);
        printf("NEON %2d threads: alone %.0f experts/s | with Metal loop: %.0f experts/s (%+.1f%%) ; Metal while NEON: %.0f calls/s mean %.3f ms p95 %.3f ms (vs alone %.3f / %.3f: %+.1f%% mean)\n",
               T, neon_alone, neon_conc, 100.0 * (neon_conc / neon_alone - 1), mrate, mm, mp, mm0, mp0, 100.0 * (mm / mm0 - 1));
    }
}

// ---- split: one layer's E experts, Em on Metal (async submit) and the rest on NEON workers; wall of the whole call ---------------
typedef struct { int first, n; int T; } SplitArg;
static SplitArg sp;
static void neon_task(int id, void *arg) {
    SplitArg *a = arg;
    for (int i = id; i < a->n; i += a->T) neon_expert(bptr[(a->first + i * 7) % NB], xs, nbufs[id]->y, nbufs[id]);
}
static void bench_split(void) {
    printf("\n== SPLIT (m=1 row, one layer): E experts, Em on Metal (async) + E-Em on %d NEON workers (whole-expert tasks), wall ms per layer call ==\n", 9);
    printf("E  Em | NEON(E-Em) alone | Metal(Em) alone | split wall (min med mean p95)\n");
    Pool *pool = pool_new(9);
    static int32_t ei[64], ri[64]; static float w[64], y[H], y2[H];
    int Es[] = { 10, 13, 20 };
    for (int ei_ = 0; ei_ < 3; ei_++) {
        int E = Es[ei_];
        int ems[] = { 0, E / 4, E / 3, E / 2, (2 * E) / 3, E };
        for (int mi = 0; mi < 6; mi++) {
            int Em = ems[mi], En = E - Em; enum { IT = 200 }; double wall[IT], na[IT], ma[IT];
            for (int it = -10; it < IT; it++) {
                int first = ((it + 10) * 53 + 11 + 1000) % NB;
                int np = 0; for (int i = 0; i < Em; i++) { ei[np] = (first + (En + i) * 7) % NB; ri[np] = 0; w[np++] = 0.1f; }
                double t = now_ms();
                exl3m_job *job = NULL;
                if (Em) exl3m_moe_rows_submit(ctx, xs, 1, np, ei, ri, w, bptr, y, &job);
                sp.first = first; sp.n = En; sp.T = 9;
                double tn0 = now_ms();
                if (En) { pool_start(pool, neon_task, &sp); pool_wait(pool); }
                double tn1 = now_ms();
                if (job) exl3m_wait(ctx, job);
                double t2 = now_ms();
                if (it >= 0) { wall[it] = t2 - t; na[it] = tn1 - tn0; ma[it] = 0; }
            }
            St s = stat(wall, IT), sn = stat(na, IT);
            // Metal alone for Em experts
            double mt = 0;
            if (Em) { double v[IT]; for (int it = -10; it < IT; it++) { int first = ((it + 10) * 53 + 11 + 1000) % NB; int np = 0; for (int i = 0; i < Em; i++) { ei[np] = (first + (En + i) * 7) % NB; ri[np] = 0; w[np++] = 0.1f; }
                double t = now_ms(); exl3m_moe_rows(ctx, xs, 1, np, ei, ri, w, bptr, y2); if (it >= 0) v[it] = now_ms() - t; } mt = stat(v, IT).med; }
            printf("%2d %2d | %6.3f | %6.3f | %6.3f %6.3f %6.3f %6.3f\n", E, Em, En ? sn.med : 0.0, mt, s.min, s.med, s.mean, s.p95);
        }
    }
}

// ---- mmap-registered file region: cold vs warm ---------------------------------------------------------------------------------------
static void bench_mmap(const char *packpath) {
    printf("\n== MMAP: NoCopy registration of a file-backed mmap of experts.bin (layer 30, 24 blobs), first (cold or cached) vs repeat ==\n");
    int fd = open(packpath, O_RDONLY);
    size_t len = 24 * (size_t)BLOB; off_t off = (off_t)(30 * LAYER) & ~(off_t)16383; size_t adj = (size_t)(30 * LAYER - (uint64_t)off);
    uint8_t *mm = mmap(0, len + adj + 16384, PROT_READ, MAP_SHARED, fd, off);
    if (mm == MAP_FAILED) { perror("mmap"); return; }
    double t = now_ms();
    int64_t r = exl3m_register(ctx, mm, len + adj);
    printf("register file-backed mmap %.1f MB: %lld, %.2f ms (%s)\n", (double)len / 1e6, (long long)r, now_ms() - t, r < 0 ? exl3m_last_error() : "ok");
    if (r < 0) return;
    // drop file cache for that range: not possible without privileges (purge); report as observed
    const void *bp[24]; for (int i = 0; i < 24; i++) bp[i] = mm + adj + (size_t)i * BLOB;
    static int32_t ei[64], ri[64]; static float w[64], y[H];
    for (int i = 0; i < 12; i++) { ei[i] = i; ri[i] = 0; w[i] = 0.1f; }
    for (int rep = 0; rep < 6; rep++) { double a = now_ms(); int rc = exl3m_moe_rows(ctx, xs, 1, 12, ei, ri, w, bp, y); printf("  call %d: %.3f ms rc=%d (blobs mapped %s)\n", rep, now_ms() - a, rc, rc ? exl3m_last_error() : ""); }
    exl3m_stats s; exl3m_get_stats(ctx, &s); printf("  stats: mapped %llu staged %llu\n", (unsigned long long)s.blobs_mapped, (unsigned long long)s.blobs_staged);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 256; const char *sec = argc > 2 ? argv[2] : "all";
    char pack_def[1024], cpu_def[1024];
    snprintf(pack_def, sizeof pack_def, "%s/exl3/pack/experts.bin", getenv("HOME") ? getenv("HOME") : ".");
    snprintf(cpu_def, sizeof cpu_def, "%s/exl3/cpu/libexl3cpu.dylib", getenv("HOME") ? getenv("HOME") : ".");
    const char *pack = getenv("PACK") ? getenv("PACK") : pack_def;
    void *cpu = dlopen(getenv("CPULIB") ? getenv("CPULIB") : cpu_def, RTLD_NOW);
    if (!cpu) { fprintf(stderr, "dlopen cpu: %s\n", dlerror()); return 1; }
    cpu_had_in = dlsym(cpu, "exl3_cpu_had_in"); cpu_gemv = dlsym(cpu, "exl3_cpu_gemv");
    ctx = exl3m_create(4, 2);
    if (!ctx) { fprintf(stderr, "create: %s\n", exl3m_last_error()); return 1; }
    printf("device: %s\n", exl3m_device_name(ctx));
    for (int i = 0; i < 16; i++) nbufs[i] = aligned_alloc(64, (sizeof(NeonBuf) + 63) & ~(size_t)63);
    xs = malloc((size_t)16384 * H * 4); srand(3); for (size_t i = 0; i < (size_t)16384 * H; i++) xs[i] = ((float)rand() / RAND_MAX - 0.5f) * 3.f;
    load_pool(n, pack);
    int all = !strcmp(sec, "all");
    if (all || !strcmp(sec, "decode")) bench_decode();
    if (all || !strcmp(sec, "prefill")) { int nt[] = { 2048, 8192 }; bench_prefill(2, nt); }
    if (all || !strcmp(sec, "conc")) bench_conc();
    if (all || !strcmp(sec, "split")) bench_split();
    if (all || !strcmp(sec, "mmap")) bench_mmap(pack);
    exl3m_destroy(ctx);
    return 0;
}
