// M2: EXL3 (exllamav3 v1.x) trellis decode + GEMV on the Apple M4 CPU (NEON, fp16 arithmetic, FMLAL, UDOT).
// Format and bit conventions: exl3_ref.py (trellis uint16 [k/16, n/16, 16K], 256 elements per tile, tensor-core order).
//
//   exl3_cpu_decode    trellis tiles -> fp16 values in stream order, bit-identical to exl3_ref codebook_lut[decode_states]
//   exl3_cpu_had_in    A fp16 [m, k] * suh -> fp32 FWHT per 128 -> * r_scale -> fp16, bit-identical to exl3_ref.had_in
//   exl3_cpu_gemv      C fp32 [m, n] = had_out(A_had @ W_inner) * svh  (exl3_gemv_kernel semantics, fp32 accumulation)
//
// Decode: the 16-bit window of element t ends at bit (t+1)K of the tile's MSB-first stream (circular). With the stream
// byte-reversed per 32-bit word into a buffer that starts with the stream's last 2 bytes, element t starts at bit (t+1)K,
// so group g (elements 8g..8g+7) sits at byte g*K with the same in-group layout for every g: one 16-byte load, two TBL
// byte gathers (24-bit big-endian windows into u32 lanes), one USHL + AND each.
// GEMV: a tensor-core tile's stream element 8t+j (t = 4c+q) is row (2q, 2q+1, 2q+8, 2q+9)[j%4] of column c + 8*(j/4),
// so each decoded f16x8 group feeds column c with its low half and column c+8 with its high half against the same four
// inputs: FMLAL low/high into one f32x4 accumulator per column; the row order inside a column is irrelevant to the sum.
// Build (mini): clang -O3 -mcpu=apple-m4 -ffp-contract=off -shared -o libexl3cpu.dylib exl3_cpu.c
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <string.h>

enum { CB_3INST = 0, CB_MCG = 1, CB_MUL1 = 2 };

typedef struct { uint8x16_t idx0, idx1; int32x4_t sh0, sh1; } Layout;

static Layout layout(int K) {
    uint8_t idx[2][16]; int32_t sh[2][4];
    for (int j = 0; j < 8; j++) {
        int s = (j + 1) * K, b = s >> 3, off = s & 7;
        uint8_t *p = idx[j / 4] + 4 * (j % 4);
        p[0] = (uint8_t)(b + 2); p[1] = (uint8_t)(b + 1); p[2] = (uint8_t)b; p[3] = 0xFF;   // TBL: 0xFF -> 0
        sh[j / 4][j % 4] = -(8 - off);
    }
    Layout L = { vld1q_u8(idx[0]), vld1q_u8(idx[1]), vld1q_s32(sh[0]), vld1q_s32(sh[1]) };
    return L;
}

// Tile (32K bytes) -> buf[0..1] = last 2 stream bytes, buf[2..] = stream MSB-first. buf holds >= 2 + 32K + 16 bytes.
static inline void stage(const uint8_t *src, int K, uint8_t *buf) {
    for (int i = 0; i < 2 * K; i++) vst1q_u8(buf + 2 + 16 * i, vrev32q_u8(vld1q_u8(src + 16 * i)));
    buf[0] = buf[32 * K]; buf[1] = buf[32 * K + 1];
}

static inline void states8(const uint8_t *buf, int g, int K, const Layout *L, uint32x4_t *s0, uint32x4_t *s1) {
    uint8x16_t blk = vld1q_u8(buf + g * K);
    uint32x4_t m = vdupq_n_u32(0xFFFF);
    *s0 = vandq_u32(vshlq_u32(vreinterpretq_u32_u8(vqtbl1q_u8(blk, L->idx0)), L->sh0), m);
    *s1 = vandq_u32(vshlq_u32(vreinterpretq_u32_u8(vqtbl1q_u8(blk, L->idx1)), L->sh1), m);
}

// codebook.cuh: MUL1 = hfma2(0x6400 + bytesum(s * 0x83DCD12D), 0x1EEE, 0xC931); 3INST/MCG = lop3 then half lo + half hi.
static inline float16x8_t codebook(uint32x4_t s0, uint32x4_t s1, int cb) {
    if (cb == CB_MUL1) {
        uint32x4_t mul = vdupq_n_u32(0x83DCD12Du), z = vdupq_n_u32(0);
        uint8x16_t one = vdupq_n_u8(1);
        uint32x4_t b0 = vdotq_u32(z, vreinterpretq_u8_u32(vmulq_u32(s0, mul)), one);
        uint32x4_t b1 = vdotq_u32(z, vreinterpretq_u8_u32(vmulq_u32(s1, mul)), one);
        uint16x8_t b = vuzp1q_u16(vreinterpretq_u16_u32(b0), vreinterpretq_u16_u32(b1));
        float16x8_t h = vreinterpretq_f16_u16(vaddq_u16(b, vdupq_n_u16(0x6400)));
        return vfmaq_f16(vreinterpretq_f16_u16(vdupq_n_u16(0xC931)), h, vreinterpretq_f16_u16(vdupq_n_u16(0x1EEE)));
    }
    uint32x4_t mul = vdupq_n_u32(cb == CB_MCG ? 0xCBAC1FEDu : 89226354u), add = vdupq_n_u32(cb == CB_MCG ? 0 : 64248484u);
    uint32x4_t mask = vdupq_n_u32(0x8FFF8FFFu), x0r = vdupq_n_u32(0x3B603B60u);
    uint32x4_t x0 = veorq_u32(vandq_u32(vmlaq_u32(add, s0, mul), mask), x0r);
    uint32x4_t x1 = veorq_u32(vandq_u32(vmlaq_u32(add, s1, mul), mask), x0r);
    uint16x8_t lo = vuzp1q_u16(vreinterpretq_u16_u32(x0), vreinterpretq_u16_u32(x1));
    uint16x8_t hi = vuzp2q_u16(vreinterpretq_u16_u32(x0), vreinterpretq_u16_u32(x1));
    return vaddq_f16(vreinterpretq_f16_u16(lo), vreinterpretq_f16_u16(hi));
}

static int bad_args(int K, int cb) { return K < 1 || K > 8 || cb < 0 || cb > 2; }

int exl3_cpu_decode(const uint16_t *trellis, int64_t ntiles, int K, int cb, uint16_t *out) {
    if (bad_args(K, cb)) return -1;
    Layout L = layout(K);
    uint8_t buf[2 + 256 + 16];
    for (int64_t i = 0; i < ntiles; i++) {
        stage((const uint8_t *)(trellis + i * 16 * K), K, buf);
        for (int g = 0; g < 32; g++) {
            uint32x4_t s0, s1;
            states8(buf, g, K, &L, &s0, &s1);
            vst1q_f16((__fp16 *)(out + i * 256 + 8 * g), codebook(s0, s1, cb));
        }
    }
    return 0;
}

static const float R_SCALE = 0.088388347648f;

static void fwht128(float *v) {
    for (int st = 1; st < 128; st <<= 1)
        for (int b = 0; b < 128; b += 2 * st)
            for (int i = b; i < b + st; i++) { float a = v[i], c = v[i + st]; v[i] = a + c; v[i + st] = a - c; }
}

void exl3_cpu_had_in(const uint16_t *A, const uint16_t *suh, int m, int k, uint16_t *out) {
    const __fp16 *a = (const __fp16 *)A, *s = (const __fp16 *)suh;
    __fp16 *o = (__fp16 *)out;
    float v[128];
    for (int r = 0; r < m; r++)
        for (int b = 0; b < k; b += 128) {
            for (int i = 0; i < 128; i++) v[i] = (float)(__fp16)((float)a[(int64_t)r * k + b + i] * (float)s[b + i]);
            fwht128(v);
            for (int i = 0; i < 128; i++) o[(int64_t)r * k + b + i] = (__fp16)(v[i] * R_SCALE);
        }
}

#define MAX_M 8

typedef struct {
    const uint16_t *trellis; const float16x8_t *xq; float *y;   // xq [m][k/16][4]; y [m][n] (inner, before had_out)
    int m, k, n, K, cb, tiles_per_job; Layout L;
} GemvJob;

// Column tiles [j0, j1): y[:, 16j..16j+15] = A_had @ W_inner over all k.
static void gemv_cols(const GemvJob *J, int j0, int j1) {
    const int K = J->K, cb = J->cb, kt = J->k / 16, nt = J->n / 16, m = J->m;
    uint8_t buf[2 + 256 + 16];
    for (int j = j0; j < j1; j++) {
        float32x4_t acc[MAX_M][16];
        for (int r = 0; r < m; r++) for (int c = 0; c < 16; c++) acc[r][c] = vdupq_n_f32(0);
        for (int i = 0; i < kt; i++) {
            stage((const uint8_t *)(J->trellis + ((int64_t)i * nt + j) * 16 * K), K, buf);
            if (m == 1) {
                const float16x8_t *x = J->xq + (int64_t)i * 4;
                for (int c = 0; c < 8; c++)
                    for (int q = 0; q < 4; q++) {
                        uint32x4_t s0, s1;
                        states8(buf, 4 * c + q, K, &J->L, &s0, &s1);
                        float16x8_t w = codebook(s0, s1, cb);
                        acc[0][c] = vfmlalq_low_f16(acc[0][c], w, x[q]);
                        acc[0][c + 8] = vfmlalq_high_f16(acc[0][c + 8], w, x[q]);
                    }
            } else {
                float16x8_t w[32];
                for (int g = 0; g < 32; g++) { uint32x4_t s0, s1; states8(buf, g, K, &J->L, &s0, &s1); w[g] = codebook(s0, s1, cb); }
                for (int r = 0; r < m; r++) {
                    const float16x8_t *x = J->xq + ((int64_t)r * kt + i) * 4;
                    for (int c = 0; c < 8; c++)
                        for (int q = 0; q < 4; q++) {
                            acc[r][c] = vfmlalq_low_f16(acc[r][c], w[4 * c + q], x[q]);
                            acc[r][c + 8] = vfmlalq_high_f16(acc[r][c + 8], w[4 * c + q], x[q]);
                        }
                }
            }
        }
        for (int r = 0; r < m; r++)
            for (int c = 0; c < 16; c++) J->y[(int64_t)r * J->n + 16 * j + c] = vaddvq_f32(acc[r][c]);
    }
}

// C fp32 [m, n] = had_out(A_had @ W_inner): H128 per 128 output columns, / sqrt(128), * svh. m <= 8, k % 16 == 0,
// n % 128 == 0. y_scratch: m * n floats, xq_scratch: m * k * 2 halves (16-byte aligned). nthreads <= 1: calling thread.
int exl3_cpu_gemv(const uint16_t *A_had, const uint16_t *trellis, const uint16_t *svh, int m, int k, int n, int K, int cb,
                  float *C, float *y_scratch, uint16_t *xq_scratch, int nthreads) {
    if (bad_args(K, cb) || m < 1 || m > MAX_M || k % 16 || n % 128) return -1;
    const __fp16 *a = (const __fp16 *)A_had;
    __fp16 *xq = (__fp16 *)xq_scratch;
    for (int r = 0; r < m; r++)                       // xq[r][i][q] = (x[2q], x[2q+1], x[2q+8], x[2q+9]) x 2
        for (int i = 0; i < k / 16; i++)
            for (int q = 0; q < 4; q++) {
                const __fp16 *x = a + (int64_t)r * k + 16 * i;
                __fp16 *d = xq + (((int64_t)r * (k / 16) + i) * 4 + q) * 8;
                d[0] = d[4] = x[2 * q]; d[1] = d[5] = x[2 * q + 1]; d[2] = d[6] = x[2 * q + 8]; d[3] = d[7] = x[2 * q + 9];
            }
    GemvJob J = { trellis, (const float16x8_t *)xq, y_scratch, m, k, n, K, cb, 0, layout(K) };
    int nt = n / 16;
    if (nthreads <= 1) gemv_cols(&J, 0, nt);
    else {
        int per = (nt + nthreads - 1) / nthreads;
        GemvJob *pj = &J;
        dispatch_apply((size_t)nthreads, DISPATCH_APPLY_AUTO, ^(size_t t) {
            int j0 = (int)t * per, j1 = j0 + per < nt ? j0 + per : nt;
            if (j0 < j1) gemv_cols(pj, j0, j1);
        });
    }
    const __fp16 *sv = (const __fp16 *)svh;
    const float inv = 0.08838834764831845f;
    for (int r = 0; r < m; r++)
        for (int b = 0; b < n; b += 128) {
            float *v = y_scratch + (int64_t)r * n + b;
            fwht128(v);
            for (int i = 0; i < 128; i++) C[(int64_t)r * n + b + i] = v[i] * inv * (float)sv[b + i];
        }
    return 0;
}

// M4 CPU tier (Strata ExpertPool): the same GEMV split into 128-column blocks so one expert's projection runs on several
// threads. exl3_cpu_prep_xq builds the permuted input once per (expert, projection); exl3_cpu_gemv_blocks then computes
// output columns [128*b0, 128*b1) of C (row stride n); blocks are independent, so threads may share C. Bitwise equal
// to exl3_cpu_gemv's columns: same per-tile order, same fp32 reduction, same had_out.
void exl3_cpu_prep_xq(const uint16_t *A_had, int m, int k, uint16_t *xq_scratch) {
    const __fp16 *a = (const __fp16 *)A_had;
    __fp16 *xq = (__fp16 *)xq_scratch;
    for (int r = 0; r < m; r++)
        for (int i = 0; i < k / 16; i++)
            for (int q = 0; q < 4; q++) {
                const __fp16 *x = a + (int64_t)r * k + 16 * i;
                __fp16 *d = xq + (((int64_t)r * (k / 16) + i) * 4 + q) * 8;
                d[0] = d[4] = x[2 * q]; d[1] = d[5] = x[2 * q + 1]; d[2] = d[6] = x[2 * q + 8]; d[3] = d[7] = x[2 * q + 9];
            }
}

int exl3_cpu_gemv_blocks(const uint16_t *xq_scratch, const uint16_t *trellis, const uint16_t *svh, int m, int k, int n,
                         int K, int cb, int b0, int b1, float *C, float *y_scratch) {
    if (bad_args(K, cb) || m < 1 || m > MAX_M || k % 16 || n % 128 || b0 < 0 || b1 > n / 128 || b0 >= b1) return -1;
    GemvJob J = { trellis, (const float16x8_t *)xq_scratch, NULL, m, k, n, K, cb, 0, layout(K) };
    float tmp[MAX_M * 128];
    for (int b = b0; b < b1; b++) {
        // one 128-column block: 8 tiles of 16 columns, written into tmp with row stride 128
        const int kt = k / 16, nt = n / 16;
        uint8_t buf[2 + 256 + 16];
        for (int j = 8 * b; j < 8 * b + 8; j++) {
            float32x4_t acc[MAX_M][16];
            for (int r = 0; r < m; r++) for (int c = 0; c < 16; c++) acc[r][c] = vdupq_n_f32(0);
            for (int i = 0; i < kt; i++) {
                stage((const uint8_t *)(J.trellis + ((int64_t)i * nt + j) * 16 * K), K, buf);
                if (m == 1) {
                    const float16x8_t *x = J.xq + (int64_t)i * 4;
                    for (int c = 0; c < 8; c++)
                        for (int q = 0; q < 4; q++) {
                            uint32x4_t s0, s1;
                            states8(buf, 4 * c + q, K, &J.L, &s0, &s1);
                            float16x8_t wv = codebook(s0, s1, cb);
                            acc[0][c] = vfmlalq_low_f16(acc[0][c], wv, x[q]);
                            acc[0][c + 8] = vfmlalq_high_f16(acc[0][c + 8], wv, x[q]);
                        }
                } else {
                    float16x8_t wv[32];
                    for (int g = 0; g < 32; g++) { uint32x4_t s0, s1; states8(buf, g, K, &J.L, &s0, &s1); wv[g] = codebook(s0, s1, cb); }
                    for (int r = 0; r < m; r++) {
                        const float16x8_t *x = J.xq + ((int64_t)r * kt + i) * 4;
                        for (int c = 0; c < 8; c++)
                            for (int q = 0; q < 4; q++) {
                                acc[r][c] = vfmlalq_low_f16(acc[r][c], wv[4 * c + q], x[q]);
                                acc[r][c + 8] = vfmlalq_high_f16(acc[r][c + 8], wv[4 * c + q], x[q]);
                            }
                    }
                }
            }
            for (int r = 0; r < m; r++)
                for (int c = 0; c < 16; c++) tmp[r * 128 + 16 * (j - 8 * b) + c] = vaddvq_f32(acc[r][c]);
        }
        const __fp16 *sv = (const __fp16 *)svh;
        const float inv = 0.08838834764831845f;
        for (int r = 0; r < m; r++) {
            float *v = tmp + r * 128;
            fwht128(v);
            for (int i = 0; i < 128; i++) C[(int64_t)r * n + 128 * b + i] = v[i] * inv * (float)sv[128 * b + i];
        }
    }
    (void)y_scratch;   // reserved; each block reduces in a stack buffer
    return 0;
}
